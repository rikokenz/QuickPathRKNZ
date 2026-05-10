bl_info = {
    "name": "QuickPath RKNZ",
    "author": "Rikokensfw",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > QuickPath RKNZ",
    "description": "Lightweight realtime motion path visualization for bones and objects | RKNZ",
    "category": "Animation",
}

import bpy
import gpu
import uuid
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix, Quaternion, Euler

# ─── Global State ─────────────────────────────────────────────────────────────
_draw_handler = None
_path_cache: dict = {}  # { uid: { frame: Vector } }

# ─── Core Math / Sampling ─────────────────────────────────────────────────────

def get_bone_world_pos(arm, bone_name):
    if arm is None or arm.type != 'ARMATURE': return None
    pb = arm.pose.bones.get(bone_name)
    if pb is None: return None
    return (arm.matrix_world @ pb.matrix).translation.copy()

def build_fc_bone(obj, bone_name):
    if not obj.animation_data or not obj.animation_data.action: return None
    prefix = f'pose.bones["{bone_name}"].'
    m = {}
    for fc in obj.animation_data.action.fcurves:
        if fc.data_path.startswith(prefix):
            m[(fc.data_path[len(prefix):], fc.array_index)] = fc
    return m or None

def build_fc_obj(obj):
    if not obj.animation_data or not obj.animation_data.action: return None
    m = {}
    for fc in obj.animation_data.action.fcurves:
        m[(fc.data_path, fc.array_index)] = fc
    return m or None

def ev(fc_map, prop, count, frame, default):
    r = list(default)
    for i in range(count):
        fc = fc_map.get((prop, i))
        if fc: r[i] = fc.evaluate(frame)
    return r

def rot_mat(fc_map, rm, frame):
    if rm == 'QUATERNION':
        return Quaternion(ev(fc_map,'rotation_quaternion',4,frame,[1,0,0,0])).to_matrix().to_4x4()
    if rm == 'AXIS_ANGLE':
        aa = ev(fc_map,'rotation_axis_angle',4,frame,[0,0,1,0])
        return Quaternion(aa[1:4],aa[0]).to_matrix().to_4x4()
    return Euler(ev(fc_map,'rotation_euler',3,frame,[0,0,0]),rm).to_matrix().to_4x4()

def obj_world_mat(obj, frame):
    fc = build_fc_obj(obj)
    if not fc: return obj.matrix_world.copy()
    loc   = ev(fc,'location',3,frame,list(obj.location))
    scale = ev(fc,'scale',3,frame,list(obj.scale))
    return Matrix.Translation(loc) @ rot_mat(fc,obj.rotation_mode,frame) @ Matrix.Diagonal((*scale,1.0))

def bone_mat_from_action(arm, bone_name, frame):
    arm_data = arm.data
    eb = arm_data.bones.get(bone_name)
    if eb is None: return None
    chain = []
    b = eb
    while b: chain.append(b); b = b.parent
    chain.reverse()
    pose_mat = Matrix.Identity(4)
    for cb in chain:
        rest = cb.matrix_local.copy()
        rest_local = (cb.parent.matrix_local.inverted() @ rest) if cb.parent else rest
        fc = build_fc_bone(arm, cb.name) or {}
        pb = arm.pose.bones.get(cb.name)
        rm = pb.rotation_mode if pb else 'QUATERNION'
        loc   = ev(fc,'location',3,frame,[0,0,0])
        scale = ev(fc,'scale',3,frame,[1,1,1])
        lm = Matrix.Translation(loc) @ rot_mat(fc,rm,frame) @ Matrix.Diagonal((*scale,1.0))
        pose_mat = pose_mat @ rest_local @ lm
    return pose_mat

def bone_world_pos_fc(arm, bone_name, frame):
    m = bone_mat_from_action(arm, bone_name, frame)
    return (arm.matrix_world @ m).translation.copy() if m else None

FAST_C = {'COPY_LOCATION','COPY_ROTATION','COPY_TRANSFORMS','COPY_SCALE',
           'DAMPED_TRACK','TRACK_TO','LOCKED_TRACK','CHILD_OF'}

def tgt_world_mat(c, frame):
    t = c.target
    if t is None: return None
    if c.subtarget and t.type == 'ARMATURE':
        m = bone_mat_from_action(t, c.subtarget, frame)
        return (t.matrix_world @ m) if m else None
    return obj_world_mat(t, frame)

def apply_c(c, bm, frame):
    inf = c.influence
    if c.type == 'COPY_LOCATION':
        tgt = tgt_world_mat(c, frame)
        if tgt is None: return bm
        nl = bm.translation.copy()
        for i,(u,iv) in enumerate(zip([c.use_x,c.use_y,c.use_z],[c.invert_x,c.invert_y,c.invert_z])):
            if u:
                val = tgt.translation[i] * (-1 if iv else 1)
                nl[i] += (val - nl[i]) * inf
        r = bm.copy(); r.translation = nl; return r
    if c.type == 'COPY_TRANSFORMS':
        tgt = tgt_world_mat(c, frame)
        return bm.lerp(tgt, inf) if tgt else bm
    if c.type == 'CHILD_OF':
        tgt = tgt_world_mat(c, frame)
        if tgt is None: return bm
        io = c.inverse_matrix if hasattr(c,'inverse_matrix') else Matrix.Identity(4)
        return bm.lerp(tgt @ io @ bm, inf)
    return bm

def all_fast(arm, bn):
    pb = arm.pose.bones.get(bn)
    return True if pb is None else all(c.mute or c.type in FAST_C for c in pb.constraints)

def bone_pos_constrained(arm, bn, frame):
    pb = arm.pose.bones.get(bn)
    if pb is None: return None
    for c in pb.constraints:
        if not c.mute and c.type not in FAST_C: return None
    m = bone_mat_from_action(arm, bn, frame)
    if m is None: return None
    wm = arm.matrix_world @ m
    for c in pb.constraints:
        if not c.mute: wm = apply_c(c, wm, frame)
    return wm.translation.copy()

def sample_entry(scene, node):
    """Sample a single entry using its own frame range."""
    uid = node.uid
    _path_cache[uid] = {}
    fs = node.frame_start; fe = node.frame_end; fst = max(1, node.frame_step)

    if node.track_type == 'BONE':
        arm = bpy.data.objects.get(node.armature_name)
        bn  = node.bone_name
        if arm is None or not bn: return
        has_action     = arm.animation_data is not None and arm.animation_data.action is not None
        pb             = arm.pose.bones.get(bn)
        has_c          = pb and len(pb.constraints) > 0
        c_ok           = all_fast(arm, bn)
        if has_action and (not has_c or c_ok):
            for f in range(fs, fe+1, fst):
                pos = bone_pos_constrained(arm,bn,f) if has_c else bone_world_pos_fc(arm,bn,f)
                if pos: _path_cache[uid][f] = pos
        else:
            cur = scene.frame_current
            for f in range(fs, fe+1, fst):
                scene.frame_set(f)
                pos = get_bone_world_pos(arm, bn)
                if pos: _path_cache[uid][f] = pos
            scene.frame_set(cur)
    else:
        obj = bpy.data.objects.get(node.object_name)
        if obj is None: return
        has_action = obj.animation_data is not None and obj.animation_data.action is not None
        if has_action:
            for f in range(fs, fe+1, fst):
                _path_cache[uid][f] = obj_world_mat(obj,f).translation.copy()
        else:
            cur = scene.frame_current
            for f in range(fs, fe+1, fst):
                scene.frame_set(f)
                _path_cache[uid][f] = obj.matrix_world.translation.copy()
            scene.frame_set(cur)

def get_cached(uid):
    c = _path_cache.get(uid, {})
    if not c: return [], []
    frames = sorted(c.keys())
    return frames, [c[f] for f in frames]

# ─── Tree Helpers ─────────────────────────────────────────────────────────────

def make_uid(): return uuid.uuid4().hex[:12]

def get_node(props, uid):
    for n in props.nodes:
        if n.uid == uid: return n
    return None

def get_children(props, parent_uid):
    return [n for n in props.nodes if n.parent_uid == parent_uid]

def get_depth(props, node):
    d = 0; uid = node.parent_uid
    while uid:
        p = get_node(props, uid)
        if p is None: break
        d += 1; uid = p.parent_uid
    return d

def all_entry_descendants(props, col_uid):
    r = []
    for c in get_children(props, col_uid):
        if c.item_type == 'ENTRY': r.append(c)
        else: r.extend(all_entry_descendants(props, c.uid))
    return r

def parent_uid_for_new(props):
    a = get_node(props, props.active_uid)
    if a and a.item_type == 'COLLECTION': return a.uid
    if a: return a.parent_uid
    return ''

def next_order(props, parent_uid):
    s = [n for n in props.nodes if n.parent_uid == parent_uid]
    return max((n.order for n in s), default=-1) + 1

def node_matches(node, ftext):
    if not ftext: return True
    return ftext.lower() in (node.label or "").lower()

def col_has_match(props, uid, ftext):
    for c in get_children(props, uid):
        if c.item_type == 'ENTRY' and node_matches(c, ftext): return True
        if c.item_type == 'COLLECTION' and col_has_match(props, c.uid, ftext): return True
    return False

def build_draw_order(props):
    ftext   = props.filter_text.strip()
    finvert = props.filter_invert
    sort    = props.sort_mode  # 'DEFAULT','AZ','ZA'
    result  = []

    def sort_key(n):
        return (n.label or "").lower()

    def walk(parent_uid):
        children = [n for n in props.nodes if n.parent_uid == parent_uid]
        if sort == 'DEFAULT':
            children.sort(key=lambda n: n.order)
        elif sort == 'AZ':
            cols = sorted([n for n in children if n.item_type=='COLLECTION'], key=sort_key)
            ents = sorted([n for n in children if n.item_type=='ENTRY'],      key=sort_key)
            children = cols + ents
        else:  # ZA
            cols = sorted([n for n in children if n.item_type=='COLLECTION'], key=sort_key, reverse=True)
            ents = sorted([n for n in children if n.item_type=='ENTRY'],      key=sort_key, reverse=True)
            children = cols + ents

        for node in children:
            if ftext:
                if node.item_type == 'ENTRY':
                    m = node_matches(node, ftext)
                    if finvert: m = not m
                    if not m: continue
                else:
                    m = col_has_match(props, node.uid, ftext)
                    if finvert: m = not m
                    if not m: continue
            result.append(node)
            if node.item_type == 'COLLECTION' and not node.collapsed:
                walk(node.uid)

    walk('')
    return result

def checked_entries(props):
    """All ENTRY nodes with calc_selected=True."""
    return [n for n in props.nodes if n.item_type == 'ENTRY' and n.calc_selected]

# ─── GPU Drawing ──────────────────────────────────────────────────────────────

def brighten(c, f=1.5):
    return (min(1.0,c[0]*f), min(1.0,c[1]*f), min(1.0,c[2]*f), c[3])

def desaturate(c, amt=0.5):
    lum = 0.299*c[0] + 0.587*c[1] + 0.114*c[2]
    return (c[0]+(lum-c[0])*amt, c[1]+(lum-c[1])*amt, c[2]+(lum-c[2])*amt, c[3])

def is_glow(props, node):
    if not props.active_uid: return False
    if node.uid == props.active_uid: return True
    a = get_node(props, props.active_uid)
    if a and a.item_type == 'COLLECTION':
        uid = node.parent_uid
        while uid:
            if uid == a.uid: return True
            p = get_node(props, uid)
            if p is None: break
            uid = p.parent_uid
    return False

def draw_paths():
    try:
        scene = bpy.context.scene
        props = scene.bmpl_props
    except Exception: return
    if not props.show_path: return

    cf  = scene.frame_current
    reg = bpy.context.region
    vp  = (reg.width, reg.height) if reg else (1920,1080)

    def line(pts, col, w):
        if len(pts) < 2: return
        sh = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        bt = batch_for_shader(sh,'LINE_STRIP',{"pos":[p.to_tuple() for p in pts]})
        sh.bind(); sh.uniform_float("color",tuple(col))
        sh.uniform_float("lineWidth",w); sh.uniform_float("viewportSize",vp)
        bt.draw(sh)

    def dots(pts, col, sz):
        if not pts: return
        sh = gpu.shader.from_builtin('UNIFORM_COLOR')
        bt = batch_for_shader(sh,'POINTS',{"pos":[p.to_tuple() for p in pts]})
        gpu.state.point_size_set(sz)
        sh.bind(); sh.uniform_float("color",tuple(col)); bt.draw(sh)

    def glow(pts, col, bw):
        base = max(bw, 3.0)
        gc = desaturate(col, 0.5)
        for wm, am in [(8,0.04),(5,0.07),(3,0.12),(1.8,0.20)]:
            line(pts,(gc[0],gc[1],gc[2],gc[3]*am), base*wm)

    def outline(pts, w):
        line(pts,(1,1,1,0.6),w+2.0)

    for node in props.nodes:
        if node.item_type != 'ENTRY' or not node.enabled: continue
        frames, positions = get_cached(node.uid)
        if len(positions) < 2: continue

        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('ALWAYS' if node.in_front else 'LESS_EQUAL')

        glowing = is_glow(props, node)
        bright  = node.calc_selected

        def mc(c): return brighten(c) if bright else c

        if node.use_frame_colors:
            split = len(frames)
            for i,f in enumerate(frames):
                if f >= cf: split = i; break
            bef = positions[:split+1]; aft = positions[split:]
            cb = mc(node.color_before); ca = mc(node.color_after)
            if glowing: glow(bef,cb,node.line_width); glow(aft,ca,node.line_width)
            if bright:  outline(bef,node.line_width); outline(aft,node.line_width)
            line(bef,cb,node.line_width); line(aft,ca,node.line_width)
            if props.show_dots:
                dots(bef,(cb[0],cb[1],cb[2],cb[3]*.7),node.dot_size)
                dots(aft,(ca[0],ca[1],ca[2],ca[3]*.7),node.dot_size)
        else:
            c = mc(node.path_color)
            if glowing: glow(positions,c,node.line_width)
            if bright:  outline(positions,node.line_width)
            line(positions,c,node.line_width)
            if props.show_dots:
                dots(positions,(c[0],c[1],c[2],c[3]*.7),node.dot_size)

        if cf in frames:
            dots([positions[frames.index(cf)]],(1,1,1,1),node.dot_size*2.5)

        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')

# ─── Handlers ─────────────────────────────────────────────────────────────────

def do_resample(scene):
    props = scene.bmpl_props
    if not props.live_update: return
    for node in props.nodes:
        if node.item_type == 'ENTRY' and node.enabled:
            sample_entry(scene, node)
    for wm in bpy.data.window_managers:
        for win in wm.windows:
            if win.screen:
                for area in win.screen.areas:
                    if area.type == 'VIEW_3D': area.tag_redraw()

def cleanup_orphans(scene):
    props = scene.bmpl_props
    to_rm = []
    for i,n in enumerate(props.nodes):
        if n.item_type != 'ENTRY': continue
        obj_name = n.armature_name if n.track_type == 'BONE' else n.object_name
        if bpy.data.objects.get(obj_name) is None: to_rm.append(i)
    for i in reversed(to_rm):
        _path_cache.pop(props.nodes[i].uid, None)
        props.nodes.remove(i)
    if props.active_uid not in [n.uid for n in props.nodes]:
        props.active_uid = ''

# ─── Live update: depsgraph dirty-flag + timer debounce ──────────────────────
#
# Strategy: on_depsgraph watches for ANY update on objects/armatures/actions
# that are used by tracked entries (direct + constraint sources). When a match
# is found it sets _resample_pending = True. A persistent timer then calls
# do_resample on the next Blender tick, safely outside the depsgraph callback.
# This is the most reliable approach in Blender 3.6 because:
#   - keyframe insert fires depsgraph with ACTION or OBJECT id_type (varies)
#   - moving a key fires OBJECT/ARMATURE (pose re-evaluated)
#   - msgbus does NOT reliably fire for keyframe_points edits in 3.6

_resample_pending = False

def _get_tracked_ids(scene):
    """Return a set of (python_type_name, datablock_name) for all objects/actions
    relevant to tracked entries. Keys match type(u.id).__name__ from depsgraph."""
    ids = set()
    props = scene.bmpl_props
    for node in props.nodes:
        if node.item_type != 'ENTRY' or not node.enabled: continue
        actions = _collect_actions_for_node(node)
        for action in actions:
            ids.add(('Action', action.name))
        if node.track_type == 'BONE':
            obj = bpy.data.objects.get(node.armature_name)
            if obj: ids.add(('Object', obj.name))
        else:
            obj = bpy.data.objects.get(node.object_name)
            if obj: ids.add(('Object', obj.name))
    return ids

def _resample_timer():
    """Timer callback: do the actual resample when pending, then unregister."""
    global _resample_pending
    if not _resample_pending:
        return None
    _resample_pending = False
    for scene in bpy.data.scenes:
        try:
            if not scene.bmpl_props.live_update: continue
        except AttributeError:
            continue
        cleanup_orphans(scene)
        do_resample(scene)
    return None

def refresh_subscriptions():
    """No-op kept for API compatibility — tracking is now depsgraph-based."""
    pass

@bpy.app.handlers.persistent
def on_depsgraph(scene, depsgraph):
    global _resample_pending

    try:
        live = scene.bmpl_props.live_update
    except AttributeError:
        live = False

    # DEBUG: print all updates so we can see what fires in Blender 3.6
    if live:
        print("[QuickPath] depsgraph fired")
        for u in depsgraph.updates:
            if u.id:
                print(f"  type={type(u.id).__name__!r}  name={u.id.name!r}")

    if not live: return
    if _resample_pending: return

    _resample_pending = True
    bpy.app.timers.register(_resample_timer, first_interval=0.0)

@bpy.app.handlers.persistent
def on_load(filepath):
    global _draw_handler, _resample_pending
    _path_cache.clear()
    _resample_pending = False
    if _draw_handler:
        try: bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        except: pass
        _draw_handler = None
    _draw_handler = bpy.types.SpaceView3D.draw_handler_add(draw_paths, (), 'WINDOW', 'POST_VIEW')
    if on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(on_depsgraph)
    for sc in bpy.data.scenes:
        if hasattr(sc, 'bmpl_props'):
            sc.bmpl_props.frame_start = sc.frame_start
            sc.bmpl_props.frame_end   = sc.frame_end

def on_live_toggle(self, context):
    if self.live_update:
        do_resample(context.scene)

def on_filter_update(self, context):
    """Live filter — redraw panel on every keystroke."""
    for area in context.screen.areas:
        if area.type == 'VIEW_3D': area.tag_redraw()

def register_handlers():
    global _draw_handler
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(draw_paths, (), 'WINDOW', 'POST_VIEW')
    if on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(on_depsgraph)
    if on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load)

def unregister_handlers():
    global _draw_handler, _resample_pending
    _resample_pending = False
    if _draw_handler:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None
    for h, lst in [(on_depsgraph, bpy.app.handlers.depsgraph_update_post),
                   (on_load,      bpy.app.handlers.load_post)]:
        if h in lst: lst.remove(h)

# ─── Property Groups ──────────────────────────────────────────────────────────

class QPRK_Node(bpy.types.PropertyGroup):
    uid:               bpy.props.StringProperty(default="")
    parent_uid:        bpy.props.StringProperty(default="")
    order:             bpy.props.IntProperty(default=0)
    item_type:         bpy.props.EnumProperty(
        items=[('COLLECTION','Collection',''),('ENTRY','Entry','')], default='ENTRY')
    label:             bpy.props.StringProperty(name="Name", default="")
    # Collection
    collapsed:         bpy.props.BoolProperty(default=False)
    col_calc_selected: bpy.props.BoolProperty(default=True)
    # Entry tracking
    track_type:        bpy.props.EnumProperty(
        items=[('BONE','Bone',''),('OBJECT','Object','')], default='BONE')
    armature_name:     bpy.props.StringProperty(default="")
    bone_name:         bpy.props.StringProperty(default="")
    object_name:       bpy.props.StringProperty(default="")
    # Entry per-entry frame range
    frame_start:       bpy.props.IntProperty(name="Start", default=1)
    frame_end:         bpy.props.IntProperty(name="End",   default=250)
    frame_step:        bpy.props.IntProperty(name="Step",  default=1, min=1, max=10)
    # Entry visual
    enabled:           bpy.props.BoolProperty(default=True)
    in_front:          bpy.props.BoolProperty(default=False)
    calc_selected:     bpy.props.BoolProperty(default=True)
    path_color:        bpy.props.FloatVectorProperty(
        subtype='COLOR_GAMMA', size=4, min=0, max=1, default=(1.0,0.5,0.0,1.0))
    use_frame_colors:  bpy.props.BoolProperty(default=False)
    color_before:      bpy.props.FloatVectorProperty(
        subtype='COLOR_GAMMA', size=4, min=0, max=1, default=(0.8,0.1,0.1,1.0))
    color_after:       bpy.props.FloatVectorProperty(
        subtype='COLOR_GAMMA', size=4, min=0, max=1, default=(0.1,0.8,0.2,1.0))
    line_width:        bpy.props.FloatProperty(default=2.0, min=0.5, max=10.0)
    dot_size:          bpy.props.FloatProperty(default=4.0, min=1.0, max=20.0)


class QPRK_SceneProps(bpy.types.PropertyGroup):
    nodes:          bpy.props.CollectionProperty(type=QPRK_Node)
    active_uid:     bpy.props.StringProperty(default="")
    # Global frame range (override)
    frame_start:    bpy.props.IntProperty(name="Start", default=1)
    frame_end:      bpy.props.IntProperty(name="End",   default=250)
    frame_step:     bpy.props.IntProperty(name="Step",  default=1, min=1, max=10)
    show_path:      bpy.props.BoolProperty(name="Show Paths", default=True)
    show_dots:      bpy.props.BoolProperty(name="Show Dots",  default=True)
    live_update:    bpy.props.BoolProperty(
        name="Auto Update on Keyframe", default=False, update=on_live_toggle)
    filter_text:    bpy.props.StringProperty(
        name="Filter", default="", update=on_filter_update)
    filter_invert:  bpy.props.BoolProperty(name="Invert", default=False)
    sort_mode:      bpy.props.EnumProperty(
        name="Sort", default='DEFAULT',
        items=[('DEFAULT','Default','Manual order'),
               ('AZ','A→Z','Alphabetical ascending'),
               ('ZA','Z→A','Alphabetical descending')])
    reparent_target: bpy.props.StringProperty(default="")

# ─── Operators ────────────────────────────────────────────────────────────────

class QPRK_OT_AddEntries(bpy.types.Operator):
    """Add selected bones or objects as entries"""
    bl_idname = "qprk.add_entries"; bl_label = "Add Selected"
    def already(self, props, tt, arm="", bn="", obj=""):
        for n in props.nodes:
            if n.item_type != 'ENTRY' or n.track_type != tt: continue
            if tt == 'BONE' and n.armature_name==arm and n.bone_name==bn: return True
            if tt == 'OBJECT' and n.object_name==obj: return True
        return False
    def execute(self, context):
        props  = context.scene.bmpl_props
        p_uid  = parent_uid_for_new(props)
        added  = 0; last = None
        scene  = context.scene
        active_obj = context.active_object
        sel_bones = []
        if active_obj and active_obj.type == 'ARMATURE' and active_obj.pose:
            sel_bones = [pb for pb in active_obj.pose.bones if getattr(pb, 'select', False) or getattr(pb.bone, 'select', False)]
        if sel_bones:
            for pb in sel_bones:
                if self.already(props,'BONE',active_obj.name,pb.name): continue
                n = props.nodes.add()
                n.uid=make_uid(); n.parent_uid=p_uid; n.item_type='ENTRY'; n.track_type='BONE'
                n.armature_name=active_obj.name; n.bone_name=pb.name
                n.label=f"{active_obj.name} / {pb.name}"
                n.order=next_order(props,p_uid)
                n.frame_start=scene.frame_start; n.frame_end=scene.frame_end
                last=n.uid; added+=1
        else:
            objs = [o for o in scene.objects if o.select_get()] or ([active_obj] if active_obj else [])
            for obj in objs:
                if self.already(props,'OBJECT',obj=obj.name): continue
                n = props.nodes.add()
                n.uid=make_uid(); n.parent_uid=p_uid; n.item_type='ENTRY'; n.track_type='OBJECT'
                n.object_name=obj.name; n.label=obj.name
                n.order=next_order(props,p_uid)
                n.frame_start=scene.frame_start; n.frame_end=scene.frame_end
                last=n.uid; added+=1
        if added==0:
            self.report({'WARNING'},"Nothing new to add"); return {'CANCELLED'}
        if last: props.active_uid=last
        self.report({'INFO'},f"Added {added} entr{'y' if added==1 else 'ies'}")
        return {'FINISHED'}


class QPRK_OT_AddCollection(bpy.types.Operator):
    bl_idname = "qprk.add_collection"; bl_label = "Add Folder"
    def execute(self, context):
        props = context.scene.bmpl_props
        p_uid = parent_uid_for_new(props)
        n = props.nodes.add()
        n.uid=make_uid(); n.parent_uid=p_uid; n.item_type='COLLECTION'
        n.label="Collection"; n.order=next_order(props,p_uid)
        props.active_uid=n.uid
        return {'FINISHED'}


class QPRK_OT_Remove(bpy.types.Operator):
    bl_idname = "qprk.remove"; bl_label = "Remove"
    def collect(self, props, uid):
        u=[uid]
        for c in get_children(props,uid): u.extend(self.collect(props,c.uid))
        return u
    def execute(self, context):
        props = context.scene.bmpl_props
        node  = get_node(props, props.active_uid)
        if node is None: return {'CANCELLED'}
        uids = self.collect(props, node.uid)
        for uid in uids: _path_cache.pop(uid,None)
        for i in reversed([i for i,n in enumerate(props.nodes) if n.uid in uids]):
            props.nodes.remove(i)
        props.active_uid=''
        return {'FINISHED'}


class QPRK_OT_MoveUp(bpy.types.Operator):
    bl_idname = "qprk.move_up"; bl_label = "Move Up"
    def execute(self, context):
        props = context.scene.bmpl_props
        node  = get_node(props, props.active_uid)
        if node is None: return {'CANCELLED'}
        sibs = sorted([n for n in props.nodes if n.parent_uid==node.parent_uid],key=lambda n:n.order)
        idx  = next((i for i,n in enumerate(sibs) if n.uid==node.uid),None)
        if idx is None or idx==0: return {'CANCELLED'}
        sibs[idx].order,sibs[idx-1].order = sibs[idx-1].order,sibs[idx].order
        return {'FINISHED'}


class QPRK_OT_MoveDown(bpy.types.Operator):
    bl_idname = "qprk.move_down"; bl_label = "Move Down"
    def execute(self, context):
        props = context.scene.bmpl_props
        node  = get_node(props, props.active_uid)
        if node is None: return {'CANCELLED'}
        sibs = sorted([n for n in props.nodes if n.parent_uid==node.parent_uid],key=lambda n:n.order)
        idx  = next((i for i,n in enumerate(sibs) if n.uid==node.uid),None)
        if idx is None or idx==len(sibs)-1: return {'CANCELLED'}
        sibs[idx].order,sibs[idx+1].order = sibs[idx+1].order,sibs[idx].order
        return {'FINISHED'}


class QPRK_OT_Reparent(bpy.types.Operator):
    """Move active node to a different collection or root"""
    bl_idname = "qprk.reparent"; bl_label = "Move To..."
    def invoke(self, context, event):
        context.scene.bmpl_props.reparent_target = ""
        return context.window_manager.invoke_props_dialog(self, width=280)
    def draw(self, context):
        layout = self.layout
        props  = context.scene.bmpl_props
        sel    = props.reparent_target
        layout.label(text="Move to:")
        op = layout.operator("qprk.reparent_pick", text="[ Root level ]",
                             icon='HOME', emboss=(sel==""))
        op.picked_uid = ""
        def draw_cols(parent_uid, depth):
            for col in sorted([n for n in props.nodes
                               if n.parent_uid==parent_uid and n.item_type=='COLLECTION'],
                              key=lambda n:n.order):
                row = layout.row()
                if depth: row.separator(factor=depth*2.0)
                op2 = row.operator("qprk.reparent_pick",
                                   text=col.label or "Collection",
                                   icon='FILE_FOLDER', emboss=(sel==col.uid))
                op2.picked_uid = col.uid
                draw_cols(col.uid, depth+1)
        draw_cols("",0)
    def execute(self, context):
        props     = context.scene.bmpl_props
        node      = get_node(props, props.active_uid)
        if node is None: return {'CANCELLED'}
        target_uid = props.reparent_target
        if target_uid:
            tgt = get_node(props, target_uid)
            if tgt is None or tgt.item_type != 'COLLECTION':
                self.report({'WARNING'},"Invalid target"); return {'CANCELLED'}
            check = tgt
            while check:
                if check.uid == node.uid:
                    self.report({'WARNING'},"Cannot move into itself"); return {'CANCELLED'}
                check = get_node(props, check.parent_uid)
        node.parent_uid = target_uid
        node.order = next_order(props, target_uid)
        props.reparent_target = ""
        return {'FINISHED'}


class QPRK_OT_ReparentPick(bpy.types.Operator):
    bl_idname = "qprk.reparent_pick"; bl_label = ""
    picked_uid: bpy.props.StringProperty(default="")
    def execute(self, context):
        context.scene.bmpl_props.reparent_target = self.picked_uid
        return {'FINISHED'}


class QPRK_OT_Rename(bpy.types.Operator):
    bl_idname = "qprk.rename"; bl_label = "Rename"
    new_name: bpy.props.StringProperty(name="Name", default="")
    def invoke(self, context, event):
        node = get_node(context.scene.bmpl_props, context.scene.bmpl_props.active_uid)
        self.new_name = node.label if node else ""
        return context.window_manager.invoke_props_dialog(self, width=250)
    def draw(self, context): self.layout.prop(self,"new_name",text="Name")
    def execute(self, context):
        node = get_node(context.scene.bmpl_props, context.scene.bmpl_props.active_uid)
        if node: node.label = self.new_name
        return {'FINISHED'}


class QPRK_OT_ToggleCollapse(bpy.types.Operator):
    bl_idname = "qprk.toggle_collapse"; bl_label = "Toggle"
    uid: bpy.props.StringProperty()
    def execute(self, context):
        node = get_node(context.scene.bmpl_props, self.uid)
        if node: node.collapsed = not node.collapsed
        return {'FINISHED'}


class QPRK_OT_SetActive(bpy.types.Operator):
    bl_idname = "qprk.set_active"; bl_label = "Select"
    uid: bpy.props.StringProperty()
    def execute(self, context):
        context.scene.bmpl_props.active_uid = self.uid
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
        return {'FINISHED'}


class QPRK_OT_Deselect(bpy.types.Operator):
    bl_idname = "qprk.deselect"; bl_label = "Deselect All"
    def execute(self, context):
        context.scene.bmpl_props.active_uid = ""
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
        return {'FINISHED'}


class QPRK_OT_ToggleColCalc(bpy.types.Operator):
    bl_idname = "qprk.toggle_col_calc"; bl_label = "Toggle Collection Calc"
    uid: bpy.props.StringProperty()
    def execute(self, context):
        props = context.scene.bmpl_props
        node  = get_node(props, self.uid)
        if node is None or node.item_type != 'COLLECTION': return {'CANCELLED'}
        node.col_calc_selected = not node.col_calc_selected
        for e in all_entry_descendants(props, node.uid):
            e.calc_selected = node.col_calc_selected
        return {'FINISHED'}


class QPRK_OT_Calculate(bpy.types.Operator):
    """Calculate paths for all checked entries using their own frame ranges"""
    bl_idname = "qprk.calculate"; bl_label = "Calculate Checked"
    def execute(self, context):
        props   = context.scene.bmpl_props
        targets = checked_entries(props)
        if not targets:
            self.report({'WARNING'},"No entries checked"); return {'CANCELLED'}
        scene = context.scene
        for node in targets: sample_entry(scene, node)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
        self.report({'INFO'},f"Calculated {len(targets)} path(s)")
        return {'FINISHED'}


class QPRK_OT_CalculateEntry(bpy.types.Operator):
    """Calculate path for this single entry"""
    bl_idname = "qprk.calculate_entry"; bl_label = "Calculate"
    uid: bpy.props.StringProperty()
    def execute(self, context):
        props = context.scene.bmpl_props
        node  = get_node(props, self.uid)
        if node is None or node.item_type != 'ENTRY': return {'CANCELLED'}
        sample_entry(context.scene, node)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
        return {'FINISHED'}


class QPRK_OT_CalculateGlobal(bpy.types.Operator):
    """Calculate all entries using the global frame range"""
    bl_idname = "qprk.calculate_global"; bl_label = "Calculate All (Global Range)"
    def execute(self, context):
        props   = context.scene.bmpl_props
        targets = [n for n in props.nodes if n.item_type == 'ENTRY']
        if not targets:
            self.report({'WARNING'},"No entries to calculate"); return {'CANCELLED'}
        scene = context.scene
        # Temporarily override each entry's frame range with global
        saved = [(n, n.frame_start, n.frame_end, n.frame_step) for n in targets]
        for n in targets:
            n.frame_start = props.frame_start
            n.frame_end   = props.frame_end
            n.frame_step  = props.frame_step
        for node in targets: sample_entry(scene, node)
        # Restore
        for n,fs,fe,fst in saved:
            n.frame_start=fs; n.frame_end=fe; n.frame_step=fst
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
        self.report({'INFO'},f"Calculated {len(targets)} path(s) with global range")
        return {'FINISHED'}


class QPRK_OT_ClearEntry(bpy.types.Operator):
    """Clear path for this single entry"""
    bl_idname = "qprk.clear_entry"; bl_label = "Clear"
    uid: bpy.props.StringProperty()
    def execute(self, context):
        node = get_node(context.scene.bmpl_props, self.uid)
        if node: _path_cache.pop(node.uid, None)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
        return {'FINISHED'}


class QPRK_OT_ClearChecked(bpy.types.Operator):
    """Clear paths for all checked entries"""
    bl_idname = "qprk.clear_checked"; bl_label = "Clear Checked Paths"
    def execute(self, context):
        for n in checked_entries(context.scene.bmpl_props):
            _path_cache.pop(n.uid, None)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
        return {'FINISHED'}


class QPRK_OT_ClearAll(bpy.types.Operator):
    bl_idname = "qprk.clear_all"; bl_label = "Clear All Paths"
    def execute(self, context):
        _path_cache.clear()
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
        return {'FINISHED'}


class QPRK_OT_CalculateActiveRangeChecked(bpy.types.Operator):
    """Calculate all checked entries using the active entry's frame range"""
    bl_idname = "qprk.calculate_active_range_checked"; bl_label = "Calculate All Checked (Active Range)"
    uid: bpy.props.StringProperty()
    def execute(self, context):
        props  = context.scene.bmpl_props
        active = get_node(props, self.uid)
        if active is None or active.item_type != 'ENTRY': return {'CANCELLED'}
        targets = checked_entries(props)
        if not targets:
            self.report({'WARNING'},"No entries checked"); return {'CANCELLED'}
        scene = context.scene
        # Temporarily set all checked entries to active's frame range, sample, then restore
        saved = [(n, n.frame_start, n.frame_end, n.frame_step) for n in targets]
        for n in targets:
            n.frame_start = active.frame_start
            n.frame_end   = active.frame_end
            n.frame_step  = active.frame_step
        for node in targets: sample_entry(scene, node)
        for n, fs, fe, fst in saved:
            n.frame_start = fs; n.frame_end = fe; n.frame_step = fst
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
        self.report({'INFO'}, f"Calculated {len(targets)} path(s) with active range")
        return {'FINISHED'}


class QPRK_OT_ToggleNodeFront(bpy.types.Operator):
    """Toggle In Front. If entries are checked, toggles all checked entries. For a collection, toggles all its entries."""
    bl_idname = "qprk.toggle_node_front"; bl_label = "Toggle In Front"
    uid: bpy.props.StringProperty()
    def execute(self, context):
        props = context.scene.bmpl_props
        node  = get_node(props, self.uid)
        if node is None: return {'CANCELLED'}
        checked = checked_entries(props)
        if checked:
            # Toggle all checked entries based on majority state
            v = not all(e.in_front for e in checked)
            for e in checked: e.in_front = v
        elif node.item_type == 'COLLECTION':
            entries = all_entry_descendants(props, node.uid)
            if not entries: return {'CANCELLED'}
            v = not all(e.in_front for e in entries)
            for e in entries: e.in_front = v
        else:
            node.in_front = not node.in_front
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
        return {'FINISHED'}


class QPRK_OT_ToggleNodeVis(bpy.types.Operator):
    """Toggle Visibility. If entries are checked, toggles all checked entries. For a collection, toggles all its entries."""
    bl_idname = "qprk.toggle_node_vis"; bl_label = "Toggle Visibility"
    uid: bpy.props.StringProperty()
    def execute(self, context):
        props = context.scene.bmpl_props
        node  = get_node(props, self.uid)
        if node is None: return {'CANCELLED'}
        checked = checked_entries(props)
        if checked:
            v = not all(e.enabled for e in checked)
            for e in checked: e.enabled = v
        elif node.item_type == 'COLLECTION':
            entries = all_entry_descendants(props, node.uid)
            if not entries: return {'CANCELLED'}
            v = not all(e.enabled for e in entries)
            for e in entries: e.enabled = v
        else:
            node.enabled = not node.enabled
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
        return {'FINISHED'}


class QPRK_OT_ToggleAllCalc(bpy.types.Operator):
    bl_idname = "qprk.toggle_all_calc"; bl_label = "Toggle All Calc"
    def execute(self, context):
        entries = [n for n in context.scene.bmpl_props.nodes if n.item_type=='ENTRY']
        if not entries: return {'CANCELLED'}
        v = not all(e.calc_selected for e in entries)
        for e in entries: e.calc_selected = v
        for c in [n for n in context.scene.bmpl_props.nodes if n.item_type=='COLLECTION']:
            c.col_calc_selected = v
        return {'FINISHED'}


class QPRK_OT_ToggleAllFront(bpy.types.Operator):
    bl_idname = "qprk.toggle_all_front"; bl_label = "Toggle All In Front"
    def execute(self, context):
        entries = [n for n in context.scene.bmpl_props.nodes if n.item_type=='ENTRY']
        if not entries: return {'CANCELLED'}
        v = not all(e.in_front for e in entries)
        for e in entries: e.in_front = v
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
        return {'FINISHED'}


class QPRK_OT_ToggleAllVis(bpy.types.Operator):
    bl_idname = "qprk.toggle_all_vis"; bl_label = "Toggle All Visibility"
    def execute(self, context):
        entries = [n for n in context.scene.bmpl_props.nodes if n.item_type=='ENTRY']
        if not entries: return {'CANCELLED'}
        v = not all(e.enabled for e in entries)
        for e in entries: e.enabled = v
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
        return {'FINISHED'}


class QPRK_OT_CycleSort(bpy.types.Operator):
    """Cycle sort order: Default → A→Z → Z→A → Default"""
    bl_idname = "qprk.cycle_sort"; bl_label = "Cycle Sort"
    def execute(self, context):
        props = context.scene.bmpl_props
        cycle = {'DEFAULT':'AZ','AZ':'ZA','ZA':'DEFAULT'}
        props.sort_mode = cycle[props.sort_mode]
        return {'FINISHED'}

class QPRK_OT_SyncFrameRange(bpy.types.Operator):
    bl_idname = "qprk.sync_frame_range"; bl_label = "Sync from Scene"
    def execute(self, context):
        props = context.scene.bmpl_props
        props.frame_start = context.scene.frame_start
        props.frame_end   = context.scene.frame_end
        return {'FINISHED'}


class QPRK_OT_ApplyToChecked(bpy.types.Operator):
    """Apply active entry's parameters to all other checked entries"""
    bl_idname = "qprk.apply_to_checked"; bl_label = "Apply to All Checked"
    def execute(self, context):
        props  = context.scene.bmpl_props
        active = get_node(props, props.active_uid)
        if active is None or active.item_type != 'ENTRY':
            self.report({'WARNING'},"Select an entry first"); return {'CANCELLED'}
        targets = [n for n in checked_entries(props) if n.uid != active.uid]
        if not targets:
            self.report({'INFO'},"No other checked entries"); return {'FINISHED'}
        for n in targets:
            n.line_width       = active.line_width
            n.dot_size         = active.dot_size
            n.use_frame_colors = active.use_frame_colors
            n.path_color       = active.path_color[:]
            n.color_before     = active.color_before[:]
            n.color_after      = active.color_after[:]
            n.in_front         = active.in_front
            n.enabled          = active.enabled
        self.report({'INFO'},f"Applied to {len(targets)} entr{'y' if len(targets)==1 else 'ies'}")
        return {'FINISHED'}


# ─── Panel ────────────────────────────────────────────────────────────────────

class QPRK_PT_Panel(bpy.types.Panel):
    bl_label      = "QuickPath RKNZ"
    bl_idname     = "QPRK_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type= 'UI'
    bl_category   = "QuickPath RKNZ"

    def draw_node_row(self, col, node, props, depth):
        is_active = node.uid == props.active_uid
        row = col.row(align=True); row.scale_y = 0.9
        if depth: row.separator(factor=depth*2.2)

        if node.item_type == 'COLLECTION':
            op = row.operator("qprk.toggle_col_calc",text='',icon='CHECKBOX_HLT' if node.col_calc_selected else 'CHECKBOX_DEHLT',
                              emboss=False)
            op.uid = node.uid
            op2 = row.operator("qprk.toggle_collapse",text='',icon='TRIA_DOWN' if not node.collapsed else 'TRIA_RIGHT',
                               emboss=False)
            op2.uid = node.uid
            sub = row.row(align=True); sub.alignment='LEFT'
            op3 = sub.operator("qprk.set_active", text=node.label or "Collection",
                               icon='FILE_FOLDER', emboss=is_active)
            op3.uid = node.uid
            # Front + Vis for collection (controls all entries inside)
            right = row.row(align=True); right.alignment='RIGHT'
            entries = all_entry_descendants(props, node.uid)
            col_in_front = entries and all(e.in_front for e in entries)
            col_enabled  = entries and all(e.enabled  for e in entries)
            opf = right.operator("qprk.toggle_node_front", text='',
                                 icon='XRAY' if col_in_front else 'MESH_CUBE', emboss=False)
            opf.uid = node.uid
            opv = right.operator("qprk.toggle_node_vis", text='',
                                 icon='HIDE_OFF' if col_enabled else 'HIDE_ON', emboss=False)
            opv.uid = node.uid
        else:
            row.prop(node,"calc_selected",text="")
            sub=row.row(); sub.active=node.enabled; sub.alignment='LEFT'
            icon='BONE_DATA' if node.track_type=='BONE' else 'OBJECT_DATA'
            op=sub.operator("qprk.set_active",text=node.label or "Entry",
                            icon=icon,emboss=is_active)
            op.uid=node.uid
            # Color swatch + Front + Vis per entry
            right=row.row(align=True); right.alignment='RIGHT'
            cr=right.row(align=True)
            cr.scale_x=0.28 if node.use_frame_colors else 0.45
            if node.use_frame_colors:
                cr.prop(node,"color_before",text="")
                cr.prop(node,"color_after",text="")
            else:
                cr.prop(node,"path_color",text="")
            opf=right.operator("qprk.toggle_node_front",text='',
                               icon='XRAY' if node.in_front else 'MESH_CUBE',emboss=False)
            opf.uid=node.uid
            opv=right.operator("qprk.toggle_node_vis",text='',
                               icon='HIDE_OFF' if node.enabled else 'HIDE_ON',emboss=False)
            opv.uid=node.uid

    def draw(self, context):
        layout = self.layout
        props  = context.scene.bmpl_props

        # ── Visibility ──
        row=layout.row(align=True)
        row.prop(props,"show_path",text="Show Paths",
                 icon='HIDE_OFF' if props.show_path else 'HIDE_ON')
        row.prop(props,"show_dots",text="Dots",icon='KEYFRAME')
        layout.separator()

        # ── Toggle-all row (above list, order: Calc, Front, Vis) ──
        layout.label(text="Tracked Items:", icon='ARMATURE_DATA')
        row=layout.row(align=True)
        row.label(text="All:")
        row.operator("qprk.toggle_all_calc", icon='CHECKBOX_HLT', text="Calc")
        row.operator("qprk.toggle_all_front",icon='XRAY',         text="Front")
        row.operator("qprk.toggle_all_vis",  icon='HIDE_OFF',     text="Vis")
        row.separator()
        row.operator("qprk.deselect",text="",icon='RESTRICT_SELECT_ON')

        # ── Tree list + vertical sidebar ──
        draw_order = build_draw_order(props)
        list_row = layout.row(align=False)

        # Left: the list box
        box = list_row.box()
        box.scale_x = 1.0
        # Invisible spacer row — keeps box width stable when collections collapse
        spacer = box.row()
        spacer.label(text=" " * 60)
        spacer.scale_y = 0.01
        if draw_order:
            col=box.column(align=True)
            for node in draw_order:
                self.draw_node_row(col, node, props, get_depth(props,node))
        else:
            box.label(text="No items yet.",icon='INFO')

        # Right: vertical column of icon buttons controlling the active/checked entry
        sidebar = list_row.column(align=True)
        sidebar.scale_y = 1.1
        sidebar.operator("qprk.add_entries",    text='', icon='ADD')
        sidebar.operator("qprk.add_collection", text='', icon='FILE_FOLDER')
        sidebar.separator()
        sidebar.operator("qprk.move_up",        text='', icon='TRIA_UP')
        sidebar.operator("qprk.move_down",      text='', icon='TRIA_DOWN')
        sidebar.separator()
        sidebar.operator("qprk.rename",         text='', icon='SORTALPHA')
        sidebar.operator("qprk.reparent",       text='', icon='OUTLINER')
        sidebar.separator()
        sidebar.operator("qprk.remove",         text='', icon='X')

        # ── Filter + sort (below list) ──
        row=layout.row(align=True)
        row.prop(props,"filter_text",text="",icon='VIEWZOOM')
        sort_icon={'DEFAULT':'SORTALPHA','AZ':'SORT_ASC','ZA':'SORT_DESC'}[props.sort_mode]
        row.operator("qprk.cycle_sort",text="",icon=sort_icon)

        layout.separator()

        # ── Active / multi-edit detail ──
        active = get_node(props, props.active_uid)
        checked = checked_entries(props)
        multi   = len(checked) > 1

        if active and active.item_type == 'ENTRY':
            box=layout.box()
            icon='BONE_DATA' if active.track_type=='BONE' else 'OBJECT_DATA'
            row=box.row()
            row.label(text=active.label or "Entry", icon=icon)
            if multi:
                row.label(text=f"({len(checked)} checked)",icon='INFO')

            # Per-entry frame range
            row=box.row(align=True)
            row.prop(active,"frame_start")
            row.prop(active,"frame_end")
            row.prop(active,"frame_step",text="Step")
            row=box.row(align=True)
            op=row.operator("qprk.calculate_entry",icon='PLAY',text="Calculate This")
            op.uid=active.uid
            op2=row.operator("qprk.clear_entry",icon='X',text="Clear This")
            op2.uid=active.uid

            box.separator(factor=0.5)

            # Visual params
            row=box.row(align=True)
            row.prop(active,"line_width",text="Width")
            row.prop(active,"dot_size",text="Dot")
            row=box.row()
            row.prop(active,"use_frame_colors",text="Before / After Colors")
            if active.use_frame_colors:
                row=box.row(align=True)
                row.prop(active,"color_before",text="Before")
                row.prop(active,"color_after", text="After")
            else:
                box.prop(active,"path_color",text="Color")

            if multi:
                box.separator(factor=0.3)
                box.operator("qprk.apply_to_checked",icon='PASTEDOWN',
                             text=f"Apply Params to All {len(checked)} Checked")
                op_calc=box.operator("qprk.calculate_active_range_checked",icon='PLAY',
                             text=f"Calculate All {len(checked)} Checked Paths")
                op_calc.uid=active.uid
                box.operator("qprk.clear_checked",icon='X',
                             text=f"Clear All {len(checked)} Checked Paths")

        layout.separator()

        # ── Global frame range ──
        box=layout.box()
        box.label(text="Global Frame Range (overrides all):", icon='TIME')
        row=box.row(align=True)
        row.prop(props,"frame_start")
        row.prop(props,"frame_end")
        row=box.row(align=True)
        row.prop(props,"frame_step")
        row.operator("qprk.sync_frame_range",text="",icon='SCENE_DATA')
        box.operator("qprk.calculate_global",icon='PLAY',
                     text="Calculate All Entries (Global Range)")

        layout.separator()

        # ── Auto update ──
        box=layout.box()
        box.label(text="Auto Update",icon='REC')
        box.prop(props,"live_update",text="Update on Keyframe",toggle=True,
                 icon='RADIOBUT_ON' if props.live_update else 'RADIOBUT_OFF')
        if props.live_update:
            box.label(text="Resamples checked entries on keyframe change",icon='INFO')
        layout.separator()
        layout.operator("qprk.clear_all",icon='TRASH',text="Clear All Paths")


# ─── Registration ─────────────────────────────────────────────────────────────

classes = (
    QPRK_Node,
    QPRK_SceneProps,
    QPRK_OT_AddEntries,
    QPRK_OT_AddCollection,
    QPRK_OT_Remove,
    QPRK_OT_MoveUp,
    QPRK_OT_MoveDown,
    QPRK_OT_Reparent,
    QPRK_OT_ReparentPick,
    QPRK_OT_Rename,
    QPRK_OT_ToggleCollapse,
    QPRK_OT_SetActive,
    QPRK_OT_Deselect,
    QPRK_OT_ToggleColCalc,
    QPRK_OT_Calculate,
    QPRK_OT_CalculateEntry,
    QPRK_OT_CalculateGlobal,
    QPRK_OT_ClearEntry,
    QPRK_OT_ClearChecked,
    QPRK_OT_ClearAll,
    QPRK_OT_CalculateActiveRangeChecked,
    QPRK_OT_ToggleNodeFront,
    QPRK_OT_ToggleNodeVis,
    QPRK_OT_ToggleAllCalc,
    QPRK_OT_ToggleAllFront,
    QPRK_OT_ToggleAllVis,
    QPRK_OT_CycleSort,
    QPRK_OT_SyncFrameRange,
    QPRK_OT_ApplyToChecked,
    QPRK_PT_Panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.bmpl_props = bpy.props.PointerProperty(type=QPRK_SceneProps)
    register_handlers()

def unregister():
    unregister_handlers()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.bmpl_props

if __name__ == "__main__":
    register()
