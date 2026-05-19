bl_info = {
    "name": "QuickPath RKNZ",
    "author": "Rikokensfw",
    "version": (1, 5, 9),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > QuickPath RKNZ",
    "description": "Lightweight realtime motion path visualization for bones, objects and constraint influence | RKNZ",
    "category": "Animation",
}

import bpy
import gpu
import uuid
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix, Quaternion, Euler

# ─── Global State ─────────────────────────────────────────────────────────────
_draw_handler = None
_text_handler = None
_path_cache: dict = {}  # { uid: { frame: Vector } }
_pending_text_draws: list = []  # [ (str, x, y, r, g, b, a, in_front) ] filled by POST_VIEW, consumed by POST_PIXEL

# F-curve fingerprint for KEYFRAME mode — resample only when curves actually change.
# Stores a dict { (action_ptr, data_path, array_index): (n_keys, tuple_of_(frame,value)_pairs) }
_fcurve_fingerprint: dict = {}

# Constraint state fingerprint — resample when constraint properties change
# (mute, influence, type, target, etc.) even if no F-curves exist for them.
# Used by KEYFRAME mode.
_constraint_state_fingerprint: dict = {}


# ─── Core Math / Sampling ─────────────────────────────────────────────────────

def get_bone_world_pos(arm, bone_name):
    if arm is None or arm.type != 'ARMATURE': return None
    pb = arm.pose.bones.get(bone_name)
    if pb is None: return None
    return (arm.matrix_world @ pb.matrix).translation.copy()

def _iter_fcurves(action, anim_data=None):
    """Yield FCurves from an Action, compatible with Blender 3.x through 5.x."""
    is_legacy = getattr(action, 'is_action_legacy', None)
    if is_legacy is True:
        yield from action.fcurves
        return

    if is_legacy is None:
        try:
            fcs = list(action.fcurves)
            yield from fcs
            return
        except AttributeError:
            pass

    if not hasattr(action, 'layers') or not action.layers:
        return

    slot = None
    if anim_data is not None and hasattr(anim_data, 'action_slot'):
        slot = anim_data.action_slot

    try:
        strip = action.layers[0].strips[0]
    except (IndexError, AttributeError):
        return

    if slot is not None:
        try:
            cb = strip.channelbag(slot)
            if cb is not None:
                yield from cb.fcurves
                return
        except (TypeError, AttributeError):
            pass

    if hasattr(strip, 'channelbags'):
        for cb in strip.channelbags:
            yield from cb.fcurves


def build_fc_bone(obj, bone_name):
    if not obj.animation_data or not obj.animation_data.action: return None
    prefix = f'pose.bones["{bone_name}"].'
    m = {}
    for fc in _iter_fcurves(obj.animation_data.action, obj.animation_data):
        if fc.data_path.startswith(prefix):
            m[(fc.data_path[len(prefix):], fc.array_index)] = fc
    return m or None

def build_fc_obj(obj):
    if not obj.animation_data or not obj.animation_data.action: return None
    m = {}
    for fc in _iter_fcurves(obj.animation_data.action, obj.animation_data):
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

def bone_mat_from_action(arm, bone_name, frame):
    arm_data = arm.data
    eb = arm_data.bones.get(bone_name)
    if eb is None: return None
    chain = []
    b = eb
    while b: chain.append(b); b = b.parent
    chain.reverse()
    pose_mat = Matrix.Identity(4)
    bone_local_mat = Matrix.Identity(4)  # local pose matrix of the target bone only
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
        if cb.name == bone_name:
            bone_local_mat = lm  # capture the leaf bone's own local pose matrix
    return pose_mat, bone_local_mat

def arm_world_mat_at(arm, frame):
    fc = build_fc_obj(arm)
    if not fc:
        # No F-curves on the armature itself — use the static local matrix,
        # but still recurse into an animated parent if one exists.
        if arm.parent is not None:
            parent_wm = obj_world_mat(arm.parent, frame)
            base_wm = parent_wm @ arm.matrix_parent_inverse @ arm.matrix_local
        else:
            base_wm = arm.matrix_world.copy()
    else:
        loc   = ev(fc, 'location', 3, frame, list(arm.location))
        scale = ev(fc, 'scale',    3, frame, list(arm.scale))
        local_mat = Matrix.Translation(loc) @ rot_mat(fc, arm.rotation_mode, frame) @ Matrix.Diagonal((*scale, 1.0))
        # If the armature is parented, fold in the animated parent world matrix.
        if arm.parent is not None:
            parent_wm = obj_world_mat(arm.parent, frame)
            base_wm = parent_wm @ arm.matrix_parent_inverse @ local_mat
        else:
            base_wm = local_mat
    # Apply the armature's own object-level constraints (e.g. COPY_TRANSFORMS,
    # CHILD_OF) so that bone world positions are correct when the armature
    # itself is driven by an object constraint.
    for c in arm.constraints:
        if not c.mute and c.type in FAST_C_OBJ:
            base_wm = apply_c(c, base_wm, frame, obj=arm)
    return base_wm

def bone_world_pos_fc(arm, bone_name, frame):
    result = bone_mat_from_action(arm, bone_name, frame)
    if result is None: return None
    m, _ = result
    return (arm_world_mat_at(arm, frame) @ m).translation.copy()

FAST_C = {'COPY_LOCATION','COPY_ROTATION','COPY_TRANSFORMS','COPY_SCALE',
           'DAMPED_TRACK','TRACK_TO','LOCKED_TRACK','CHILD_OF'}

# Same set but for object constraints — identical types are supported.
FAST_C_OBJ = FAST_C

_live_matrix_overrides: dict = {}
_live_bone_overrides: dict = {}

def tgt_world_mat(c, frame):
    t = c.target
    if t is None: return None
    if c.subtarget and t.type == 'ARMATURE':
        # For subtarget bones we can only override the whole armature matrix;
        # use live matrix_world if available, else fall back to F-curve math.
        if t.name in _live_matrix_overrides:
            arm_wm = _live_matrix_overrides[t.name]
            result = bone_mat_from_action(t, c.subtarget, frame)
            if result is None: return None
            m, _ = result
            return arm_wm @ m
        result = bone_mat_from_action(t, c.subtarget, frame)
        if result is None: return None
        m, _ = result
        return (arm_world_mat_at(t, frame) @ m)
    if t.name in _live_matrix_overrides:
        return _live_matrix_overrides[t.name].copy()
    return obj_world_mat(t, frame)

def obj_world_mat(obj, frame):
    if obj.name in _live_matrix_overrides:
        return _live_matrix_overrides[obj.name].copy()
    fc = build_fc_obj(obj)
    if not fc:
        # No F-curves on this object — use its static local matrix.
        # If it has an animated parent we must still recurse to get the
        # correct world matrix at this frame.
        if obj.parent is not None:
            parent_wm = obj_world_mat(obj.parent, frame)
            return parent_wm @ obj.matrix_parent_inverse @ obj.matrix_local
        return obj.matrix_world.copy()
    loc   = ev(fc,'location',3,frame,list(obj.location))
    scale = ev(fc,'scale',3,frame,list(obj.scale))
    local_mat = Matrix.Translation(loc) @ rot_mat(fc,obj.rotation_mode,frame) @ Matrix.Diagonal((*scale,1.0))
    # If the object is parented, combine with the animated parent world matrix.
    if obj.parent is not None:
        parent_wm = obj_world_mat(obj.parent, frame)
        return parent_wm @ obj.matrix_parent_inverse @ local_mat
    return local_mat

def get_constraint_influence(obj_or_arm, bone_name, constraint, frame):
    """Return the influence of a constraint at the given frame.
    Evaluates the influence F-Curve if it is animated; otherwise returns
    the static constraint.influence value.

    Works for both bone constraints and plain object constraints:
      - Bone:   pass the armature as obj_or_arm with the bone name.
      - Object: pass the object as obj_or_arm with bone_name="" (empty).
    """
    if obj_or_arm is None or not obj_or_arm.animation_data or not obj_or_arm.animation_data.action:
        return constraint.influence
    if bone_name:
        path = f'pose.bones["{bone_name}"].constraints["{constraint.name}"].influence'
    else:
        path = f'constraints["{constraint.name}"].influence'
    for fc in _iter_fcurves(obj_or_arm.animation_data.action, obj_or_arm.animation_data):
        if fc.data_path == path and fc.array_index == 0:
            return fc.evaluate(frame)
    return constraint.influence


def apply_c(c, bm, frame, pose_mat=None, arm=None, bone_name="", arm_world_mat=None, obj=None):
    # obj = plain object (for object constraints with animated influence)
    # arm = armature object (for bone constraints with animated influence)
    # Falls back to static c.influence when neither is supplied.
    if obj is not None:
        inf = get_constraint_influence(obj, "", c, frame)
    elif arm is not None:
        inf = get_constraint_influence(arm, bone_name, c, frame)
    else:
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
        io = c.inverse_matrix if hasattr(c, 'inverse_matrix') else Matrix.Identity(4)
        # bm is already the bone's world matrix (arm_world @ armature_space_mat).
        # CHILD_OF formula: constrained_world = tgt @ inverse_matrix @ arm_world_inv @ bm
        # i.e. convert bm back to armature space, then parent under tgt.
        if arm_world_mat is not None:
            constrained = tgt @ io @ arm_world_mat.inverted() @ bm
        else:
            # Fallback: use pose_mat (armature-space bone matrix) if arm_world_mat unavailable
            pm = pose_mat if pose_mat is not None else bm
            constrained = tgt @ io @ pm
        return bm.lerp(constrained, inf)
    return bm

def all_fast(arm, bn):
    pb = arm.pose.bones.get(bn)
    return True if pb is None else all(c.mute or c.type in FAST_C for c in pb.constraints)

def bone_pos_constrained(arm, bn, frame):
    pb = arm.pose.bones.get(bn)
    if pb is None: return None
    for c in pb.constraints:
        if not c.mute and c.type not in FAST_C: return None
    result = bone_mat_from_action(arm, bn, frame)
    if result is None: return None
    m, bone_local_mat = result
    arm_wm = arm_world_mat_at(arm, frame)
    wm = arm_wm @ m
    for c in pb.constraints:
        if not c.mute: wm = apply_c(c, wm, frame, pose_mat=bone_local_mat, arm=arm, bone_name=bn, arm_world_mat=arm_wm)
    return wm.translation.copy()

def obj_all_fast(obj):
    """Return True if every constraint on obj is either muted or in FAST_C_OBJ."""
    return all(c.mute or c.type in FAST_C_OBJ for c in obj.constraints)

def obj_pos_constrained(obj, frame):
    """Return the world position of obj at frame, applying supported constraints.

    Mirrors bone_pos_constrained() for plain objects.  Returns None when a
    non-fast constraint is present (caller should fall back to frame_set).
    Passes obj= into apply_c so animated constraint influence F-curves are
    evaluated correctly, matching how bone constraints work.
    """
    if not obj_all_fast(obj):
        return None
    wm = obj_world_mat(obj, frame)
    for c in obj.constraints:
        if not c.mute:
            wm = apply_c(c, wm, frame, obj=obj)
    return wm.translation.copy()

def arm_has_transform_keys(arm):
    fc = build_fc_obj(arm)
    if not fc: return False
    for prop in ('location', 'rotation_euler', 'rotation_quaternion',
                 'rotation_axis_angle', 'scale'):
        if any(k[0] == prop for k in fc): return True
    return False

def constraint_targets_all_keyed(arm, bn):
    pb = arm.pose.bones.get(bn)
    if pb is None: return True
    for c in pb.constraints:
        if c.mute or c.type not in FAST_C: continue
        t = c.target
        if t is None: continue
        if c.subtarget and t.type == 'ARMATURE':
            if not arm_has_transform_keys(t):
                return False
        else:
            fc = build_fc_obj(t)
            if not fc:
                return False
    return True

def collect_missing_keys(nodes):
    missing = []
    seen    = set()

    def add(entry_label, obj_name, bone_name, reason):
        key = (obj_name, bone_name, reason)
        if key not in seen:
            seen.add(key)
            missing.append(dict(entry_label=entry_label, obj_name=obj_name,
                                bone_name=bone_name, reason=reason))

    for node in nodes:
        if node.item_type != 'ENTRY':
            continue

        if node.track_type == 'BONE':
            arm = bpy.data.objects.get(node.armature_name)
            if arm is None: continue
            bn  = node.bone_name
            lbl = node.label or node.bone_name

            arm_has_action = arm.animation_data is not None and arm.animation_data.action is not None

            if not arm_has_action:
                add(lbl, arm.name, "", "Armature has no action / bone has no keyframes")
                continue

            if not arm_has_transform_keys(arm):
                from mathutils import Vector
                if (arm.location.length > 1e-5 or
                        any(abs(s - 1.0) > 1e-5 for s in arm.scale) or
                        arm.rotation_euler.to_quaternion().angle > 1e-5):
                    add(lbl, arm.name, "",
                        "Armature object is offset from origin but has no transform keyframes")

            pb = arm.pose.bones.get(bn)
            if pb:
                for c in pb.constraints:
                    if c.mute or c.type not in FAST_C: continue
                    t = c.target
                    if t is None: continue
                    if c.subtarget and t.type == 'ARMATURE':
                        if not arm_has_transform_keys(t):
                            from mathutils import Vector
                            if (t.location.length > 1e-5 or
                                    any(abs(s - 1.0) > 1e-5 for s in t.scale) or
                                    t.rotation_euler.to_quaternion().angle > 1e-5):
                                add(lbl, t.name, c.subtarget,
                                    f"Constraint '{c.name}' target armature is offset but un-keyed")
                    else:
                        if not build_fc_obj(t):
                            add(lbl, t.name, "",
                                f"Constraint '{c.name}' target object has no transform keyframes")

        else:  # OBJECT
            obj = bpy.data.objects.get(node.object_name)
            if obj is None: continue
            lbl = node.label or node.object_name

            has_action = obj.animation_data is not None and obj.animation_data.action is not None
            if not has_action:
                add(lbl, obj.name, "", "Object has no action / no keyframes")
                continue

            for c in obj.constraints:
                if c.mute or c.type not in FAST_C_OBJ: continue
                t = c.target
                if t is None: continue
                if not build_fc_obj(t):
                    add(lbl, t.name, "",
                        f"Constraint '{c.name}' target object has no transform keyframes")

    return missing

def entry_uses_slow_path(node):
    if node.item_type != 'ENTRY': return False
    if node.track_type == 'BONE':
        arm = bpy.data.objects.get(node.armature_name)
        if arm is None: return False
        bn  = node.bone_name
        pb  = arm.pose.bones.get(bn)
        has_c = pb and len(pb.constraints) > 0
        c_ok  = all_fast(arm, bn)
        arm_has_action     = arm.animation_data is not None and arm.animation_data.action is not None
        arm_transform_safe = (not arm_has_transform_keys(arm)) or arm_has_action
        targets_safe       = (not has_c) or constraint_targets_all_keyed(arm, bn)
        # Also check the armature's own object constraints — if any are present
        # and not all supported (FAST_C_OBJ), we must fall back to frame_set.
        arm_obj_c_ok = obj_all_fast(arm)
        return not (arm_has_action and arm_transform_safe and targets_safe and (not has_c or c_ok) and arm_obj_c_ok)
    else:
        obj = bpy.data.objects.get(node.object_name)
        if obj is None: return False
        has_action = obj.animation_data is not None and obj.animation_data.action is not None
        has_c      = len(obj.constraints) > 0
        c_ok       = obj_all_fast(obj)
        return not (has_action and (not has_c or c_ok))


def sample_entry(scene, node):
    """Sample a single entry using its own frame range."""
    uid = node.uid
    _path_cache[uid] = {}
    fs = node.frame_start; fe = node.frame_end; fst = max(1, node.frame_step)

    if node.track_type == 'BONE':
        arm = bpy.data.objects.get(node.armature_name)
        bn  = node.bone_name
        if arm is None or not bn: return
        pb    = arm.pose.bones.get(bn)
        has_c = pb and len(pb.constraints) > 0
        c_ok  = all_fast(arm, bn)
        arm_has_action = arm.animation_data is not None and arm.animation_data.action is not None
        arm_transform_safe = (not arm_has_transform_keys(arm)) or arm_has_action
        targets_safe = (not has_c) or constraint_targets_all_keyed(arm, bn)
        # Also gate on the armature's own object constraints; arm_world_mat_at
        # applies FAST_C_OBJ ones, but unsupported ones require frame_set.
        arm_obj_c_ok = obj_all_fast(arm)
        use_fast = arm_has_action and arm_transform_safe and targets_safe and (not has_c or c_ok) and arm_obj_c_ok
        if use_fast:
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
        has_action   = obj.animation_data is not None and obj.animation_data.action is not None
        has_c        = len(obj.constraints) > 0
        c_ok         = obj_all_fast(obj)
        # Use fast path when there's a keyed action AND all constraints (if any)
        # are in the supported set.  Fall back to frame_set otherwise so that
        # unsupported constraints (FOLLOW_PATH, ARMATURE, etc.) are evaluated
        # correctly by Blender's full depsgraph.
        use_fast = has_action and (not has_c or c_ok)
        if use_fast:
            for f in range(fs, fe+1, fst):
                if has_c:
                    pos = obj_pos_constrained(obj, f)
                    if pos: _path_cache[uid][f] = pos
                else:
                    _path_cache[uid][f] = obj_world_mat(obj, f).translation.copy()
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
    sort    = props.sort_mode
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
    # Always clear pending text draws so POST_PIXEL handler has nothing to
    # render when the path (and therefore frame-number labels) is hidden.
    _pending_text_draws.clear()
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

    def dots(pts, col, sz, in_front=False):
        if not pts: return
        space = bpy.context.space_data
        region_3d = space.region_3d if space else None
        if region_3d is None: return

        vm  = region_3d.view_matrix
        h   = reg.height if reg else 1080

        view_right_ws = Vector((vm[0][0], vm[1][0], vm[2][0]))
        view_up_ws    = Vector((vm[0][1], vm[1][1], vm[2][1]))

        is_persp = region_3d.is_perspective
        if is_persp:
            cam_pos = vm.inverted().translation
            lens    = space.lens if space else 50.0

        verts   = []
        indices = []
        base    = 0

        for p in pts:
            if is_persp:
                ray = p - cam_pos
                dist = ray.length
                if dist < 1e-6:
                    continue
                pixel_size = dist * 36.0 / (h * lens)

                fwd = ray / dist
                ref_up = view_up_ws
                right  = fwd.cross(ref_up)
                rlen   = right.length
                if rlen < 1e-6:
                    ref_up = view_right_ws
                    right  = fwd.cross(ref_up)
                    rlen   = right.length
                if rlen < 1e-6:
                    right = view_right_ws
                else:
                    right /= rlen
                up = right.cross(fwd).normalized()
            else:
                pixel_size = region_3d.view_distance / (h * 0.5)
                right = view_right_ws
                up    = view_up_ws

            half = (sz * 0.5) * pixel_size

            r = right * half
            u = up    * half
            c = p

            verts += [
                (c - r - u).to_tuple(),
                (c + r - u).to_tuple(),
                (c + r + u).to_tuple(),
                (c - r + u).to_tuple(),
            ]
            indices += [(base, base+1, base+2), (base, base+2, base+3)]
            base += 4

        if not verts: return

        sh = gpu.shader.from_builtin('UNIFORM_COLOR')
        bt = batch_for_shader(sh, 'TRIS', {"pos": verts}, indices=indices)
        sh.bind()
        sh.uniform_float("color", tuple(col))
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('ALWAYS' if in_front else 'LESS_EQUAL')
        bt.draw(sh)

    def get_keyframe_frames_for_node(node):
        """Return a set of frame numbers that have actual keyframes for this node."""
        kf_frames = set()
        if node.track_type == 'BONE':
            arm = bpy.data.objects.get(node.armature_name)
            if arm and arm.animation_data and arm.animation_data.action:
                prefix = f'pose.bones["{node.bone_name}"].'
                for fc in _iter_fcurves(arm.animation_data.action, arm.animation_data):
                    if fc.data_path.startswith(prefix):
                        for kp in fc.keyframe_points:
                            kf_frames.add(int(round(kp.co[0])))
        else:
            obj = bpy.data.objects.get(node.object_name)
            if obj and obj.animation_data and obj.animation_data.action:
                for fc in _iter_fcurves(obj.animation_data.action, obj.animation_data):
                    if fc.data_path in ('location', 'rotation_euler',
                                        'rotation_quaternion', 'rotation_axis_angle', 'scale'):
                        for kp in fc.keyframe_points:
                            kf_frames.add(int(round(kp.co[0])))
        return kf_frames

    def collect_frame_number_labels(frames, positions, col, kf_frames, dot_sz, in_front=False):
        """
        For every frame in `frames` that is in `kf_frames`, project the 3D
        position to 2D and push a (text, x, y, r, g, b, a, in_front) tuple into
        _pending_text_draws.  Offset is proportional to dot_sz so the label
        sits just outside the dot billboard.

        Depth occlusion is tested HERE in POST_VIEW while the depth buffer is
        still valid.  Testing it later in POST_PIXEL always reads 1.0 (max
        depth) because the framebuffer has already been composited.
        """
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        from gpu.matrix import get_projection_matrix, get_model_view_matrix
        import mathutils

        space      = bpy.context.space_data
        region_3d  = space.region_3d if space else None
        if region_3d is None:
            return
        offset = dot_sz * 0.55 + 4.0   # pixels: clear the dot then a small gap
        r, g, b, a = col[0], col[1], col[2], col[3]

        # Snapshot MVP and framebuffer once for the whole batch (POST_VIEW context)
        mv  = get_model_view_matrix()
        pr  = get_projection_matrix()
        mvp = pr @ mv
        fb  = gpu.state.active_framebuffer_get()

        for f, p in zip(frames, positions):
            if f not in kf_frames:
                continue
            co2d = location_3d_to_region_2d(reg, region_3d, p)
            if co2d is None:
                continue

            # Depth occlusion test — skip label when hidden behind geometry
            if not in_front:
                p4   = mathutils.Vector((p.x, p.y, p.z, 1.0))
                clip = mvp @ p4
                if abs(clip.w) > 1e-8:
                    ndc_z     = clip.z / clip.w          # -1..1 OpenGL NDC
                    win_depth = (ndc_z + 1.0) * 0.5      # remap to 0..1
                    px = max(0, min(int(co2d[0]), reg.width  - 1))
                    py = max(0, min(int(co2d[1]), reg.height - 1))
                    try:
                        buf_depth = fb.read_depth(px, py, 1, 1)[0][0]
                        if win_depth > buf_depth + 0.001:
                            continue   # occluded — skip this label
                    except Exception:
                        pass

            _pending_text_draws.append((str(f),
                                        co2d[0] + offset,
                                        co2d[1] + offset,
                                        r, g, b, a,
                                        in_front,
                                        p.copy()))

    def glow(pts, col, bw):
        base = max(bw, 3.0)
        gc = desaturate(col, 0.5)
        for wm, am in [(8,0.04),(5,0.07),(3,0.12),(1.8,0.20)]:
            line(pts,(gc[0],gc[1],gc[2],gc[3]*am), base*wm)

    def outline(pts, w):
        line(pts,(1,1,1,0.6),w+2.0)

    show_dots  = props.show_dots
    show_fnums = props.show_frame_numbers

    # Clear the shared text-draw list so POST_PIXEL handler gets fresh data
    _pending_text_draws.clear()

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
            fb  = frames[:split+1];    fa  = frames[split:]
            cb = mc(node.color_before); ca = mc(node.color_after)
            if glowing: glow(bef,cb,node.line_width); glow(aft,ca,node.line_width)
            if bright:  outline(bef,node.line_width); outline(aft,node.line_width)
            line(bef,cb,node.line_width); line(aft,ca,node.line_width)
            if show_dots:
                if node.dot_color_mode == 'SPLIT':
                    dots(bef, tuple(node.dot_color_before), node.dot_size, node.in_front)
                    dots(aft, tuple(node.dot_color_after),  node.dot_size, node.in_front)
                else:
                    dc = tuple(node.dot_color)
                    dots(bef, dc, node.dot_size, node.in_front)
                    dots(aft, dc, node.dot_size, node.in_front)
            if show_fnums:
                kf_frames = get_keyframe_frames_for_node(node)
                if node.dot_color_mode == 'SPLIT':
                    collect_frame_number_labels(fb, bef, tuple(node.dot_color_before), kf_frames, node.dot_size, node.in_front)
                    collect_frame_number_labels(fa, aft, tuple(node.dot_color_after),  kf_frames, node.dot_size, node.in_front)
                else:
                    dc = tuple(node.dot_color)
                    collect_frame_number_labels(fb, bef, dc, kf_frames, node.dot_size, node.in_front)
                    collect_frame_number_labels(fa, aft, dc, kf_frames, node.dot_size, node.in_front)
        else:
            c = mc(node.path_color)
            if glowing: glow(positions,c,node.line_width)
            if bright:  outline(positions,node.line_width)
            line(positions,c,node.line_width)
            if show_dots:
                if node.dot_color_mode == 'SPLIT':
                    split = len(frames)
                    for i,f in enumerate(frames):
                        if f >= cf: split = i; break
                    bef = positions[:split+1]; aft = positions[split:]
                    dots(bef, tuple(node.dot_color_before), node.dot_size, node.in_front)
                    dots(aft, tuple(node.dot_color_after),  node.dot_size, node.in_front)
                else:
                    dots(positions, tuple(node.dot_color), node.dot_size, node.in_front)
            if show_fnums:
                kf_frames = get_keyframe_frames_for_node(node)
                if node.dot_color_mode == 'SPLIT':
                    split = len(frames)
                    for i,f in enumerate(frames):
                        if f >= cf: split = i; break
                    bef = positions[:split+1]; aft = positions[split:]
                    fb  = frames[:split+1];    fa  = frames[split:]
                    collect_frame_number_labels(fb, bef, tuple(node.dot_color_before), kf_frames, node.dot_size, node.in_front)
                    collect_frame_number_labels(fa, aft, tuple(node.dot_color_after),  kf_frames, node.dot_size, node.in_front)
                else:
                    collect_frame_number_labels(frames, positions, tuple(node.dot_color), kf_frames, node.dot_size, node.in_front)

        # Current frame dot — white, same size as other dots
        if cf in frames:
            dots([positions[frames.index(cf)]], (1,1,1,1), node.dot_size, node.in_front)

        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')



def draw_frame_numbers_2d():
    """POST_PIXEL handler: draw frame-number labels collected by draw_paths.

    Occlusion culling was already applied in POST_VIEW (collect_frame_number_labels)
    while the depth buffer was still valid.  Everything in _pending_text_draws
    is already confirmed visible — just draw it.
    """
    if not _pending_text_draws:
        return
    try:
        import blf
        font_id = 0
        blf.size(font_id, 11)
        offsets = [(-2,-2),(-2,0),(-2,2),(0,-2),(0,2),(2,-2),(2,0),(2,2)]

        # Draw thick black outline by stamping black text at 8 surrounding offsets
        for entry in _pending_text_draws:
            text, x, y, r, g, b, a, in_front, pos_3d = entry
            blf.color(font_id, 0.0, 0.0, 0.0, a)
            for ox, oy in offsets:
                blf.position(font_id, x + ox, y + oy, 0)
                blf.draw(font_id, text)
        # Draw colored text on top
        for entry in _pending_text_draws:
            text, x, y, r, g, b, a, in_front, pos_3d = entry
            blf.color(font_id, r, g, b, a)
            blf.position(font_id, x, y, 0)
            blf.draw(font_id, text)
    except Exception:
        pass

# ─── Handlers ─────────────────────────────────────────────────────────────────


def _iter_all_fcurves_for_node(node):
    """Yield (action, anim_data, fc) for every F-curve relevant to node."""
    if node.track_type == 'BONE':
        arm = bpy.data.objects.get(node.armature_name)
        if arm is None or not arm.animation_data or not arm.animation_data.action:
            return
        ad     = arm.animation_data
        action = ad.action
        prefix = f'pose.bones["{node.bone_name}"].'
        for fc in _iter_fcurves(action, ad):
            if fc.data_path.startswith(prefix):
                yield action, ad, fc
    else:
        obj = bpy.data.objects.get(node.object_name)
        if obj is None or not obj.animation_data or not obj.animation_data.action:
            return
        ad     = obj.animation_data
        action = ad.action
        for fc in _iter_fcurves(action, ad):
            if fc.data_path in ('location',
                                'rotation_euler', 'rotation_quaternion',
                                'rotation_axis_angle', 'scale'):
                yield action, ad, fc
            elif (fc.data_path.startswith('constraints[') and
                  fc.data_path.endswith('].influence')):
                yield action, ad, fc


_TRANSFORM_PATHS = frozenset({
    'location',
    'rotation_euler', 'rotation_quaternion', 'rotation_axis_angle',
    'scale',
})

def _iter_obj_transform_fcurves(obj):
    """Yield (action, anim_data, fc) for transform + constraint-influence
    F-curves on a plain object (used for related-object watching)."""
    if obj is None or not obj.animation_data or not obj.animation_data.action:
        return
    ad = obj.animation_data
    action = ad.action
    for fc in _iter_fcurves(action, ad):
        if fc.data_path in _TRANSFORM_PATHS:
            yield action, ad, fc
        elif (fc.data_path.startswith('constraints[') and
              fc.data_path.endswith('].influence')):
            yield action, ad, fc


def _iter_bone_chain_fcurves(arm, bone_name):
    """Yield (action, anim_data, fc) for bone_name AND every ancestor bone
    in arm's pose hierarchy.  Needed because bone_mat_from_action() walks
    the full chain, so edits to a parent bone's curves affect the child's
    world position."""
    if arm is None or not arm.animation_data or not arm.animation_data.action:
        return
    ad = arm.animation_data
    action = ad.action

    # Collect the full ancestor chain (bone → root)
    eb = arm.data.bones.get(bone_name)
    if eb is None:
        return
    chain_names = []
    b = eb
    while b is not None:
        chain_names.append(b.name)
        b = b.parent

    for bn in chain_names:
        prefix = f'pose.bones["{bn}"].'
        for fc in _iter_fcurves(action, ad):
            if fc.data_path.startswith(prefix):
                yield action, ad, fc


def _iter_related_fcurves_for_node(node):
    """Yield (action, anim_data, fc) for every F-curve that indirectly
    affects the tracked entry but is NOT the entry's own curves.

    Covers:
      • Constraint target objects and their object-parent chains
      • Subtarget bone full chains (including ancestor bones in the armature)
      • Object-parent chain of the tracked armature/object itself
      • The armature object's own transform curves (for BONE entries), since
        arm_world_mat_at() reads those to compute world position

    These are included in the KEYFRAME-mode fingerprint so that editing any
    related F-curve in the Graph Editor immediately triggers a resample.
    """
    visited_objs:  set = set()   # object names already yielded
    visited_bones: set = set()   # (arm_name, bone_name) prefixes already yielded

    def _yield_obj(obj):
        if obj is None or obj.name in visited_objs:
            return
        visited_objs.add(obj.name)
        yield from _iter_obj_transform_fcurves(obj)
        # Also walk the object-parent chain
        cur = obj.parent
        while cur is not None and cur.name not in visited_objs:
            visited_objs.add(cur.name)
            yield from _iter_obj_transform_fcurves(cur)
            cur = cur.parent

    def _yield_bone_chain(arm, bone_name):
        key = (arm.name if arm else '', bone_name)
        if arm is None or key in visited_bones:
            return
        visited_bones.add(key)
        yield from _iter_bone_chain_fcurves(arm, bone_name)

    if node.track_type == 'BONE':
        arm = bpy.data.objects.get(node.armature_name)
        if arm is None:
            return

        # The armature's own object-level transform curves matter (arm_world_mat_at).
        # Mark the armature visited so we don't double-yield in the target loop below.
        visited_objs.add(arm.name)
        yield from _iter_obj_transform_fcurves(arm)

        # Object-parent chain of the armature
        cur = arm.parent
        while cur is not None and cur.name not in visited_objs:
            visited_objs.add(cur.name)
            yield from _iter_obj_transform_fcurves(cur)
            cur = cur.parent

        # All bone-ancestor chains in the same armature that are NOT the
        # tracked bone itself (those are covered by _iter_all_fcurves_for_node).
        eb = arm.data.bones.get(node.bone_name)
        if eb is not None:
            b = eb.parent  # start from the parent, not the bone itself
            while b is not None:
                prefix_key = (arm.name, b.name)
                if prefix_key not in visited_bones:
                    visited_bones.add(prefix_key)
                    prefix = f'pose.bones["{b.name}"].'
                    if arm.animation_data and arm.animation_data.action:
                        ad = arm.animation_data
                        for fc in _iter_fcurves(ad.action, ad):
                            if fc.data_path.startswith(prefix):
                                yield ad.action, ad, fc
                b = b.parent

        # Constraint targets on the tracked bone
        pb = arm.pose.bones.get(node.bone_name)
        if pb:
            for c in pb.constraints:
                if c.mute:
                    continue
                t = getattr(c, 'target', None)
                if t is None:
                    continue
                subtarget = getattr(c, 'subtarget', '')
                if subtarget and t.type == 'ARMATURE':
                    yield from _yield_bone_chain(t, subtarget)
                    # Also the armature object's own transform
                    if t.name not in visited_objs:
                        visited_objs.add(t.name)
                        yield from _iter_obj_transform_fcurves(t)
                else:
                    yield from _yield_obj(t)

    else:  # OBJECT
        obj = bpy.data.objects.get(node.object_name)
        if obj is None:
            return

        # Object-parent chain of the tracked object itself
        visited_objs.add(obj.name)
        cur = obj.parent
        while cur is not None and cur.name not in visited_objs:
            visited_objs.add(cur.name)
            yield from _iter_obj_transform_fcurves(cur)
            cur = cur.parent

        # Constraint targets on the tracked object
        for c in obj.constraints:
            if c.mute:
                continue
            t = getattr(c, 'target', None)
            if t is None:
                continue
            subtarget = getattr(c, 'subtarget', '')
            if subtarget and t.type == 'ARMATURE':
                yield from _yield_bone_chain(t, subtarget)
                if t.name not in visited_objs:
                    visited_objs.add(t.name)
                    yield from _iter_obj_transform_fcurves(t)
            else:
                yield from _yield_obj(t)




def _read_fc_at(fc, frame):
    """Return (keyframe_point, value) if a keypoint exists at frame, else (None, None)."""
    for kp in fc.keyframe_points:
        if abs(kp.co[0] - frame) < 0.5:
            return kp, kp.co[1]
    return None, None


def _write_fc_at(fc, frame, value):
    """Set or insert a keyframe at frame=value.  Returns old_value (or None if inserted new).

    Only co[1] (the key value) is modified — handles are intentionally left
    untouched.  Overwriting handle_left/handle_right while Blender's own
    transform operator is still live corrupts bezier tangents and causes
    crashes on mouse-release.
    """
    kp, old = _read_fc_at(fc, frame)
    if kp is not None:
        kp.co[1] = value
        fc.update()
        return old
    else:
        fc.keyframe_points.insert(frame, value, options={'FAST'})
        fc.update()
        return None   # sentinel: new keyframe, restore = delete


def _restore_fc_at(fc, frame, old_value):
    """Undo _write_fc_at: restore old_value or delete the inserted keyframe."""
    if old_value is None:
        kp, _ = _read_fc_at(fc, frame)
        if kp is not None:
            fc.keyframe_points.remove(kp, fast=True)
            fc.update()
    else:
        kp, _ = _read_fc_at(fc, frame)
        if kp is not None:
            kp.co[1] = old_value
        fc.update()


def _bone_channel_values(arm, bn):
    """Return dict {(data_path_suffix, array_index): value} for all animated
    channels of pose-bone bn, reading from the live pose (not F-curves)."""
    pb = arm.pose.bones.get(bn)
    if pb is None:
        return {}
    out = {}
    loc = pb.location
    out[('location', 0)] = loc.x
    out[('location', 1)] = loc.y
    out[('location', 2)] = loc.z
    rm = pb.rotation_mode
    if rm == 'QUATERNION':
        q = pb.rotation_quaternion
        out[('rotation_quaternion', 0)] = q.w
        out[('rotation_quaternion', 1)] = q.x
        out[('rotation_quaternion', 2)] = q.y
        out[('rotation_quaternion', 3)] = q.z
    elif rm == 'AXIS_ANGLE':
        aa = pb.rotation_axis_angle
        for i in range(4):
            out[('rotation_axis_angle', i)] = aa[i]
    else:
        e = pb.rotation_euler
        out[('rotation_euler', 0)] = e.x
        out[('rotation_euler', 1)] = e.y
        out[('rotation_euler', 2)] = e.z
    sc = pb.scale
    out[('scale', 0)] = sc.x
    out[('scale', 1)] = sc.y
    out[('scale', 2)] = sc.z
    return out


def _obj_channel_values(obj):
    """Return dict {(data_path, array_index): value} for all animated
    channels of obj, reading from the live object (not F-curves)."""
    out = {}
    loc = obj.location
    out[('location', 0)] = loc.x
    out[('location', 1)] = loc.y
    out[('location', 2)] = loc.z
    rm = obj.rotation_mode
    if rm == 'QUATERNION':
        q = obj.rotation_quaternion
        out[('rotation_quaternion', 0)] = q.w
        out[('rotation_quaternion', 1)] = q.x
        out[('rotation_quaternion', 2)] = q.y
        out[('rotation_quaternion', 3)] = q.z
    elif rm == 'AXIS_ANGLE':
        aa = obj.rotation_axis_angle
        for i in range(4):
            out[('rotation_axis_angle', i)] = aa[i]
    else:
        e = obj.rotation_euler
        out[('rotation_euler', 0)] = e.x
        out[('rotation_euler', 1)] = e.y
        out[('rotation_euler', 2)] = e.z
    sc = obj.scale
    out[('scale', 0)] = sc.x
    out[('scale', 1)] = sc.y
    out[('scale', 2)] = sc.z
    return out


def _constraint_influence_values(pb):
    """Return dict {constraint_name: influence} for all constraints on a pose bone."""
    return {c.name: c.influence for c in pb.constraints}


def _obj_constraint_influence_values(obj):
    """Return dict {constraint_name: influence} for all constraints on a plain object."""
    return {c.name: c.influence for c in obj.constraints}


def _write_live_pose_for_related(scene, frame, updated_names):
    """Write the current live pose into F-curves for every object/bone that
    influences a tracked entry but is NOT itself a tracked entry.

    Covers plain-object targets, bone subtargets and their full bone parent
    chains within the armature, and object-level parent chains.
    Only writes objects whose armature/object name appears in updated_names.
    """
    props = scene.bmpl_props

    tracked_names: set = set()
    for node in props.nodes:
        if node.item_type != 'ENTRY' or not node.enabled:
            continue
        if node.track_type == 'BONE':
            tracked_names.add(node.armature_name)
        else:
            tracked_names.add(node.object_name)

    written_objs:  set = set()
    written_bones: set = set()

    def _write_obj(obj):
        if obj is None or obj.name in written_objs:
            return
        if obj.name not in updated_names:
            return
        written_objs.add(obj.name)
        if not obj.animation_data or not obj.animation_data.action:
            return
        ad        = obj.animation_data
        live_vals = _obj_channel_values(obj)
        inf_vals  = _obj_constraint_influence_values(obj)
        for fc in _iter_fcurves(ad.action, ad):
            if (fc.data_path.startswith('constraints[') and
                    fc.data_path.endswith('].influence')):
                try:
                    cname = fc.data_path.split('"')[1]
                except IndexError:
                    continue
                if cname in inf_vals:
                    _write_fc_at(fc, frame, inf_vals[cname])
            else:
                k = (fc.data_path, fc.array_index)
                if k in live_vals:
                    _write_fc_at(fc, frame, live_vals[k])

    def _write_single_bone(arm, bone_name):
        """Write F-curves for one bone. Does NOT recurse into parents."""
        bkey = (arm.name, bone_name)
        if bkey in written_bones:
            return
        written_bones.add(bkey)
        if not arm.animation_data or not arm.animation_data.action:
            return
        ad     = arm.animation_data
        pb     = arm.pose.bones.get(bone_name)
        if pb is None:
            return
        prefix    = f'pose.bones["{bone_name}"].'
        live_vals = _bone_channel_values(arm, bone_name)
        inf_vals  = _constraint_influence_values(pb)
        for fc in _iter_fcurves(ad.action, ad):
            if not fc.data_path.startswith(prefix):
                continue
            suffix = fc.data_path[len(prefix):]
            if suffix.startswith('constraints[') and suffix.endswith('].influence'):
                try:
                    cname = suffix.split('"')[1]
                except IndexError:
                    continue
                if cname in inf_vals:
                    _write_fc_at(fc, frame, inf_vals[cname])
            else:
                k = (suffix, fc.array_index)
                if k in live_vals:
                    _write_fc_at(fc, frame, live_vals[k])

    def _write_bone_chain(arm, bone_name):
        """Write the bone and every bone-parent ancestor up the chain.

        bone_mat_from_action() evaluates F-curves for every bone in the chain
        to compute the subtarget's world position.  If any ancestor bone is
        being dragged and its F-curves are stale, the result is wrong — so we
        must stamp the live value for each one.

        The gate (arm.name in updated_names) is checked once here rather than
        per-bone: if the armature was touched by the depsgraph this tick, all
        of its dirty bones should be written.
        """
        if arm.name not in updated_names:
            return
        # Walk up the DATA bone parent chain (arm.data.bones, not pose.bones)
        # to get the full ancestor list including bones with no F-curves.
        eb = arm.data.bones.get(bone_name)
        if eb is None:
            return
        b = eb
        while b is not None:
            _write_single_bone(arm, b.name)
            b = b.parent

    def add_obj_parents(obj):
        cur = obj.parent
        while cur is not None:
            if cur.name not in tracked_names:
                _write_obj(cur)
            cur = cur.parent

    for node in props.nodes:
        if node.item_type != 'ENTRY' or not node.enabled:
            continue

        if node.track_type == 'BONE':
            arm = bpy.data.objects.get(node.armature_name)
            if arm is None:
                continue
            if arm.name not in tracked_names:
                add_obj_parents(arm)
            pb = arm.pose.bones.get(node.bone_name)
            if pb:
                for c in pb.constraints:
                    if c.mute:
                        continue
                    t = getattr(c, 'target', None)
                    if t is None or t.name in tracked_names:
                        continue
                    subtarget = getattr(c, 'subtarget', '')
                    if subtarget and t.type == 'ARMATURE':
                        # Write the subtarget bone AND its full bone-parent chain
                        _write_bone_chain(t, subtarget)
                    else:
                        _write_obj(t)
                    add_obj_parents(t)
        else:
            obj = bpy.data.objects.get(node.object_name)
            if obj is None:
                continue
            if obj.name not in tracked_names:
                add_obj_parents(obj)
            for c in obj.constraints:
                if c.mute:
                    continue
                t = getattr(c, 'target', None)
                if t is None or t.name in tracked_names:
                    continue
                subtarget = getattr(c, 'subtarget', '')
                if subtarget and t.type == 'ARMATURE':
                    _write_bone_chain(t, subtarget)
                else:
                    _write_obj(t)
                add_obj_parents(t)

def _write_live_pose_to_fcurves(scene, frame):
    """Write the current evaluated pose of every tracked entry into F-curves
    at *frame*.  Returns True if anything was written.

    Handles three cases:
      * BONE entries  -- loc/rot/scale channels + constraint influence channels
      * OBJECT entries with an action -- loc/rot/scale channels
      * OBJECT entries without an action -- update path cache directly from
        live matrix_world (no F-curve to write; path stays current in realtime)
    """
    props = scene.bmpl_props
    wrote_anything = False

    for node in props.nodes:
        if node.item_type != 'ENTRY' or not node.enabled:
            continue

        if node.track_type == 'BONE':
            arm = bpy.data.objects.get(node.armature_name)
            if arm is None or not arm.animation_data or not arm.animation_data.action:
                continue
            ad      = arm.animation_data
            action  = ad.action
            bn      = node.bone_name
            pb      = arm.pose.bones.get(bn)
            if pb is None:
                continue

            live_vals = _bone_channel_values(arm, bn)
            prefix    = f'pose.bones["{bn}"].'
            inf_vals  = _constraint_influence_values(pb)

            for fc in _iter_fcurves(action, ad):
                if not fc.data_path.startswith(prefix):
                    continue
                suffix = fc.data_path[len(prefix):]

                # Constraint influence: constraints["Name"].influence
                if suffix.startswith('constraints[') and suffix.endswith('].influence'):
                    try:
                        cname = suffix.split('"')[1]
                    except IndexError:
                        continue
                    if cname not in inf_vals:
                        continue
                    _write_fc_at(fc, frame, inf_vals[cname])
                    wrote_anything = True
                    continue

                # Regular transform channel
                key = (suffix, fc.array_index)
                if key not in live_vals:
                    continue
                _write_fc_at(fc, frame, live_vals[key])
                wrote_anything = True

        else:  # OBJECT
            obj = bpy.data.objects.get(node.object_name)
            if obj is None:
                continue

            if obj.animation_data and obj.animation_data.action:
                ad     = obj.animation_data
                action = ad.action
                live_vals = _obj_channel_values(obj)
                inf_vals  = _obj_constraint_influence_values(obj)
                for fc in _iter_fcurves(action, ad):
                    # Constraint influence: constraints["Name"].influence
                    if (fc.data_path.startswith('constraints[') and
                            fc.data_path.endswith('].influence')):
                        try:
                            cname = fc.data_path.split('"')[1]
                        except IndexError:
                            continue
                        if cname not in inf_vals:
                            continue
                        _write_fc_at(fc, frame, inf_vals[cname])
                        wrote_anything = True
                        continue
                    # Regular transform channel
                    key = (fc.data_path, fc.array_index)
                    if key not in live_vals:
                        continue
                    _write_fc_at(fc, frame, live_vals[key])
                    wrote_anything = True

            else:
                # Unkeyed object: no F-curve; stamp path cache directly so the
                # drawn path follows the object in realtime.
                uid = node.uid
                if uid not in _path_cache:
                    _path_cache[uid] = {}
                _path_cache[uid][frame] = obj.matrix_world.translation.copy()
                wrote_anything = True

    return wrote_anything







def _build_fcurve_fingerprint(scene):
    """Return a dict snapshot of every F-curve key position+value for all tracked entries.

    Key:   (action_as_pointer, data_path, array_index)
    Value: tuple of (frame, value, handles, interp, easing) rounded to 4 dp.

    Includes both the entry's own curves AND all related curves from:
      • Constraint target objects/bones (and their bone-parent chains)
      • Object-parent chains of the tracked object/armature
      • The armature's own transform curves (for BONE entries)

    This ensures that editing ANY influencing F-curve in the Graph Editor
    triggers a resample in KEYFRAME mode, not just the tracked bone/object's
    own curves.
    """
    props = scene.bmpl_props
    fp = {}

    def _fingerprint_fc(action, fc):
        key = (action.as_pointer(), fc.data_path, fc.array_index)
        if key in fp:
            return
        fp[key] = tuple(
            (
                round(kp.co[0], 4),           # frame
                round(kp.co[1], 4),           # value
                round(kp.handle_left[0], 4),  # left handle X
                round(kp.handle_left[1], 4),  # left handle Y
                round(kp.handle_right[0], 4), # right handle X
                round(kp.handle_right[1], 4), # right handle Y
                kp.interpolation,             # e.g. BEZIER / LINEAR / CONSTANT
                kp.easing,                    # easing modifier
            )
            for kp in fc.keyframe_points
        )

    for node in props.nodes:
        if node.item_type != 'ENTRY' or not node.enabled:
            continue
        # Own curves
        for action, ad, fc in _iter_all_fcurves_for_node(node):
            _fingerprint_fc(action, fc)
        # Related curves: constraint targets, parent chains, armature transforms
        for action, ad, fc in _iter_related_fcurves_for_node(node):
            _fingerprint_fc(action, fc)
    return fp


def _fcurves_changed(scene):
    """Return True when the F-curve fingerprint differs from the last stored one.
    Updates the stored fingerprint as a side-effect when a change is detected.
    """
    global _fcurve_fingerprint
    current = _build_fcurve_fingerprint(scene)
    if current != _fcurve_fingerprint:
        _fcurve_fingerprint = current
        return True
    return False


def _build_constraint_state_fingerprint(scene):
    """Return a lightweight snapshot of constraint properties for all tracked entries
    AND their related objects (constraint targets, parent chains).

    Captures: mute, influence, type, and (where applicable) target name + subtarget.
    This lets us detect constraint changes that don't touch any F-curve — e.g.
    muting a constraint, changing its type, or swapping its target object —
    even when the change is on a related object rather than the tracked entry itself.

    Key:   (owner_id, constraint_name)  — owner_id is a stable string per object/bone
    Value: tuple of relevant property values
    """
    props = scene.bmpl_props
    fp = {}

    def _snap_obj_constraints(obj, owner_id_prefix):
        """Add constraint snapshots for a plain object."""
        for c in obj.constraints:
            target_name = ''
            subtarget   = ''
            try:
                if hasattr(c, 'target') and c.target is not None:
                    target_name = c.target.name
                if hasattr(c, 'subtarget'):
                    subtarget = c.subtarget or ''
            except Exception:
                pass
            key = (owner_id_prefix + '|obj|' + obj.name, c.name)
            fp[key] = (c.type, c.mute, round(c.influence, 6), target_name, subtarget)

    def _snap_bone_chain_constraints(arm, bone_name, owner_id_prefix):
        """Add constraint snapshots for bone_name and all its ancestor bones."""
        eb = arm.data.bones.get(bone_name)
        if eb is None:
            return
        b = eb
        while b is not None:
            pb = arm.pose.bones.get(b.name)
            if pb is not None:
                for c in pb.constraints:
                    target_name = ''
                    subtarget   = ''
                    try:
                        if hasattr(c, 'target') and c.target is not None:
                            target_name = c.target.name
                        if hasattr(c, 'subtarget'):
                            subtarget = c.subtarget or ''
                    except Exception:
                        pass
                    key = (owner_id_prefix + '|bone|' + arm.name + '.' + b.name, c.name)
                    fp[key] = (c.type, c.mute, round(c.influence, 6), target_name, subtarget)
            b = b.parent

    visited_objs: set = set()   # object names already fingerprinted

    for node in props.nodes:
        if node.item_type != 'ENTRY' or not node.enabled:
            continue

        uid = node.uid

        if node.track_type == 'BONE':
            arm = bpy.data.objects.get(node.armature_name)
            if arm is None:
                continue
            pb = arm.pose.bones.get(node.bone_name)
            constraints = pb.constraints if pb is not None else []

            # Tracked bone's own constraints
            for c in constraints:
                target_name = ''
                subtarget   = ''
                try:
                    if hasattr(c, 'target') and c.target is not None:
                        target_name = c.target.name
                    if hasattr(c, 'subtarget'):
                        subtarget = c.subtarget or ''
                except Exception:
                    pass
                fp[(uid, c.name)] = (c.type, c.mute, round(c.influence, 6), target_name, subtarget)

            # Armature object-level constraints and parent chain
            if arm.name not in visited_objs:
                visited_objs.add(arm.name)
                _snap_obj_constraints(arm, uid)
                cur = arm.parent
                while cur is not None and cur.name not in visited_objs:
                    visited_objs.add(cur.name)
                    _snap_obj_constraints(cur, uid)
                    cur = cur.parent

            # Constraint targets of the tracked bone
            if pb:
                for c in pb.constraints:
                    if c.mute:
                        continue
                    t = getattr(c, 'target', None)
                    if t is None:
                        continue
                    subtarget = getattr(c, 'subtarget', '')
                    if subtarget and t.type == 'ARMATURE':
                        _snap_bone_chain_constraints(t, subtarget, uid)
                        if t.name not in visited_objs:
                            visited_objs.add(t.name)
                            _snap_obj_constraints(t, uid)
                    else:
                        if t.name not in visited_objs:
                            visited_objs.add(t.name)
                            _snap_obj_constraints(t, uid)
                            cur = t.parent
                            while cur is not None and cur.name not in visited_objs:
                                visited_objs.add(cur.name)
                                _snap_obj_constraints(cur, uid)
                                cur = cur.parent

        else:  # OBJECT
            obj = bpy.data.objects.get(node.object_name)
            if obj is None:
                continue

            # Tracked object's own constraints
            for c in obj.constraints:
                target_name = ''
                subtarget   = ''
                try:
                    if hasattr(c, 'target') and c.target is not None:
                        target_name = c.target.name
                    if hasattr(c, 'subtarget'):
                        subtarget = c.subtarget or ''
                except Exception:
                    pass
                fp[(uid, c.name)] = (c.type, c.mute, round(c.influence, 6), target_name, subtarget)

            # Object parent chain
            if obj.name not in visited_objs:
                visited_objs.add(obj.name)
            cur = obj.parent
            while cur is not None and cur.name not in visited_objs:
                visited_objs.add(cur.name)
                _snap_obj_constraints(cur, uid)
                cur = cur.parent

            # Constraint targets of the tracked object
            for c in obj.constraints:
                if c.mute:
                    continue
                t = getattr(c, 'target', None)
                if t is None:
                    continue
                subtarget = getattr(c, 'subtarget', '')
                if subtarget and t.type == 'ARMATURE':
                    _snap_bone_chain_constraints(t, subtarget, uid)
                    if t.name not in visited_objs:
                        visited_objs.add(t.name)
                        _snap_obj_constraints(t, uid)
                else:
                    if t.name not in visited_objs:
                        visited_objs.add(t.name)
                        _snap_obj_constraints(t, uid)
                        cur2 = t.parent
                        while cur2 is not None and cur2.name not in visited_objs:
                            visited_objs.add(cur2.name)
                            _snap_obj_constraints(cur2, uid)
                            cur2 = cur2.parent

    return fp


def _constraint_state_changed(scene):
    """Return True when constraint state differs from the last stored snapshot.
    Updates the stored fingerprint as a side-effect when a change is detected.
    """
    global _constraint_state_fingerprint
    current = _build_constraint_state_fingerprint(scene)
    if current != _constraint_state_fingerprint:
        _constraint_state_fingerprint = current
        return True
    return False


def do_resample(scene):
    props = scene.bmpl_props
    if props.update_mode == 'MANUAL': return
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


def _related_object_names(scene):
    """Return a set of object names that, when transformed, should trigger a
    write + resample for any tracked entry.

    Includes for each tracked entry:
      • The tracked object itself (or armature for BONE entries)
      • All constraint target objects (and their armatures for subtargets)
      • The full object-level parent chain of the above
      • For bone subtargets: the containing armature is already included, so
        moving ANY bone in that armature (including parent bones of the
        subtarget) will trigger the gate correctly.
    """
    props = scene.bmpl_props
    names: set = set()

    def add_obj_parents(obj):
        cur = obj
        while cur is not None:
            names.add(cur.name)
            cur = cur.parent

    for node in props.nodes:
        if node.item_type != 'ENTRY' or not node.enabled:
            continue

        if node.track_type == 'BONE':
            arm = bpy.data.objects.get(node.armature_name)
            if arm is None:
                continue
            names.add(arm.name)
            add_obj_parents(arm)
            pb = arm.pose.bones.get(node.bone_name)
            if pb:
                for c in pb.constraints:
                    if c.mute:
                        continue
                    t = getattr(c, 'target', None)
                    if t is not None:
                        names.add(t.name)
                        add_obj_parents(t)
                        # For subtarget bones: the armature name is already
                        # added above.  Any bone movement in that armature
                        # marks the armature in the depsgraph update set, so
                        # parent bones of the subtarget are covered implicitly.
        else:
            obj = bpy.data.objects.get(node.object_name)
            if obj is None:
                continue
            names.add(obj.name)
            add_obj_parents(obj)
            for c in obj.constraints:
                if c.mute:
                    continue
                t = getattr(c, 'target', None)
                if t is not None:
                    names.add(t.name)
                    add_obj_parents(t)

    return names


@bpy.app.handlers.persistent
def on_depsgraph(scene, depsgraph):
    # Cleanup orphans on object/armature changes
    for u in depsgraph.updates:
        if u.id and u.id.id_type in ('OBJECT', 'ARMATURE'):
            cleanup_orphans(scene)
            break

    try:
        update_mode = scene.bmpl_props.update_mode
        if update_mode == 'MANUAL':
            return
    except AttributeError:
        return

    # ── KEYFRAME mode ─────────────────────────────────────────────────────────
    # Read-only: resample whenever the F-curve data actually changes,
    # OR whenever constraint properties change (mute, influence, type, target)
    # even if those constraints have no F-curves.
    if update_mode == 'KEYFRAME':
        if _fcurves_changed(scene) or _constraint_state_changed(scene):
            do_resample(scene)
        return



# ─── Handler Registration ─────────────────────────────────────────────────────

def register_handlers():
    global _draw_handler, _text_handler, _fcurve_fingerprint, _constraint_state_fingerprint
    global _live_matrix_overrides, _live_bone_overrides
    _fcurve_fingerprint              = {}     # reset on (re)register
    _constraint_state_fingerprint    = {}     # reset on (re)register
    _live_matrix_overrides           = {}
    _live_bone_overrides             = {}
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_paths, (), 'WINDOW', 'POST_VIEW')
    if _text_handler is None:
        _text_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_frame_numbers_2d, (), 'WINDOW', 'POST_PIXEL')
    if on_depsgraph not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(on_depsgraph)


def unregister_handlers():
    global _draw_handler, _text_handler, _constraint_state_fingerprint
    global _live_matrix_overrides, _live_bone_overrides
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None
    if _text_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_text_handler, 'WINDOW')
        _text_handler = None
    if on_depsgraph in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph)
    _constraint_state_fingerprint = {}
    _live_matrix_overrides        = {}
    _live_bone_overrides          = {}


# ─── Property Update Callbacks ────────────────────────────────────────────────

def on_dots_update(self, context):
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def on_show_path_update(self, context):
    """When Paths visibility is toggled:
    - Toggled OFF: save current Dots & Frame Numbers states, then hide both.
    - Toggled ON:  restore the previously saved Dots & Frame Numbers states.
    This avoids broken dot positions when the path geometry is hidden,
    and avoids the performance cost of computing dot positions with no path.
    """
    if not self.show_path:
        # Save current states before hiding
        self.dots_saved_before_path_hide   = self.show_dots
        self.fnums_saved_before_path_hide  = self.show_frame_numbers
        # Suppress their own update callbacks by setting directly
        # (BoolProperty update fires, but we want that for the redraw)
        self.show_dots          = False
        self.show_frame_numbers = False
    else:
        # Restore saved states
        self.show_dots          = self.dots_saved_before_path_hide
        self.show_frame_numbers = self.fnums_saved_before_path_hide
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def on_update_mode_change(self, context):
    # When switching to Keyframe Update, do an immediate resample
    # so the path appears without waiting for the next depsgraph event.
    global _fcurve_fingerprint, _constraint_state_fingerprint
    if self.update_mode == 'KEYFRAME':
        # Seed both fingerprints so the *next* change (not the switch itself)
        # triggers the resample.
        _fcurve_fingerprint           = _build_fcurve_fingerprint(context.scene)
        _constraint_state_fingerprint = _build_constraint_state_fingerprint(context.scene)
    if self.update_mode == 'KEYFRAME':
        do_resample(context.scene)


def on_filter_update(self, context):
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


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
    col_in_front:      bpy.props.BoolProperty(default=False)
    col_enabled:       bpy.props.BoolProperty(default=True)
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
    dot_color_mode:    bpy.props.EnumProperty(
        items=[('SOLID','Solid','Single color for all dots'),
               ('SPLIT','Split','Separate before/after colors')],
        default='SOLID')
    dot_color:         bpy.props.FloatVectorProperty(
        subtype='COLOR_GAMMA', size=4, min=0, max=1, default=(1.0,1.0,1.0,1.0))
    dot_color_before:  bpy.props.FloatVectorProperty(
        subtype='COLOR_GAMMA', size=4, min=0, max=1, default=(1.0,1.0,1.0,1.0))
    dot_color_after:   bpy.props.FloatVectorProperty(
        subtype='COLOR_GAMMA', size=4, min=0, max=1, default=(1.0,1.0,1.0,1.0))


class QPRK_SceneProps(bpy.types.PropertyGroup):
    nodes:          bpy.props.CollectionProperty(type=QPRK_Node)
    active_uid:     bpy.props.StringProperty(default="")
    # Global frame range (override)
    frame_start:    bpy.props.IntProperty(name="Start", default=1)
    frame_end:      bpy.props.IntProperty(name="End",   default=250)
    frame_step:     bpy.props.IntProperty(name="Step",  default=1, min=1, max=10)
    show_path:      bpy.props.BoolProperty(
        name="Show Paths", default=True,
        update=on_show_path_update)
    show_dots:      bpy.props.BoolProperty(
        name="Show Dots", default=True,
        update=on_dots_update)
    show_frame_numbers: bpy.props.BoolProperty(
        name="Show Frame Numbers", default=False,
        update=on_dots_update)
    # Internal saved states used by on_show_path_update (not shown in UI)
    dots_saved_before_path_hide:  bpy.props.BoolProperty(default=True)
    fnums_saved_before_path_hide: bpy.props.BoolProperty(default=False)
    update_mode: bpy.props.EnumProperty(
        name="Update Mode",
        default='MANUAL',
        update=on_update_mode_change,
        items=[
            ('MANUAL',   "Manual Update",
             "Path only updates when you press Calculate. No automatic resampling."),
            ('KEYFRAME', "Keyframe Update",
             "Path updates after each confirmed keyframe. Does not write changes to F-curves "
             "while moving — only recalculates once the transform is finalized."),
        ])
    filter_text:    bpy.props.StringProperty(
        name="Filter", default="", update=on_filter_update)
    filter_invert:  bpy.props.BoolProperty(name="Invert", default=False)
    sort_mode:      bpy.props.EnumProperty(
        name="Sort", default='DEFAULT',
        items=[('DEFAULT','Default','Manual order'),
               ('AZ','A→Z','Alphabetical ascending'),
               ('ZA','Z→A','Alphabetical descending')])
    reparent_target: bpy.props.StringProperty(default="")
    list_rows:       bpy.props.IntProperty(name="List Rows", default=0, min=0, max=40,
                         description="Extra rows for the tracked-items list (0 = auto min-height)")

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
        mode = context.mode  # 'OBJECT', 'POSE', etc.

        sel_bones = []
        # Only gather selected bones when in Pose Mode
        if mode == 'POSE' and active_obj and active_obj.type == 'ARMATURE' and active_obj.pose:
            sel_bones = [pb for pb in active_obj.pose.bones if getattr(pb, 'select', False) or getattr(pb.bone, 'select', False)]

        if sel_bones:
            # Pose Mode with selected bones → track each bone
            for pb in sel_bones:
                if self.already(props,'BONE',active_obj.name,pb.name): continue
                n = props.nodes.add()
                n.uid=make_uid(); n.parent_uid=p_uid; n.item_type='ENTRY'; n.track_type='BONE'
                n.armature_name=active_obj.name; n.bone_name=pb.name
                n.label=f"{pb.name} / {active_obj.name}"
                n.order=next_order(props,p_uid)
                n.frame_start=scene.frame_start; n.frame_end=scene.frame_end
                last=n.uid; added+=1
        else:
            # Object Mode (or Pose Mode with no bones selected) →
            # track the origin of each selected object, including armatures.
            objs = [o for o in scene.objects if o.select_get()] or ([active_obj] if active_obj else [])
            for obj in objs:
                if self.already(props,'OBJECT',obj=obj.name): continue
                n = props.nodes.add()
                n.uid=make_uid(); n.parent_uid=p_uid; n.item_type='ENTRY'; n.track_type='OBJECT'
                n.object_name=obj.name
                n.label=obj.name
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
        props   = context.scene.bmpl_props
        checked = checked_entries(props)

        # Decide where the new folder lives:
        # - If active node is a COLLECTION, nest inside it.
        # - Everything else (entry active, nothing active) → root.
        active = get_node(props, props.active_uid)
        if active and active.item_type == 'COLLECTION':
            p_uid = active.uid
        else:
            p_uid = ''

        # Create the new folder
        n = props.nodes.add()
        n.uid       = make_uid()
        n.parent_uid = p_uid
        n.item_type  = 'COLLECTION'
        n.label      = "Collection"
        n.order      = next_order(props, p_uid)
        new_uid      = n.uid

        # Move any checked entries into the new folder
        if checked:
            for entry in checked:
                entry.parent_uid = new_uid
                entry.order      = next_order(props, new_uid)

        # Keep the active selection unchanged
        return {'FINISHED'}


class QPRK_OT_Remove(bpy.types.Operator):
    bl_idname = "qprk.remove"; bl_label = "Remove"
    def collect(self, props, uid):
        u=[uid]
        for c in get_children(props,uid): u.extend(self.collect(props,c.uid))
        return u
    def execute(self, context):
        props   = context.scene.bmpl_props
        checked = checked_entries(props)
        checked_cols = [n for n in props.nodes if n.item_type == 'COLLECTION' and n.col_calc_selected]
        if checked or checked_cols:
            # Remove all checked entries AND all checked collections (with their descendants)
            uids = []
            for node in checked_cols + checked:
                uids.extend(self.collect(props, node.uid))
            uids = list(dict.fromkeys(uids))  # deduplicate, preserve order
        else:
            # Nothing checked — remove the active node
            active = get_node(props, props.active_uid)
            if active is None: return {'CANCELLED'}
            uids = self.collect(props, active.uid)
        for uid in uids: _path_cache.pop(uid, None)
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
    """Move checked entries (or active node) to a different folder or root"""
    bl_idname = "qprk.reparent"; bl_label = "Move To..."
    def invoke(self, context, event):
        context.scene.bmpl_props.reparent_target = ""
        return context.window_manager.invoke_props_dialog(self, width=280)
    def draw(self, context):
        layout = self.layout
        props  = context.scene.bmpl_props
        sel    = props.reparent_target

        # Show how many items will be moved
        checked = checked_entries(props)
        active  = get_node(props, props.active_uid)
        if checked:
            layout.label(text=f"Move {len(checked)} checked entr{'y' if len(checked)==1 else 'ies'} to:")
        elif active:
            layout.label(text=f"Move '{active.label or 'node'}' to:")
        else:
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
        props      = context.scene.bmpl_props
        target_uid = props.reparent_target

        # Validate target (if not root)
        if target_uid:
            tgt = get_node(props, target_uid)
            if tgt is None or tgt.item_type != 'COLLECTION':
                self.report({'WARNING'},"Invalid target"); return {'CANCELLED'}

        # Decide what to move: checked entries, or fall back to active node
        checked = checked_entries(props)
        if checked:
            nodes_to_move = checked
        else:
            node = get_node(props, props.active_uid)
            if node is None: return {'CANCELLED'}
            nodes_to_move = [node]

        moved = 0
        for node in nodes_to_move:
            # Prevent moving a folder into itself or a descendant
            if target_uid:
                check = get_node(props, target_uid)
                is_ancestor = False
                while check:
                    if check.uid == node.uid:
                        is_ancestor = True; break
                    check = get_node(props, check.parent_uid)
                if is_ancestor:
                    self.report({'WARNING'}, f"Cannot move '{node.label}' into itself — skipped")
                    continue
            node.parent_uid = target_uid
            node.order      = next_order(props, target_uid)
            moved += 1

        props.reparent_target = ""
        if moved:
            self.report({'INFO'}, f"Moved {moved} item{'s' if moved!=1 else ''}")
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
        return _invoke_warning_or_calc(context, targets)


class QPRK_OT_CalculateEntry(bpy.types.Operator):
    """Calculate path for this single entry"""
    bl_idname = "qprk.calculate_entry"; bl_label = "Calculate"
    uid: bpy.props.StringProperty()
    def execute(self, context):
        props = context.scene.bmpl_props
        node  = get_node(props, self.uid)
        if node is None or node.item_type != 'ENTRY': return {'CANCELLED'}
        return _invoke_warning_or_calc(context, [node])


class QPRK_OT_CalculateGlobal(bpy.types.Operator):
    """Calculate all entries using the global frame range"""
    bl_idname = "qprk.calculate_global"; bl_label = "Calculate All (Global Range)"
    def execute(self, context):
        props   = context.scene.bmpl_props
        targets = [n for n in props.nodes if n.item_type == 'ENTRY']
        if not targets:
            self.report({'WARNING'},"No entries to calculate"); return {'CANCELLED'}
        saved = [(n, n.frame_start, n.frame_end, n.frame_step) for n in targets]
        for n in targets:
            n.frame_start = props.frame_start
            n.frame_end   = props.frame_end
            n.frame_step  = props.frame_step
        return _invoke_warning_or_calc(context, targets, overrides=saved)


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
        saved = [(n, n.frame_start, n.frame_end, n.frame_step) for n in targets]
        for n in targets:
            n.frame_start = active.frame_start
            n.frame_end   = active.frame_end
            n.frame_step  = active.frame_step
        return _invoke_warning_or_calc(context, targets, overrides=saved)


class QPRK_OT_ToggleNodeFront(bpy.types.Operator):
    """Toggle In Front. If entries are checked, toggles all checked entries. For a collection, toggles all its entries."""
    bl_idname = "qprk.toggle_node_front"; bl_label = "Toggle In Front"
    uid: bpy.props.StringProperty()
    def execute(self, context):
        props = context.scene.bmpl_props
        node  = get_node(props, self.uid)
        if node is None: return {'CANCELLED'}
        if node.item_type == 'COLLECTION':
            entries = all_entry_descendants(props, node.uid)
            if not entries:
                node.col_in_front = not node.col_in_front
            else:
                v = not all(e.in_front for e in entries)
                for e in entries: e.in_front = v
                node.col_in_front = v
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
        if node.item_type == 'COLLECTION':
            entries = all_entry_descendants(props, node.uid)
            if not entries:
                node.col_enabled = not node.col_enabled
            else:
                v = not all(e.enabled for e in entries)
                for e in entries: e.enabled = v
                node.col_enabled = v
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
        props = context.scene.bmpl_props
        entries = [n for n in props.nodes if n.item_type=='ENTRY']
        if not entries: return {'CANCELLED'}
        v = not all(e.in_front for e in entries)
        for e in entries: e.in_front = v
        for c in [n for n in props.nodes if n.item_type=='COLLECTION']:
            c.col_in_front = v
        for area in context.screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
        return {'FINISHED'}


class QPRK_OT_ToggleAllVis(bpy.types.Operator):
    bl_idname = "qprk.toggle_all_vis"; bl_label = "Toggle All Visibility"
    def execute(self, context):
        props = context.scene.bmpl_props
        entries = [n for n in props.nodes if n.item_type=='ENTRY']
        if not entries: return {'CANCELLED'}
        v = not all(e.enabled for e in entries)
        for e in entries: e.enabled = v
        for c in [n for n in props.nodes if n.item_type=='COLLECTION']:
            c.col_enabled = v
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
            n.dot_color_mode   = active.dot_color_mode
            n.dot_color        = active.dot_color[:]
            n.dot_color_before = active.dot_color_before[:]
            n.dot_color_after  = active.dot_color_after[:]
        self.report({'INFO'},f"Applied to {len(targets)} entr{'y' if len(targets)==1 else 'ies'}")
        return {'FINISHED'}


# ─── Keyframe Warning Dialog ──────────────────────────────────────────────────

_pending_calc: dict = {}

def _run_pending_calc(context):
    job = _pending_calc.pop('pending', None)
    _pending_calc.pop('missing', None)
    if job is None: return
    scene = context.scene
    for node in job['nodes']:
        sample_entry(scene, node)
    for n, fs, fe, fst in (job.get('overrides') or []):
        n.frame_start = fs; n.frame_end = fe; n.frame_step = fst
    for area in context.screen.areas:
        if area.type == 'VIEW_3D': area.tag_redraw()


def _insert_transform_keyframe(obj, frame, bone_name=""):
    if bone_name:
        pb = obj.pose.bones.get(bone_name)
        if pb is None: return
        pb.keyframe_insert('location',            frame=frame)
        pb.keyframe_insert('rotation_euler',      frame=frame)
        pb.keyframe_insert('rotation_quaternion', frame=frame)
        pb.keyframe_insert('scale',               frame=frame)
    else:
        obj.keyframe_insert('location',      frame=frame)
        obj.keyframe_insert('rotation_euler', frame=frame)
        obj.keyframe_insert('scale',          frame=frame)


class QPRK_OT_KeyframeWarning(bpy.types.Operator):
    """Warning dialog: missing transform keyframes detected"""
    bl_idname  = "qprk.keyframe_warning"
    bl_label   = "Missing Transform Keyframes"
    bl_options = {'REGISTER', 'INTERNAL'}

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=480)

    def draw(self, context):
        layout = self.layout
        missing = _pending_calc.get('missing', [])

        col = layout.column(align=True)
        col.label(text="Missing Transform Keyframes", icon='ERROR')
        col.label(text="Some objects/bones have no transform keyframes.")
        col.label(text="The fast path cannot be used for these entries.")
        layout.separator()

        box = layout.box()
        box.label(text="Items that need keyframes:", icon='KEYFRAME')
        if missing:
            for item in missing:
                row = box.row(align=True)
                row.label(text="", icon='DOT')
                sub = row.column(align=True)
                name = item['obj_name']
                if item['bone_name']:
                    name += f"  /  {item['bone_name']}"
                sub.label(text=name)
                sub.label(text=item['reason'], icon='INFO')
        else:
            box.label(text="(none found — this is a bug, please report)", icon='QUESTION')

        layout.separator()
        col2 = layout.column(align=True)
        col2.label(text="Add Keyframes & Calculate:", icon='KEYFRAME_HLT')
        col2.label(text="  Inserts a loc/rot/scale key at the current frame on each")
        col2.label(text="  missing item, then calculates using the fast path.")
        col2.separator()
        col2.label(text="Calculate Anyway:", icon='PLAY')
        col2.label(text="  Skips keyframing, uses frame_set fallback (slower but correct).")
        layout.separator()

        row = layout.row(align=True)
        row.scale_y = 1.3
        row.operator("qprk.keyframe_and_calculate", text="Add Keyframes & Calculate",
                     icon='KEYFRAME_HLT')
        row.operator("qprk.calc_anyway", text="Calculate Anyway", icon='PLAY')

    def execute(self, context):
        return {'FINISHED'}


class QPRK_OT_CalcAnyway(bpy.types.Operator):
    """Calculate using frame_set fallback without adding keyframes"""
    bl_idname  = "qprk.calc_anyway"
    bl_label   = "Calculate Anyway"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        _run_pending_calc(context)
        return {'FINISHED'}


class QPRK_OT_KeyframeAndCalculate(bpy.types.Operator):
    """Add keyframes on all missing items then run the pending calculation"""
    bl_idname  = "qprk.keyframe_and_calculate"
    bl_label   = "Add Keyframes & Calculate"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        frame   = context.scene.frame_current
        missing = _pending_calc.get('missing', [])
        seen    = set()
        for item in missing:
            key = (item['obj_name'], item['bone_name'])
            if key in seen: continue
            seen.add(key)
            obj = bpy.data.objects.get(item['obj_name'])
            if obj is None: continue
            try:
                _insert_transform_keyframe(obj, frame, item['bone_name'])
            except Exception as e:
                self.report({'WARNING'}, f"Could not key {item['obj_name']}: {e}")

        _run_pending_calc(context)
        self.report({'INFO'}, f"Added keyframes on {len(seen)} item(s) and calculated")
        return {'FINISHED'}


def _invoke_warning_or_calc(context, nodes, overrides=None):
    missing = collect_missing_keys(nodes)
    job = {'nodes': nodes, 'overrides': overrides or []}
    _pending_calc['pending'] = job
    _pending_calc['missing'] = missing
    if not missing:
        _run_pending_calc(context)
        return {'FINISHED'}
    bpy.ops.qprk.keyframe_warning('INVOKE_DEFAULT')
    return {'FINISHED'}



# ─── Slow Path Tooltip Operator ───────────────────────────────────────────────

class QPRK_OT_SlowPathInfo(bpy.types.Operator):
    """This entry uses Blender's slower frame_set path because one or more
objects in its chain (the armature, a constraint target, or a parent) lack
transform keyframes. Click to see which objects are missing keyframes."""
    bl_idname  = "qprk.slow_path_info"
    bl_label   = "Uses Slow Path"
    bl_options = {'REGISTER', 'INTERNAL'}

    uid: bpy.props.StringProperty(default="")

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=420)

    def draw(self, context):
        layout  = self.layout
        props   = context.scene.bmpl_props
        node    = get_node(props, self.uid)
        if node is None:
            layout.label(text="Entry not found.", icon='ERROR')
            return

        layout.label(text=f"'{node.label or node.bone_name}' — Slow Path (frame_set)",
                     icon='ERROR')
        layout.separator()

        missing = collect_missing_keys([node])
        if missing:
            box = layout.box()
            box.label(text="Missing transform keyframes:", icon='KEYFRAME')
            for item in missing:
                row = box.row(align=True)
                row.label(text="", icon='DOT')
                sub = row.column(align=True)
                name = item['obj_name']
                if item['bone_name']:
                    name += f"  /  {item['bone_name']}"
                sub.label(text=name)
                sub.label(text=item['reason'], icon='INFO')
        else:
            layout.label(text="Entry has non-standard constraints that require", icon='INFO')
            layout.label(text="Blender's full depsgraph evaluation (frame_set).")

        layout.separator()
        layout.label(text="To use the fast path: add transform keyframes on the",  icon='KEYFRAME_HLT')
        layout.label(text="listed objects, or use 'Add Keyframes & Calculate'.")

    def execute(self, context):
        return {'FINISHED'}


class QPRK_OT_ToggleDotColorMode(bpy.types.Operator):
    """Toggle dot color mode between Solid and Split"""
    bl_idname = "qprk.toggle_dot_color_mode"; bl_label = "Toggle Dot Color Mode"
    uid: bpy.props.StringProperty()
    def execute(self, context):
        node = get_node(context.scene.bmpl_props, self.uid)
        if node is None: return {'CANCELLED'}
        node.dot_color_mode = 'SPLIT' if node.dot_color_mode == 'SOLID' else 'SOLID'
        return {'FINISHED'}


class QPRK_OT_CycleUpdateMode(bpy.types.Operator):
    """Cycle update mode:
Manual Update – path only updates on Calculate
Keyframe Update – path updates after each confirmed keyframe (no live F-curve writing during drag)"""
    bl_idname = "qprk.cycle_update_mode"
    bl_label  = "Cycle Update Mode"

    def execute(self, context):
        props = context.scene.bmpl_props
        cycle = {'MANUAL': 'KEYFRAME', 'KEYFRAME': 'MANUAL'}
        props.update_mode = cycle.get(props.update_mode, 'MANUAL')
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
        row = col.row(align=True)
        if depth: row.separator(factor=depth*2.2)

        if node.item_type == 'COLLECTION':
            # Use split like ENTRY rows so the right-side buttons stay bounded
            # and don't push the sidebar column off screen.
            outer = row.split(factor=0.52, align=True)

            left = outer.row(align=True)
            op = left.operator("qprk.toggle_col_calc",text='',icon='CHECKBOX_HLT' if node.col_calc_selected else 'CHECKBOX_DEHLT',
                              emboss=False)
            op.uid = node.uid
            op2 = left.operator("qprk.toggle_collapse",text='',icon='TRIA_DOWN' if not node.collapsed else 'TRIA_RIGHT',
                               emboss=False)
            op2.uid = node.uid
            sub = left.row(align=True); sub.alignment='LEFT'
            op3 = sub.operator("qprk.set_active", text=node.label or "Collection",
                               icon='FILE_FOLDER', emboss=is_active)
            op3.uid = node.uid

            right = outer.row(align=True); right.alignment='RIGHT'
            entries = all_entry_descendants(props, node.uid)
            col_in_front = entries and all(e.in_front for e in entries) or (not entries and node.col_in_front)
            col_enabled  = entries and all(e.enabled  for e in entries) or (not entries and node.col_enabled)
            right.separator(factor=2.0)
            opf = right.operator("qprk.toggle_node_front", text='',
                                 icon='XRAY' if col_in_front else 'MESH_CUBE',
                                 emboss=True, depress=col_in_front)
            opf.uid = node.uid
            opv = right.operator("qprk.toggle_node_vis", text='',
                                 icon='HIDE_OFF' if col_enabled else 'HIDE_ON',
                                 emboss=True, depress=col_enabled)
            opv.uid = node.uid
        else:
            slow = entry_uses_slow_path(node)
            # Split the row: left side = checkbox + name, right side = color + buttons.
            # Using a top-level split means the right block is always the same absolute
            # width regardless of how long the name is, so color slot & buttons never shift.
            outer = row.split(factor=0.52, align=True)

            # Left half: checkbox + icon + name (text left-aligned, truncates on overflow)
            left = outer.row(align=True)
            left.prop(node, "calc_selected", text="")
            name_row = left.row(align=True)
            name_row.active = node.enabled
            name_row.alignment = 'LEFT'
            if slow: name_row.alert = True
            icon = 'BONE_DATA' if node.track_type == 'BONE' else 'OBJECT_DATA'
            op = name_row.operator("qprk.set_active", text=node.label or "Entry",
                                   icon=icon, emboss=is_active)
            op.uid = node.uid

            # Right half: [warn?] + color slot + separator + front + vis
            right = outer.row(align=True)
            right.alignment = 'RIGHT'

            # slow-path warning icon (optional)
            if slow:
                op_slow = right.operator("qprk.slow_path_info", text='',
                                         icon='ERROR', emboss=False)
                op_slow.uid = node.uid

            # Color slot: a plain row(align=True) with two props placed
            # side-by-side. align=True removes the gutter between siblings
            # so the two squares touch with no gap — in solid mode they show
            # the same color and look like one continuous rectangle.
            color_row = right.row(align=True)
            if node.use_frame_colors:
                left = color_row.row(align=True)
                left.scale_x = 0.5
                left.prop(node, "color_before", text="")

                right_sub = color_row.row(align=True)
                right_sub.scale_x = 0.5
                right_sub.prop(node, "color_after", text="")

            else:
                solid = color_row.row(align=True)
                solid.scale_x = 0.72
                solid.prop(node, "path_color", text="")

            # Breathing room before icon buttons
            right.separator(factor=2.0)

            # Front button (toggle prop — always visible, blue when on)
            right.prop(node, "in_front", text='',
                       icon='XRAY' if node.in_front else 'MESH_CUBE', toggle=True)

            # Vis button (toggle prop — always visible, blue when on)
            right.prop(node, "enabled", text='',
                       icon='HIDE_OFF' if node.enabled else 'HIDE_ON', toggle=True)

    def draw(self, context):
        layout = self.layout
        props  = context.scene.bmpl_props

        # ── Visibility header ──
        # Use a row-in-box so the header label row has the lighter single-box background.
        vis_box = layout.box()
        hdr_box = vis_box.box()
        hdr_row = hdr_box.row(align=True)
        hdr_row.alignment = 'CENTER'
        hdr_row.label(text="QuickPath RKNZ", icon='ANIM')
        row = vis_box.row(align=True)
        row.prop(props,"show_path",text="Paths",
                 icon='HIDE_OFF' if props.show_path else 'HIDE_ON', toggle=True)
        row.prop(props,"show_dots",text="Dots",
                 icon='KEYFRAME_HLT' if props.show_dots else 'KEYFRAME', toggle=True)
        row.prop(props,"show_frame_numbers",text="Frames",
                 icon='LINENUMBERS_ON' if props.show_frame_numbers else 'LINENUMBERS_OFF', toggle=True)

        # ── Tree list + vertical sidebar ──
        draw_order = build_draw_order(props)
        all_entries = [n for n in props.nodes if n.item_type == 'ENTRY']

        list_row = layout.row(align=False)
        box = list_row.box()
        box.scale_x = 1.0

        # ── Title header — single box (lighter) instead of nested box ─────────
        title_hdr_box = box.box()
        title_row = title_hdr_box.row(align=True)
        title_row.alignment = 'CENTER'
        title_row.label(text="Tracked Items", icon='ARMATURE_DATA')

        # ── Column sub-header ─────────────────────────────────────────────────
        all_calc  = bool(all_entries) and all(e.calc_selected for e in all_entries)
        all_front = bool(all_entries) and all(e.in_front      for e in all_entries)
        all_vis   = bool(all_entries) and all(e.enabled        for e in all_entries)

        list_container = box.column(align=True)

        col_hdr_box = list_container.box()
        col_hdr = col_hdr_box.row(align=True)
        col_hdr.scale_y = 0.85

        # Mirror the exact split structure used in draw_node_row for ENTRY rows
        # so the header checkbox lands directly above the per-entry checkboxes.
        # draw_node_row does: row.separator(factor=depth*2.2) then row.split(0.52)
        # For depth=0 there is no leading separator, so we match directly.
        hdr_outer = col_hdr.split(factor=0.52, align=True)

        # Left half: checkbox flush-left + "Item Name" label
        left_inner = hdr_outer.row(align=True)
        op_calc = left_inner.operator("qprk.toggle_all_calc",
                                      icon='CHECKBOX_HLT' if all_calc else 'CHECKBOX_DEHLT',
                                      text="", emboss=False)
        name_lbl = left_inner.row(align=True)
        name_lbl.alignment = 'CENTER'
        name_lbl.label(text="Item Name")

        # Right half: "Line Color" label + front/vis buttons flush-right
        right_inner = hdr_outer.row(align=True)
        lc_lbl = right_inner.row(align=True)
        lc_lbl.alignment = 'EXPAND'
        lc_lbl.label(text="Line Color")

        right_btns = right_inner.row(align=True)
        right_btns.alignment = 'RIGHT'
        right_btns.separator(factor=2.0)   # mirror the separator before front button in rows
        op_front = right_btns.operator("qprk.toggle_all_front",
                                       icon='XRAY' if all_front else 'MESH_CUBE',
                                       text="", emboss=True, depress=all_front)
        op_vis = right_btns.operator("qprk.toggle_all_vis",
                                     icon='HIDE_OFF' if all_vis else 'HIDE_ON',
                                     text="", emboss=True, depress=all_vis)

        # ── Spacer to ensure minimum list width ───────────────────────────────
        spacer = list_container.row()
        spacer.label(text=" " * 60)
        spacer.scale_y = 0.01

        # ── List entries with scrollable / resizable container ────────────────
        # Sidebar has 9 buttons at scale_y=1.1 ≈ 9 rows; enforce that as min.
        # props.list_rows==0 means auto (show all entries, min = sidebar height).
        # props.list_rows>0 means user has dragged the handle to a custom height.
        SIDEBAR_ROWS = 9   # deselect + sep + add + folder + sep + up + down + sep + rename + reparent + sep + remove
        min_rows = SIDEBAR_ROWS

        if draw_order:
            total_nodes = len(draw_order)
            if props.list_rows > 0:
                visible_rows = max(min_rows, props.list_rows)
            else:
                visible_rows = max(min_rows, total_nodes)

            # Determine scroll offset so active item stays visible
            # We store nothing extra; just clip to visible_rows from the top.
            # Blender panels don't support native scroll, so we do a simple clip
            # and show a scroll hint when content is cut.
            shown = draw_order[:visible_rows]
            clipped = total_nodes - len(shown)

            for node in shown:
                row_box = list_container.box()
                row_box.scale_y = 0.85
                self.draw_node_row(row_box, node, props, get_depth(props, node))

            # Pad with empty rows when fewer entries than min_rows
            pad_needed = visible_rows - len(shown)
            for _ in range(max(0, pad_needed)):
                pad_row = list_container.row()
                pad_row.scale_y = 0.85
                pad_row.label(text="")

            # ── Resize handle (dots) ──────────────────────────────────────────

        else:
            list_container.label(text="No items yet.", icon='INFO')
            # Still show padding + handle so sidebar height is respected
            for _ in range(min_rows - 1):
                pad_row = list_container.row()
                pad_row.scale_y = 0.85
                pad_row.label(text="")

        sidebar = list_row.column(align=True)
        sidebar.scale_y = 1.1
        sidebar.operator("qprk.deselect",       text='', icon='RESTRICT_SELECT_ON')
        sidebar.separator()
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

        # ── Filter + sort ──
        row=layout.row(align=True)
        row.prop(props,"filter_text",text="",icon='VIEWZOOM')
        sort_icon={'DEFAULT':'SORTALPHA','AZ':'SORT_ASC','ZA':'SORT_DESC'}[props.sort_mode]
        row.operator("qprk.cycle_sort",text="",icon=sort_icon)

        # ── Active / multi-edit detail ──
        active = get_node(props, props.active_uid)
        checked = checked_entries(props)
        multi   = len(checked) > 1

        if active and active.item_type == 'ENTRY':
            outer_box = layout.box()

            # "Selected Parameter" header — single box (lighter shade)
            hdr_box = outer_box.box()
            hdr_row = hdr_box.row(align=True)
            hdr_row.alignment = 'CENTER'
            icon='BONE_DATA' if active.track_type=='BONE' else 'OBJECT_DATA'
            hdr_row.label(text="Selected Parameter", icon='PROPERTIES')

            # Body of Selected Parameter — in a column(align=True) touching the header
            body_col = outer_box.column(align=True)

            # Entry name sub-label row (subheader band)
            name_box = body_col.box()
            name_row = name_box.row()
            name_row.label(text=active.label or "Entry", icon=icon)

            # Frame range + calculate rows
            params_box = body_col.box()
            row=params_box.row(align=True)
            row.prop(active,"frame_start")
            row.prop(active,"frame_end")
            row.prop(active,"frame_step",text="Step")
            row=params_box.row(align=True)
            op=row.operator("qprk.calculate_entry",icon='PLAY',text="Calculate This")
            op.uid=active.uid
            op2=row.operator("qprk.clear_entry",icon='X',text="Clear This")
            op2.uid=active.uid

            params_box.separator(factor=0.3)

            row=params_box.row(align=True)
            row.prop(active,"line_width",text="Width")
            row.prop(active,"dot_size",text="Dot")

            # ── Line Color (label | color(s) | split-toggle on right) ────
            row = params_box.row(align=True)
            s1 = row.split(factor=0.30, align=True)
            s1.label(text="Line")
            s2 = s1.split(factor=0.85, align=True)
            right_c = s2.row(align=True)
            if active.use_frame_colors:
                half = right_c.split(factor=0.5, align=True)
                half.prop(active, "color_before", text="")
                half.prop(active, "color_after",  text="")
            else:
                right_c.prop(active, "path_color", text="")
            s2.prop(active, "use_frame_colors", text="",
                    icon='UV_SYNC_SELECT' if active.use_frame_colors else 'SNAP_FACE',
                    toggle=True, emboss=True)

            # ── Dot Color (label | color(s) | split-toggle on right) ─────
            row = params_box.row(align=True)
            s1 = row.split(factor=0.30, align=True)
            s1.label(text="Dot")
            s2 = s1.split(factor=0.85, align=True)
            right_c = s2.row(align=True)
            if active.dot_color_mode == 'SPLIT':
                half = right_c.split(factor=0.5, align=True)
                half.prop(active, "dot_color_before", text="")
                half.prop(active, "dot_color_after",  text="")
            else:
                right_c.prop(active, "dot_color", text="")
            op_dc = s2.operator("qprk.toggle_dot_color_mode", text="",
                                icon='UV_SYNC_SELECT' if active.dot_color_mode == 'SPLIT' else 'SNAP_FACE',
                                emboss=True)
            op_dc.uid = active.uid

            if multi:
                params_box.separator(factor=0.3)
                checked_info = params_box.row(align=True)
                checked_info.alignment = 'CENTER'
                checked_info.label(text=f"{len(checked)} items checked", icon='INFO')
                params_box.operator("qprk.apply_to_checked",icon='PASTEDOWN',
                             text=f"Apply Params to All {len(checked)} Checked")
                op_calc=params_box.operator("qprk.calculate_active_range_checked",icon='PLAY',
                             text=f"Calculate All {len(checked)} Checked Paths")
                op_calc.uid=active.uid
                params_box.operator("qprk.clear_checked",icon='X',
                             text=f"Clear All {len(checked)} Checked Paths")

        # ── Global frame range ──
        global_outer = layout.box()
        # Wrapped in box to match the shade of the body content boxes
        global_hdr_box = global_outer.box()
        global_hdr_row = global_hdr_box.row(align=True)
        global_hdr_row.alignment = 'CENTER'
        global_hdr_row.label(text="Global Frame Range", icon='TIME')
        global_body = global_outer.column(align=True)
        global_box = global_body.box()
        row=global_box.row(align=True)
        row.prop(props,"frame_start")
        row.prop(props,"frame_end")
        row=global_box.row(align=True)
        row.prop(props,"frame_step")
        row.operator("qprk.sync_frame_range",text="",icon='SCENE_DATA')
        global_box.operator("qprk.calculate_global",icon='PLAY',
                     text="Calculate All Entries")

        # ── QuickPath Update Mode ──
        update_outer = layout.box()
        # Wrapped in box to match the shade of the body content boxes
        update_hdr_box = update_outer.box()
        update_hdr_row = update_hdr_box.row(align=True)
        update_hdr_row.alignment = 'CENTER'
        update_hdr_row.label(text="QuickPath Update Mode", icon='REC')
        update_body = update_outer.column(align=True)
        update_box = update_body.box()
        row = update_box.row(align=True)
        row.scale_y = 1.2

        mode = props.update_mode
        if mode == 'MANUAL':
            mode_text = "Manual Update"
            mode_icon = 'RADIOBUT_OFF'
            mode_depress = False
        else:  # KEYFRAME
            mode_text = "Keyframe Update"
            mode_icon = 'KEYFRAME_HLT'
            mode_depress = True

        row.operator("qprk.cycle_update_mode",
                     text=mode_text,
                     icon=mode_icon,
                     depress=mode_depress)

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
    QPRK_OT_CycleUpdateMode,
    QPRK_OT_KeyframeWarning,
    QPRK_OT_CalcAnyway,
    QPRK_OT_KeyframeAndCalculate,
    QPRK_OT_SyncFrameRange,
    QPRK_OT_ApplyToChecked,
    QPRK_OT_SlowPathInfo,
    QPRK_OT_ToggleDotColorMode,
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