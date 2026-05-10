<p align="center"> 
  <img width="256" height="256" alt="icon" src="https://github.com/user-attachments/assets/c1d0ee22-39cb-4a12-bb74-5212008ddf36" />
</p>

<h1 align="center">
    QuickPathRKNZ
</h1>

A lightweight, real-time motion path visualization addon for bones and objects — made because I got frustrated with Blender's native motion path system while animating and just wanted something that worked better for me.

---

## Why QuickPath RKNZ over Blender's Native Motion Paths?

Blender's built-in motion path tool works, but it has real frustrations in a production workflow:

- **It freezes Blender while calculating.** Native paths require a full scene evaluation at every frame, which means Blender locks up during calculation — especially painful on complex rigs.
- **Paths don't update automatically.** You have to manually click "Update All Paths" every time you move a key. There's no live feedback as you animate.
- **No per-path customization.** Every path looks the same. You can't color-code different bones, control line weight, or distinguish past from future motion at a glance.
- **No organization tools.** All paths are a flat list with no way to group, label, or manage them beyond the basic object panel.

---

## Features

### Real-Time Live Update
Paths automatically recalculate whenever you move a keyframe — no manual refresh needed. Toggle live update on or off at any time from the top of the panel.

### Fast Sampling Without Freezing
QuickPath samples bones and objects by evaluating F-curves and constraints directly, without scrubbing through every frame. For rigs with supported constraints (Copy Location, Copy Transforms, Child Of, Damped Track, and more), calculation is near-instant. Complex rigs fall back to a frame-set method automatically.

### Active Entry Glow
The currently selected entry in the list gets a soft glow effect in the viewport, making it easy to identify which path you're looking at.

### Dot Markers
Optional keyframe dot markers drawn along the path. Size is adjustable per entry, giving you a clear sense of spacing and timing rhythm directly in the viewport.

### Adjustable Line Width
Control the thickness of each path line individually for better visual clarity across complex rigs.

### Collections / Folders
Group related entries into named folders. Collapse folders to keep the list clean. Front and visibility toggles on a folder affect all entries inside it at once.

### Checked Entry Batch Operations
Check multiple entries to batch-apply operations across them:
- **Apply Params** — copy the active entry's color, line width, dot size, and color mode to all checked entries
- **Calculate All Checked Paths** — recalculate all checked entries using the active entry's frame range
- **Clear All Checked Paths** — clear cached path data for all checked entries at once

### Per-Entry Frame Range
Each entry has its own start frame, end frame, and step interval — completely independent of the scene or other entries. Set a tight range around an action for fast recalculation, or a wide range to preview a full sequence.

### Global Frame Range (Calculate All)
The **Calculate All Entries (Global Range)** button in the Global Frame Range box recalculates every entry in the list using a shared frame range — useful for a quick full refresh of all paths at once.

---

## Main Controls at a Glance

| Control | What it does |
|---|---|
| **Show Paths** | Toggle path drawing on/off globally in the viewport |
| **Dots** | Toggle keyframe dot markers on/off globally |
| **Checkbox (per entry)** | Mark an entry as "checked" for batch operations |
| **Color swatch** | Set the path color; expands to Before/After colors if enabled |
| **Front button (cube/xray icon)** | Toggle whether this path draws in front of geometry |
| **Visibility button (eye icon)** | Hide or show this path in the viewport |
| **Add Selected** | Add the currently selected bones or objects as new tracked entries |
| **Add Folder** | Create a new collection to organize entries |
| **Calculate This** | Recalculate this single entry using its own frame range |
| **Clear This** | Remove the cached path data for this entry |
| **Calculate All Entries (Global Range)** | Recalculate every entry using the global frame range settings |

---

*I've only tested this on Blender 5.1, so I honestly can't guarantee it works on other versions. If you run into any errors or bugs, feel free to reach out to me on Discord (rikokenz)*

*Tested & worked on : 4.2*
