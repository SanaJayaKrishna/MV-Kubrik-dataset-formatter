"""MV-Kubric dataset app, phase 1: synchronized multi-view frame viewer.

Point it at a recording folder laid out as  <folder>/<view>/rgb/rgb_<frame>.png
(for example ~/docker/isaac-sim/workspace/dataset_test with view01 ... view06,
or the Replicator_XX folders written by the Synthetic Data Recorder). The page
shows the same frame of every view in an m x n grid and steps all views together.

Start with ~/mvkubric_app/run.sh and open http://localhost:8501.

Files: serve.py (entry point, mounts the image routes), previews.py (dataset
index and preview images), groundtruth.py (3D boxes and cameras for the overlay),
frame_decoder.py (PNG decoding in worker processes), converter.py + jobs.py (the MV-Kubric
conversion, run in a background process), viewer.js / viewer.css (the grid, runs in the browser),
preview.js / preview.css (the preview player), common.js (browser code shared by
both), resizer.js (drag handles that resize the viewer and preview panes).
Later phases will add the conversion to MV-Kubric
(see ~/docker/isaac-sim/workspace/convert_to_mvkubric.py).
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import streamlit as st

import converter as cv
import groundtruth as gtm
import jobs
import previews as pv

DEFAULT_FOLDER = "/home/jk/docker/isaac-sim/workspace/dataset_test"
MAX_GRID = 10   # upper limit for rows and for columns
SOURCE_FPS = 30  # capture rate of the recordings (Isaac Sim stage timeCodesPerSecond)
# Overlay colours of tracked objects (by object index), bright enough for the warehouse scenes.
TRACK_COLORS = ["#ff3b30", "#34c759", "#0a84ff", "#ffcc00", "#bf5af2", "#ff9f0a",
                "#64d2ff", "#ff375f", "#30d158", "#ac8e68"]
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- helpers

@st.cache_resource(max_entries=8, show_spinner="Indexing frames...")
def load_dataset(root: str, signature: tuple) -> pv.Dataset:
    return pv.index_dataset(root, signature)


def read(*names: str) -> str:
    return "\n".join((HERE / name).read_text() for name in names)


@st.cache_resource
def grid_component(js: str, css: str):
    """Register the browser-side grid once (again only if its JS/CSS change)."""
    return st.components.v2.component("mv_frame_grid", js=js, css=css)


@st.cache_resource
def player_component(js: str, css: str):
    """Register the browser-side preview player once (again only if its JS/CSS change)."""
    return st.components.v2.component("mv_preview_player", js=js, css=css)


@st.cache_resource
def resizer_component(js: str):
    """Drag handle on the viewer panel; it works on the page DOM, so no style isolation."""
    return st.components.v2.component("mv_panel_resizer", js=js, isolate_styles=False)


def auto_grid(n_views: int) -> tuple[int, int]:
    """Default rows x columns: one row up to 3 views, else a near-square grid (6 -> 2 x 3)."""
    if n_views <= 3:
        return 1, max(n_views, 1)
    cols = math.ceil(math.sqrt(n_views))
    return math.ceil(n_views / cols), cols


def clean_path(text: str) -> str:
    """Accept pasted paths with quotes, a file:// prefix, ~ or a trailing slash."""
    text = text.strip().strip("'\"").strip()
    if text.startswith("file://"):
        text = text[len("file://"):]
    text = os.path.expanduser(text)
    return text.rstrip("/") or text[:1]


def find_dataset(folder: str) -> tuple[pv.Dataset | None, tuple[str, str] | None]:
    """Index the folder; on failure return (None, (message kind, message))."""
    if not folder:
        return None, ("info", "Paste the full path of a dataset folder above, or click Browse.")
    root = Path(folder)
    if not root.is_dir():
        return None, ("error", f"Folder not found: {folder}")
    try:
        ds = load_dataset(str(root), pv.folder_signature(root))
    except OSError as exc:
        return None, ("error", f"Cannot read {folder}: {exc}")
    if not ds.views:
        return None, ("warning", f"No view folders found in {folder}. Expected a layout like "
                                 f"{folder}/view01/rgb/rgb_0000.png (one subfolder per camera).")
    if not ds.timeline:
        return None, ("warning", f"Found {len(ds.views)} view folders but no rgb_<frame>.png images "
                                 "in their rgb/ folders.")
    return ds, None


# --------------------------------------------------------------------------- folder browser

def existing_dir(text: str) -> Path:
    p = Path(clean_path(text) or DEFAULT_FOLDER)
    while not p.is_dir() and p != p.parent:
        p = p.parent
    return p if p.is_dir() else Path.home()


def browse_into(parent: str, widget_key: str) -> None:
    choice = st.session_state.get(widget_key)
    if choice:
        st.session_state.browse_dir = str(Path(parent) / choice)


def browse_to(path: str) -> None:
    st.session_state.browse_dir = path


def choose_folder(path: str) -> None:
    st.session_state.folder = path


@st.dialog("Choose dataset folder", width="large")
def browse_dialog() -> None:
    cur = Path(st.session_state.browse_dir)
    st.code(str(cur), language=None)
    try:
        subdirs = sorted((e.name for e in os.scandir(cur) if e.is_dir() and not e.name.startswith(".")),
                         key=pv.natural_key)
        views = pv.view_dirs(cur)
    except OSError as exc:
        st.error(f"Cannot read this folder: {exc}")
        subdirs, views = [], []

    if views:
        names = ", ".join(v for v, _ in views[:8]) + (" ..." if len(views) > 8 else "")
        st.success(f"{len(views)} view folder(s) with rgb/ found: {names}")
    else:
        st.caption("No view folders with an rgb/ subfolder here. Open a subfolder or go up.")

    up, pick = st.columns([1, 4], vertical_alignment="bottom")
    up.button("Up", icon=":material/arrow_upward:", width="stretch", disabled=cur.parent == cur,
              on_click=browse_to, args=(str(cur.parent),))
    pick_key = f"browse_pick::{cur}"
    pick.selectbox("Open subfolder", subdirs, index=None, key=pick_key,
                   placeholder=f"{len(subdirs)} subfolders (type to filter)" if subdirs else "No subfolders",
                   disabled=not subdirs, on_change=browse_into, args=(str(cur), pick_key))

    if st.button("Use this folder", type="primary", icon=":material/check:", disabled=not views,
                 on_click=choose_folder, args=(str(cur),)):
        st.rerun()


# --------------------------------------------------------------------------- viewer

def missing_runs(ds: pv.Dataset) -> dict[str, list[list[int]]]:
    """Per view index: frames of the timeline that this view lacks, as runs."""
    return {str(v): pv.to_runs(f for f in ds.timeline if f not in frames)
            for v, frames in enumerate(ds.frames) if len(frames) != len(ds.timeline)}


def viewer_data(ds: pv.Dataset, rows: int, cols: int) -> dict:
    """Everything the browser-side grid needs (frame lists are sent as compact runs).

    All views are sent; the grid shows rows x cols tiles and the user arranges
    which view goes on which tile (saved per dataset folder in the browser).
    """
    return {
        "ds": ds.id,
        "root": ds.root,
        "base": pv.ROUTE_PREFIX,
        "views": list(ds.views),
        "names": list(ds.name_patterns),
        "runs": pv.to_runs(ds.timeline),
        "missing": missing_runs(ds),
        "cols": cols,
        "cells": rows * cols,
        "w": pv.preview_width(cols, ds.image_size),
        "size": list(ds.image_size or (16, 9)),
    }


@st.fragment
def viewer(ds: pv.Dataset, rows: int, cols: int, preview_on: bool) -> None:
    # The grid steps frames in the browser. Once navigation pauses it reports the
    # frame (st.session_state.viewer["frame"]) and, after a rearrangement, the
    # tile order as view names (st.session_state.viewer["layout"]) for later
    # phases; being a fragment, those reports rerun only this part of the page.
    grid = grid_component(read("common.js", "viewer.js"), read("viewer.css"))
    result = grid(key="viewer", data={**viewer_data(ds, rows, cols), "previewOn": preview_on},
                  default={"frame": ds.timeline[0], "layout": None},
                  on_frame_change=lambda: None, on_layout_change=lambda: None, on_preview_change=lambda: None)
    request = getattr(result, "preview", None)   # Preview toggle switched on: frame + visible tile arrangement
    if request:
        ss = st.session_state
        ss.preview_nonce = ss.get("preview_nonce", 0) + 1
        ss.preview = {"root": ds.root, "frame": int(request["frame"]), "tiles": list(request["tiles"]),
                      "cols": int(request["cols"]), "nonce": ss.preview_nonce}
        st.rerun()   # the pane switches to the preview, which lives outside this fragment


def preview_data(ds: pv.Dataset, request: dict) -> dict:
    """Data for the player: the tiles as the viewer arranged them (trailing empty rows dropped)."""
    by_name = {name: i for i, name in enumerate(ds.views)}
    tiles = [by_name.get(name, -1) if name else -1 for name in request["tiles"]]
    cols = max(1, min(request["cols"], len(tiles)))
    while len(tiles) > cols and all(t < 0 for t in tiles[-cols:]):
        tiles = tiles[:-cols]
    return {
        "ds": ds.id,
        "root": ds.root,
        "base": pv.ROUTE_PREFIX,
        "views": list(ds.views),
        "runs": pv.to_runs(ds.timeline),
        "missing": missing_runs(ds),
        "tiles": tiles,
        "cols": cols,
        "w": pv.preview_width(cols, ds.image_size),
        "size": list(ds.image_size or (16, 9)),
        "start": request["frame"],
        "srcFps": SOURCE_FPS,
        "nonce": request["nonce"],   # a new Preview click restarts the player even for the same frame
    }


def remembered(key: str, default):
    """Give a widget back the value it had when it was last drawn.

    Streamlit forgets a widget's value when a run does not draw it (here: while the preview
    is off), so the last value is also kept under a second key.
    """
    ss = st.session_state
    if key not in ss:
        ss[key] = ss.get(f"_kept::{key}", default)
    return key


def keep(key: str) -> None:
    st.session_state[f"_kept::{key}"] = st.session_state.get(key)


def ground_truth_controls(ds: pv.Dataset) -> dict:
    """Top of the preview: one object picker. Choosing objects turns the ground-truth overlay on,
    an empty picker turns it off. Returns the overlay settings."""
    with st.spinner("Reading ground truth..."):    # once per dataset (cached), lists the objects
        truth = gtm.get(ds)
    objects = {o.path: (i, o) for i, o in enumerate(truth.objects)}
    key = remembered(f"gt_objects::{ds.root}", [])
    row = st.container(horizontal=True, vertical_alignment="center")
    row.markdown("Ground truth values", width="content")
    chosen = row.multiselect(
        "Ground truth values", list(objects), key=key, label_visibility="collapsed",
        format_func=lambda path: objects[path][1].name, disabled=not objects,
        placeholder=("Choose objects to show their tracked points and trajectories" if objects
                     else "No labelled objects (bounding_box_3d) in this recording"))
    keep(key)
    selected = [path for path in chosen if path in objects]
    pose_key = None
    if selected:
        pose = gtm.cached_poses(ds)
        if pose is None:
            with st.spinner("Reading object poses..."):
                pose = gtm.poses(ds)
        pose_key = pose.key
        if pose.error:
            st.caption(f":orange[{pose.error}. Using the recorded 3D boxes instead, so objects that turn "
                       "are tracked wrongly while they turn.]")
        elif pose.source == "boxes":
            st.caption(":orange[No scene file set (Dataset panel): poses come from the recorded 3D boxes, "
                       "which keep the first frame's orientation, so points drift on objects that turn.]")
    return {
        "on": bool(selected),
        "poseKey": pose_key,     # changes when the pose source changes, so the player reloads the points
        "trail": SOURCE_FPS,     # trajectory length in recorded frames (one second)
        "objects": [{"id": objects[path][0], "name": objects[path][1].label,
                     "color": TRACK_COLORS[objects[path][0] % len(TRACK_COLORS)]}
                    for path in selected],
    }


@st.fragment
def preview_body(ds: pv.Dataset, request: dict) -> None:
    """The pane's content while the Preview toggle is on (the viewer is hidden, not removed)."""
    overlay = ground_truth_controls(ds)
    player = player_component(read("common.js", "preview.js"), read("viewer.css", "preview.css"))
    result = player(key="player", data={**preview_data(ds, request), "gt": overlay},
                    default={"fps": SOURCE_FPS, "seconds": None},
                    on_fps_change=lambda: None, on_seconds_change=lambda: None, on_close_change=lambda: None)
    # The player reports FPS and clip length (st.session_state.player["fps"] / ["seconds"], None =
    # until the last frame) for later phases.
    if getattr(result, "close", None):   # Preview toggle switched off: back to the viewer
        st.session_state.preview = None
        st.rerun()
    convert_controls(ds, request, result)


def next_scene_name(ds: pv.Dataset) -> str:
    folder = Path(ds.root) / cv.OUTPUT_DIR
    try:
        taken = [int(p.name) for p in folder.iterdir() if p.is_dir() and p.name.isdigit()]
    except OSError:
        taken = []
    return str(max(taken, default=0) + 1)


def convert_controls(ds: pv.Dataset, request: dict, result) -> None:
    """Convert button: the clip shown in the preview (start frame, FPS, Seconds) becomes one scene."""
    ss = st.session_state
    fps = int(getattr(result, "fps", None) or SOURCE_FPS)
    seconds = getattr(result, "seconds", None)
    frames = cv.clip_frames(ds.timeline, request["frame"], fps, seconds)
    busy = jobs.is_running(ss.get("convert_job"))
    target = f"{ds.root}/{cv.OUTPUT_DIR}/{next_scene_name(ds)}"
    with st.container(horizontal=True, horizontal_alignment="center"):
        clicked = st.button("Converting..." if busy else "Convert", icon=":material/output:", key="convert",
                            disabled=busy or len(frames) < 2,
                            help="Convert the clip shown above into one MV-Kubric scene. The conversion runs in "
                                 "the background; a pane shows its progress.")
    st.caption(f"Converts {len(frames)} frames ({frames[0]} to {frames[-1]}) at {fps} fps from {len(ds.views)} "
               f"views into {target}" if frames else "The clip is empty.")
    if clicked and not busy:
        clip = {"first": frames[0], "last": frames[-1], "frames": len(frames), "fps": fps,
                "views": len(ds.views), "target": target}
        ss.convert_job = jobs.start(ds.root, request["frame"], fps, seconds, clean_path(ss[scene_key(ds)]), clip)
        st.rerun()                     # the conversion pane lives outside this fragment


# --------------------------------------------------------------------------- conversion pane

STATUS_ICONS = {
    "done": ":green[:material/check_circle:]",
    "running": ":blue[:material/progress_activity:]",
    "pending": ":gray[:material/radio_button_unchecked:]",
    "failed": ":red[:material/error:]",
    "stopped": ":orange[:material/stop_circle:]",
}


def overall_progress(state: dict) -> float:
    total = sum(step["weight"] for step in state.get("steps", [])) or 1
    done = 0.0
    for step in state.get("steps", []):
        if step["status"] == "done":
            done += step["weight"]
        elif step["status"] == "running" and step["total"]:
            done += step["weight"] * step["done"] / step["total"]
    return min(1.0, done / total)


def step_line(step: dict) -> str:
    count = f" ({step['done']}/{step['total']})" if step["status"] == "running" and step["total"] else ""
    detail = f"  \n:gray[{step['detail']}]" if step.get("detail") else ""
    return f"{STATUS_ICONS.get(step['status'], '')} {step['label']}{count}{detail}"


def conversion_pane(job: dict) -> None:
    running = jobs.is_running(job)
    body = st.fragment(run_every=0.5 if running else None)(conversion_body)
    body(job, running)


def conversion_body(job: dict, was_running: bool) -> None:
    ss = st.session_state
    state = jobs.read(job)
    running = state.get("status") in ("starting", "running")
    if was_running and not running:
        st.rerun()                     # finished: stop polling and enable Convert again
    head = st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute")
    head.markdown("##### MV-Kubric conversion")
    if running:
        if head.button("Stop", icon=":material/stop:", key="convert_stop", type="tertiary"):
            jobs.stop(job)
            st.rerun()
    elif head.button("Close", icon=":material/close:", key="convert_close", type="tertiary"):
        ss.convert_job = None
        st.rerun()
    clip = job["clip"]
    st.caption(f"Frames {clip['first']} to {clip['last']}: {clip['frames']} frames at {clip['fps']} fps from "
               f"{clip['views']} views  \nOutput: {state.get('output') or clip['target']}")
    elapsed = (state.get("finished") or time.time()) - state.get("started", job["started"])
    fraction = 1.0 if state.get("status") == "done" else overall_progress(state)
    label = {"starting": "Starting", "running": "Converting", "done": "Done", "failed": "Failed",
             "stopped": "Stopped"}.get(state.get("status"), "")
    st.progress(fraction, text=f"{label}: {fraction * 100:.0f} %  ·  {int(elapsed // 60)}:{int(elapsed % 60):02d}")

    steps = state.get("steps", [])
    completed = [s for s in steps if s["status"] == "done"]
    pending = [s for s in steps if s["status"] != "done"]
    st.markdown(f"**Completed** ({len(completed)} of {len(steps)})")
    st.markdown("\n".join(f"- {step_line(s)}" for s in completed) if completed else ":gray[Nothing yet]")
    st.markdown(f"**Pending** ({len(pending)})")
    st.markdown("\n".join(f"- {step_line(s)}" for s in pending) if pending else ":gray[Nothing left]")
    for note in state.get("notes", []):
        st.caption(f":orange[{note}]")
    if state.get("not_written"):
        st.caption("Not written, because MVTracker makes them itself: " + "; ".join(
            f"{item['name']} ({item['reason']})" for item in state["not_written"]) + ".")
    if state.get("status") == "done" and state.get("summary"):
        summary = state["summary"]
        st.success(f"Scene {summary['scene']} saved to {state['output']}: {summary['frames']} frames, "
                   f"{summary['views']} views, {summary['tracks']} tracked points on the background and "
                   f"{len(summary['objects'])} objects. {summary['check']}.")
    elif state.get("status") in ("failed", "stopped"):
        st.error(state.get("error") or "The conversion did not finish.")
        if state.get("traceback"):
            with st.expander("Details"):
                st.code(state["traceback"], language=None)




def scene_key(ds: pv.Dataset) -> str:
    return f"scene::{ds.root}"


def remember_scene(root: str, key: str) -> None:
    gtm.remember_scene(root, clean_path(st.session_state.get(key, "")))


def dataset_info(ds: pv.Dataset) -> None:
    w, h = ds.image_size or (0, 0)
    st.markdown("##### Dataset")
    st.caption(ds.root)
    st.text_input("Scene file (USD)", key=scene_key(ds), on_change=remember_scene, args=(ds.root, scene_key(ds)),
                  placeholder="the .usd file this recording was made from",
                  help="Gives the exact object poses for the ground-truth points. The recorded 3D boxes "
                       "keep each object's first-frame orientation, so objects that turn need this file.")
    pose = gtm.cached_poses(ds)
    if pose is not None and pose.source == "scene":
        st.caption("Exact object poses come from this file.")
    st.markdown(f"**{len(ds.views)}** view{'s' if len(ds.views) != 1 else ''}  ·  "
                f"**{len(ds.timeline)}** frames ({ds.timeline[0]} to {ds.timeline[-1]})  ·  {w}×{h} px")
    if len({len(f) for f in ds.frames}) > 1:
        st.caption(":orange[Views have different frame counts; missing frames show as empty tiles.]")
    st.dataframe(
        [{"View": name, "Frames": len(frames), "Missing": len(ds.timeline) - len(frames)}
         for name, frames in zip(ds.views, ds.frames)],
        hide_index=True, width="stretch")


def controls_help() -> None:
    st.markdown("##### Controls")
    st.markdown(
        "- **Step:** click ‹ or › or press the Left / Right arrow key. Hold it to keep stepping "
        "(about 30 frames per second); add Shift to step 10 frames.\n"
        "- **Jump:** type a frame number in the box and press Enter.\n"
        "- **Arrange views:** press the left mouse button on a view, drag it onto another tile "
        "and release; the two swap places. The arrangement is remembered per dataset folder.\n"
        "- **Resize the viewer:** drag its right edge for the width, its bottom edge for the height, "
        "or the corner for both; the grid scales to fit. Double-click a handle to reset. "
        "This panel moves beside or below the viewer to fit.\n"
        "- **Preview:** switch on the Preview toggle under the frame controls. The pane turns into a "
        "player for the arranged views as one video, starting at the current frame; switch the toggle "
        "off to get the viewer back as it was. Click the play button to play or pause; double-click "
        "it to stop and go back to the start frame. FPS sets how many frames are shown per second: "
        f"playback stays real time and uses evenly spaced frames of the {SOURCE_FPS} recorded each "
        "second (15 shows every 2nd frame). Seconds sets the clip length: X seconds at n FPS show "
        "n × X frames, or stop at the last recorded frame if the recording is shorter; leave it empty to "
        "play to the end. The total number of frames is shown under the player.\n"
        "- **Convert:** in the preview, Convert turns the clip shown (start frame, FPS, Seconds) into one "
        f"MV-Kubric scene in the dataset folder's {cv.OUTPUT_DIR}/ folder (1, 2, 3, ...). A pane shows the "
        "progress; it runs in the background, so the app stays usable.\n"
        "- **Ground truth:** in the preview, choose objects in Ground truth values (clear it to "
        "hide the overlay). Each object gets 48 tracked points on its real surface, taken from the "
        "recorded depth, moved with its exact pose from the Scene file above and drawn in every view "
        "with the path of the last second. A solid line in the same colour shows the route the object "
        "has travelled since the start frame (the centre of its box at floor level).")
    st.caption("The Convert button in the preview will run the MV-Kubric conversion in a later phase.")


# --------------------------------------------------------------------------- page

CSS = """
<style>
/* Tighter page, so a 2 x 3 grid plus the controls fits on a 1080p screen. */
[data-testid="stMainBlockContainer"] { padding-top: 3rem; padding-bottom: 1rem; }
/* Keep the grid fully visible during reruns (no fade-out). */
[data-testid="stElementContainer"][data-stale="true"] { opacity: 1 !important; transition: none !important; }

/* The viewer pane (which also shows the preview) and the side panel flow in one wrapping row. The
   pane's size is set by dragging its edges (resizer.js keeps it in --mvk-panel-w / -h); the side
   panel sits beside it when it fits, else below. */
.st-key-mvk_layout { align-items: flex-start; }
.st-key-mvk_layout > [data-testid="stLayoutWrapper"]:has(> .st-key-mvk_panel) {
  flex: 0 0 auto; width: var(--mvk-panel-w, 70%); min-width: min(380px, 100%); max-width: 100%;
  height: var(--mvk-panel-h, auto); position: relative;
}
.st-key-mvk_panel { min-height: 0; overflow: auto; }
.st-key-mvk_layout > [data-testid="stLayoutWrapper"]:has(> .st-key-mvk_side) { flex: 1 1 300px; min-width: 280px; }
.st-key-mvk_layout > [data-testid="stLayoutWrapper"]:has(> .st-key-mvk_convert) {
  flex-grow: var(--mvk-convert-grow, 1); flex-shrink: 0; flex-basis: var(--mvk-convert-w, 380px);
  min-width: min(380px, 100%); max-width: 100%; height: var(--mvk-convert-h, auto); position: relative;
}
.st-key-mvk_convert { min-height: 0; overflow: auto; }
.st-key-panel_resizer, .st-key-convert_resizer { display: none; }
.st-key-mvk_controls .st-key-folder { flex: 1 1 320px !important; min-width: 220px; }
.mvk-resize { position: absolute; z-index: 10; display: flex; align-items: center; justify-content: center; touch-action: none; }
.mvk-resize-x { top: 0; bottom: 0; right: -14px; width: 12px; cursor: col-resize; }
.mvk-resize-y { left: 0; right: 0; bottom: -14px; height: 12px; cursor: row-resize; }
.mvk-resize-xy { right: -18px; bottom: -18px; width: 18px; height: 18px; cursor: nwse-resize; }
.mvk-resize::after { content: ""; border-radius: 2px; background: rgba(128, 128, 128, 0.45);
  transition: background 0.15s, width 0.15s, height 0.15s, border-color 0.15s; }
.mvk-resize-x::after { width: 4px; height: 56px; }
.mvk-resize-y::after { width: 56px; height: 4px; }
.mvk-resize-xy::after { width: 9px; height: 9px; background: none; border-radius: 0 0 3px 0;
  border-right: 3px solid rgba(128, 128, 128, 0.6); border-bottom: 3px solid rgba(128, 128, 128, 0.6); }
.mvk-resize-x:hover::after, .mvk-resize-x.mvk-active::after { background: var(--mvk-accent, #ff4b4b); height: 120px; }
.mvk-resize-y:hover::after, .mvk-resize-y.mvk-active::after { background: var(--mvk-accent, #ff4b4b); width: 120px; }
.mvk-resize-xy:hover::after, .mvk-resize-xy.mvk-active::after { border-color: var(--mvk-accent, #ff4b4b); }
body.mvk-resizing, body.mvk-resizing * { user-select: none !important; }
body.mvk-resize-x-active * { cursor: col-resize !important; }
body.mvk-resize-y-active * { cursor: row-resize !important; }
body.mvk-resize-xy-active * { cursor: nwse-resize !important; }
</style>
"""

# While the Preview toggle is on, the viewer's own elements are hidden. They stay in the page so
# that the folder, grid size, frame and arrangement are all still there when it is switched off.
HIDE_VIEWER = "<style>.st-key-mvk_viewer_body { display: none !important; }</style>"


def main() -> None:
    st.set_page_config(page_title="MV dataset viewer", page_icon=":material/view_module:", layout="wide")
    ss = st.session_state
    if "folder" not in ss:
        ss.folder = DEFAULT_FOLDER if os.path.isdir(DEFAULT_FOLDER) else ""

    ds, problem = find_dataset(clean_path(ss.folder))
    request = ss.get("preview")
    if request and (ds is None or request["root"] != ds.root):
        ss.preview = request = None        # another folder was opened: back to the viewer
    preview_on = request is not None
    st.html(CSS + (HIDE_VIEWER if preview_on else ""))  # style-only HTML takes no space on the page

    if os.environ.get(pv.ROUTES_ENV) != "1":
        st.error("The image routes are not running. Start the app with `~/mvkubric_app/run.sh` "
                 "(it runs `streamlit run serve.py`), not `streamlit run app.py`.")
        return

    # One resizable pane (viewer, or preview while the toggle is on) and a side panel that sits
    # beside it when it fits and wraps below otherwise.
    with st.container(horizontal=True, key="mvk_layout"):
        panel = st.container(border=True, key="mvk_panel")
        convert_box = st.container(border=True, key="mvk_convert") if ss.get("convert_job") else None
        side = st.container(key="mvk_side")

    with panel:
        resizer_component(read("resizer.js"))(key="panel_resizer", data={"panel": "mvk_panel", "name": "panel"})
        st.markdown("##### Preview" if preview_on else "##### Multi-view dataset viewer")
        with st.container(key="mvk_viewer_body"):
            controls = st.container(horizontal=True, vertical_alignment="bottom", key="mvk_controls")
            controls.text_input("Dataset folder", key="folder", placeholder=DEFAULT_FOLDER,
                                help="Full path of a recording folder whose subfolders (view01, view02, ... "
                                     "or Replicator_XX) each contain an rgb/ folder with rgb_<frame>.png files.")
            if controls.button("Browse...", icon=":material/folder_open:"):
                ss.browse_dir = str(existing_dir(ss.folder))
                browse_dialog()
            grid_size = controls.container(horizontal=True, vertical_alignment="bottom", width="content")

            if ds is None:
                getattr(st, problem[0])(problem[1])
                # The grid-size widgets were not drawn, so Streamlit forgets their values;
                # make the next valid folder start again from the default grid.
                ss.pop("loaded_root", None)
                with side:
                    controls_help()
                return
            pv.register(ds)
            if scene_key(ds) not in ss:            # scene file for exact object poses (side panel)
                ss[scene_key(ds)] = gtm.remembered_scene(ds.root)
            gtm.use_scene(ds, clean_path(ss[scene_key(ds)]))

            # A new dataset folder resets the grid shape to fit its number of views.
            if ss.get("loaded_root") != ds.root:
                ss.loaded_root = ds.root
                ss.rows, ss.cols = auto_grid(len(ds.views))

            rows = int(grid_size.number_input("Rows (m)", key="rows", min_value=1, max_value=MAX_GRID, step=1,
                                             width=130))
            cols = int(grid_size.number_input("Columns (n)", key="cols", min_value=1, max_value=MAX_GRID, step=1,
                                             width=130))
            if rows * cols < len(ds.views):
                st.caption(f":orange[The grid has {rows * cols} tiles for {len(ds.views)} views. "
                           "Enlarge it to bring the hidden views in and arrange them.]")
            viewer(ds, rows, cols, preview_on)

        if preview_on:
            preview_body(ds, request)

    if convert_box is not None:
        with convert_box:
            resizer_component(read("resizer.js"))(key="convert_resizer", data={"panel": "mvk_convert", "name": "convert"})
            conversion_pane(ss.convert_job)

    with side:
        dataset_info(ds)
        controls_help()

main()
