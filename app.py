"""MV-Kubric dataset app, phase 1: synchronized multi-view frame viewer.

Point it at a recording folder laid out as  <folder>/<view>/rgb/rgb_<frame>.png
(for example ~/docker/isaac-sim/workspace/dataset_test with view01 ... view06,
or the Replicator_XX folders written by the Synthetic Data Recorder). The page
shows the same frame of every view in an m x n grid and steps all views together.

Start with ~/mvkubric_app/run.sh and open http://localhost:8501.

Files: serve.py (entry point, mounts the image routes), previews.py (dataset
index and preview images), groundtruth.py (3D boxes and cameras for the overlay),
frame_decoder.py (PNG decoding in worker processes), viewer.js / viewer.css (the grid, runs in the browser),
preview.js / preview.css (the preview player), common.js (browser code shared by
both), resizer.js (drag handles that resize the viewer and preview panes).
Later phases will add the conversion to MV-Kubric
(see ~/docker/isaac-sim/workspace/convert_to_mvkubric.py).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import streamlit as st

import groundtruth as gtm
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
def viewer(ds: pv.Dataset, rows: int, cols: int) -> None:
    # The grid steps frames in the browser. Once navigation pauses it reports the
    # frame (st.session_state.viewer["frame"]) and, after a rearrangement, the
    # tile order as view names (st.session_state.viewer["layout"]) for later
    # phases; being a fragment, those reports rerun only this part of the page.
    grid = grid_component(read("common.js", "viewer.js"), read("viewer.css"))
    result = grid(key="viewer", data=viewer_data(ds, rows, cols), default={"frame": ds.timeline[0], "layout": None},
                  on_frame_change=lambda: None, on_layout_change=lambda: None, on_preview_change=lambda: None)
    request = getattr(result, "preview", None)   # Preview button: frame + visible tile arrangement
    if request:
        ss = st.session_state
        ss.preview_nonce = ss.get("preview_nonce", 0) + 1
        ss.preview = {"root": ds.root, "frame": int(request["frame"]), "tiles": list(request["tiles"]),
                      "cols": int(request["cols"]), "nonce": ss.preview_nonce}
        st.rerun()   # the preview pane lives outside this fragment


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


def ground_truth_controls(ds: pv.Dataset) -> dict:
    """Top of the preview pane: overlay checkbox and object picker. Returns the overlay settings."""
    row = st.container(horizontal=True, vertical_alignment="center")
    on = row.checkbox("Ground truth values", key="gt_on",
                      help="Draw a point cloud on the ground-truth 3D box of each selected object, "
                           "with each point's trajectory over the last second, in every view.")
    truth = None
    key = f"gt_objects::{ds.root}"
    if on or gtm.is_loaded(ds) or st.session_state.get(key):   # read the boxes once first wanted
        with st.spinner("Reading ground truth..."):
            truth = gtm.get(ds)
    objects = {o.path: (i, o) for i, o in enumerate(truth.objects)} if truth else {}
    chosen = row.multiselect(
        "Objects to track", list(objects), key=key,
        format_func=lambda path: objects[path][1].name, label_visibility="collapsed",
        disabled=not (on and objects),
        placeholder=("Choose objects to track" if objects or not on
                     else "No labelled objects (bounding_box_3d) in this recording"))
    if on and objects and not chosen:
        st.caption("Choose one or more objects to show their tracked points.")
    return {
        "on": bool(on),
        "trail": SOURCE_FPS,     # trajectory length in recorded frames (one second)
        "objects": [{"id": objects[path][0], "name": objects[path][1].label,
                     "color": TRACK_COLORS[objects[path][0] % len(TRACK_COLORS)]}
                    for path in chosen if path in objects],
    }


@st.fragment
def preview_pane(ds: pv.Dataset, request: dict) -> None:
    # The player reports its FPS (st.session_state.player["fps"]) for later phases.
    head = st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute")
    head.markdown("##### Preview")
    if head.button("Close", icon=":material/close:", key="close_preview", type="tertiary"):
        st.session_state.preview = None
        st.rerun()
    overlay = ground_truth_controls(ds)
    player = player_component(read("common.js", "preview.js"), read("viewer.css", "preview.css"))
    player(key="player", data={**preview_data(ds, request), "gt": overlay}, default={"fps": SOURCE_FPS},
           on_fps_change=lambda: None)
    with st.container(horizontal=True, horizontal_alignment="center"):
        st.button("Convert", icon=":material/output:", key="convert",
                  help="Convert the recording to the MV-Kubric format (not active yet).")


def dataset_info(ds: pv.Dataset) -> None:
    w, h = ds.image_size or (0, 0)
    st.markdown("##### Dataset")
    st.caption(ds.root)
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
        "- **Preview:** click Preview under the frame controls. A pane opens that plays the arranged "
        "views as one video, starting at the current frame. Click its play button to play or pause; "
        "double-click it to stop and go back to the start frame. FPS sets how many frames are shown per "
        f"second: playback stays real time and uses evenly spaced frames of the {SOURCE_FPS} recorded "
        "each second (15 shows every 2nd frame). The pane resizes like the viewer.\n"
        "- **Ground truth:** in the preview pane, tick Ground truth values and choose objects. Each "
        "object gets 26 tracked points on its ground-truth 3D box, drawn in every view with the path "
        "of the last second.")
    st.caption("MV-Kubric conversion will be added here in a later phase.")


# --------------------------------------------------------------------------- page

CSS = """
<style>
/* Tighter page, so a 2 x 3 grid plus the controls fits on a 1080p screen. */
[data-testid="stMainBlockContainer"] { padding-top: 3rem; padding-bottom: 1rem; }
/* Keep the grid fully visible during reruns (no fade-out). */
[data-testid="stElementContainer"][data-stale="true"] { opacity: 1 !important; transition: none !important; }

/* Viewer, preview and side panel flow in one wrapping row. The viewer's and the preview's sizes are
   set by dragging their edges (resizer.js keeps them in --mvk-panel-* / --mvk-preview-*); until
   resized, the preview fills the room beside the viewer. Panels that do not fit wrap below. */
.st-key-mvk_layout { align-items: flex-start; }
.st-key-mvk_layout > [data-testid="stLayoutWrapper"]:has(> .st-key-mvk_panel) {
  flex: 0 0 auto; width: var(--mvk-panel-w, 70%); min-width: min(380px, 100%); max-width: 100%;
  height: var(--mvk-panel-h, auto); position: relative;
}
.st-key-mvk_panel { min-height: 0; overflow: auto; }
.st-key-mvk_layout > [data-testid="stLayoutWrapper"]:has(> .st-key-mvk_side) { flex: 1 1 300px; min-width: 280px; }
.st-key-mvk_layout > [data-testid="stLayoutWrapper"]:has(> .st-key-mvk_preview) {
  flex-grow: var(--mvk-preview-grow, 1); flex-shrink: 0; flex-basis: var(--mvk-preview-w, 380px);
  min-width: min(380px, 100%); max-width: 100%; height: var(--mvk-preview-h, auto); position: relative;
}
.st-key-mvk_preview { min-height: 0; overflow: auto; }
.st-key-panel_resizer, .st-key-preview_resizer { display: none; }
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


def main() -> None:
    st.set_page_config(page_title="MV dataset viewer", page_icon=":material/view_module:", layout="wide")
    st.html(CSS)  # style-only HTML goes to the event container and takes no space
    ss = st.session_state
    if "folder" not in ss:
        ss.folder = DEFAULT_FOLDER if os.path.isdir(DEFAULT_FOLDER) else ""

    if os.environ.get(pv.ROUTES_ENV) != "1":
        st.error("The image routes are not running. Start the app with `~/mvkubric_app/run.sh` "
                 "(it runs `streamlit run serve.py`), not `streamlit run app.py`.")
        return

    ds, problem = find_dataset(clean_path(ss.folder))
    request = ss.get("preview")
    if request and (ds is None or request["root"] != ds.root):
        ss.preview = request = None        # another folder was opened: close the preview

    # Resizable viewer unit (folder, grid size, grid), the preview pane when open, and a side
    # panel; they sit side by side when they fit and wrap below otherwise.
    resizer = resizer_component(read("resizer.js"))
    with st.container(horizontal=True, key="mvk_layout"):
        panel = st.container(border=True, key="mvk_panel")
        preview_box = st.container(border=True, key="mvk_preview") if request else None
        side = st.container(key="mvk_side")

    with panel:
        resizer(key="panel_resizer", data={"panel": "mvk_panel", "name": "panel"})
        st.markdown("##### Multi-view dataset viewer")
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
        viewer(ds, rows, cols)

    if preview_box is not None:
        with preview_box:
            resizer(key="preview_resizer", data={"panel": "mvk_preview", "name": "preview"})
            preview_pane(ds, request)

    with side:
        dataset_info(ds)
        controls_help()

main()
