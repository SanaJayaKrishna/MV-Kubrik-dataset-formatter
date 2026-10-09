"""MV-Kubric dataset app, phase 1: synchronized multi-view frame viewer.

Point it at a recording folder laid out as  <folder>/<view>/rgb/rgb_<frame>.png
(for example ~/docker/isaac-sim/workspace/dataset_test with view01 ... view06,
or the Replicator_XX folders written by the Synthetic Data Recorder). The page
shows the same frame of every view in an m x n grid and steps all views together.

Start with ~/mvkubric_app/run.sh and open http://localhost:8501.

Files: serve.py (entry point, mounts the image routes), previews.py (dataset
index and preview images), viewer.js / viewer.css (the grid, runs in the browser),
resizer.js (drag handle that resizes the viewer panel).
Later phases will add the conversion to MV-Kubric
(see ~/docker/isaac-sim/workspace/convert_to_mvkubric.py).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import streamlit as st

import previews as pv

DEFAULT_FOLDER = "/home/jk/docker/isaac-sim/workspace/dataset_test"
MAX_GRID = 10   # upper limit for rows and for columns
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- helpers

@st.cache_resource(max_entries=8, show_spinner="Indexing frames...")
def load_dataset(root: str, signature: tuple) -> pv.Dataset:
    return pv.index_dataset(root, signature)


@st.cache_resource
def grid_component(js: str, css: str):
    """Register the browser-side grid once (again only if viewer.js/css change)."""
    return st.components.v2.component("mv_frame_grid", js=js, css=css)


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


def open_dataset(folder: str) -> pv.Dataset | None:
    """Index the folder, or show why it cannot be viewed and return None."""
    if not folder:
        st.info("Paste the full path of a dataset folder above, or click Browse.")
        return None
    root = Path(folder)
    if not root.is_dir():
        st.error(f"Folder not found: {folder}")
        return None
    try:
        ds = load_dataset(str(root), pv.folder_signature(root))
    except OSError as exc:
        st.error(f"Cannot read {folder}: {exc}")
        return None
    if not ds.views:
        st.warning(f"No view folders found in {folder}. Expected a layout like "
                   f"{folder}/view01/rgb/rgb_0000.png (one subfolder per camera).")
        return None
    if not ds.timeline:
        st.warning(f"Found {len(ds.views)} view folders but no rgb_<frame>.png images in their rgb/ folders.")
        return None
    return ds


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

def viewer_data(ds: pv.Dataset, rows: int, cols: int) -> dict:
    """Everything the browser-side grid needs (frame lists are sent as compact runs).

    All views are sent; the grid shows rows x cols tiles and the user arranges
    which view goes on which tile (saved per dataset folder in the browser).
    """
    missing = {}
    for v, frames in enumerate(ds.frames):
        if len(frames) != len(ds.timeline):
            missing[str(v)] = pv.to_runs(f for f in ds.timeline if f not in frames)
    return {
        "ds": ds.id,
        "root": ds.root,
        "base": pv.ROUTE_PREFIX,
        "views": list(ds.views),
        "names": list(ds.name_patterns),
        "runs": pv.to_runs(ds.timeline),
        "missing": missing,
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
    grid = grid_component((HERE / "viewer.js").read_text(), (HERE / "viewer.css").read_text())
    grid(key="viewer", data=viewer_data(ds, rows, cols), default={"frame": ds.timeline[0], "layout": None},
         on_frame_change=lambda: None, on_layout_change=lambda: None)


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
        "- **Arrange views:** press the right mouse button on a view, drag it onto another tile "
        "and release; the two swap places. The arrangement is remembered per dataset folder.\n"
        "- **Resize:** drag the handle on the viewer's right edge; double-click it to reset. "
        "This panel moves beside or below the viewer to fit.")
    st.caption("MV-Kubric conversion will be added here in a later phase.")


# --------------------------------------------------------------------------- page

CSS = """
<style>
/* Tighter page, so a 2 x 3 grid plus the controls fits on a 1080p screen. */
[data-testid="stMainBlockContainer"] { padding-top: 3rem; padding-bottom: 1rem; }
/* Keep the grid fully visible during reruns (no fade-out). */
[data-testid="stElementContainer"][data-stale="true"] { opacity: 1 !important; transition: none !important; }

/* Viewer panel + side panel: the viewer's width is set by dragging its edge (resizer.js
   stores it in --mvk-panel-w); the side panel sits beside it when it fits, else wraps below. */
.st-key-mvk_layout { align-items: flex-start; }
.st-key-mvk_layout > [data-testid="stLayoutWrapper"]:has(> .st-key-mvk_panel) {
  flex: 0 0 auto; width: var(--mvk-panel-w, 70%); min-width: min(380px, 100%); max-width: 100%;
  position: relative;
}
.st-key-mvk_layout > [data-testid="stLayoutWrapper"]:has(> .st-key-mvk_side) { flex: 1 1 300px; min-width: 280px; }
.st-key-panel_resizer { display: none; }
.st-key-mvk_controls .st-key-folder { flex: 1 1 320px !important; min-width: 220px; }
.mvk-resize-handle {
  position: absolute; top: 0; bottom: 0; right: -14px; width: 12px; z-index: 10;
  display: flex; align-items: center; justify-content: center; cursor: col-resize; touch-action: none;
}
.mvk-resize-handle::after {
  content: ""; width: 4px; height: 56px; border-radius: 2px;
  background: rgba(128, 128, 128, 0.45); transition: background 0.15s, height 0.15s;
}
.mvk-resize-handle:hover::after, .mvk-resize-handle.mvk-active::after {
  background: var(--mvk-accent, #ff4b4b); height: 120px;
}
body.mvk-resizing, body.mvk-resizing * { cursor: col-resize !important; user-select: none !important; }
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

    # One resizable unit (folder, grid size, grid) and a side panel that flows around it.
    with st.container(horizontal=True, key="mvk_layout"):
        panel = st.container(border=True, key="mvk_panel")
        side = st.container(key="mvk_side")

    with panel:
        resizer_component((HERE / "resizer.js").read_text())(key="panel_resizer", data={"panel": "mvk_panel"})
        st.markdown("##### Multi-view dataset viewer")
        controls = st.container(horizontal=True, vertical_alignment="bottom", key="mvk_controls")
        controls.text_input("Dataset folder", key="folder", placeholder=DEFAULT_FOLDER,
                            help="Full path of a recording folder whose subfolders (view01, view02, ... "
                                 "or Replicator_XX) each contain an rgb/ folder with rgb_<frame>.png files.")
        if controls.button("Browse...", icon=":material/folder_open:"):
            ss.browse_dir = str(existing_dir(ss.folder))
            browse_dialog()
        grid_size = controls.container(horizontal=True, vertical_alignment="bottom", width="content")

        ds = open_dataset(clean_path(ss.folder))
        if ds is None:
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

    with side:
        dataset_info(ds)
        controls_help()


main()
