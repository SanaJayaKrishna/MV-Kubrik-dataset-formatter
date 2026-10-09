"""Dataset indexing and preview images for the MV-Kubric dataset app.

Shared by the page script (app.py) and the HTTP routes mounted by serve.py.
The page indexes a dataset folder and registers it; the browser then fetches
downscaled JPEG previews from

    /mvk/preview/<dataset id>/<view index>/<frame number>?w=<width>

Only frames of datasets that were opened in the app are served.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import math
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from starlette.requests import Request
from starlette.responses import Response

ROUTE_PREFIX = "/mvk"
ROUTES_ENV = "MVK_PREVIEW_ROUTES"   # set by serve.py when the routes are mounted

# Frame number = last run of digits before ".png", the same convention as
# convert_to_mvkubric.py (rgb_0042.png -> 42).
FRAME_RE = re.compile(r"(\d+)\.png$", re.IGNORECASE)

CACHE_BYTES = 1 << 30   # ~1 GB of preview JPEGs (about 90 KB each at 854 px wide)
JPEG_QUALITY = 85
MAX_WARM = 512          # background decodes queued at most
MAX_DATASETS = 16       # datasets kept registered for the image route


# --------------------------------------------------------------------------- dataset index

def natural_key(name: str) -> list:
    """Sort view01 < view2 < view10 and Replicator < Replicator_01 naturally."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def view_dirs(root: Path) -> list[tuple[str, Path]]:
    """(view name, rgb folder) for every subfolder of root that has an rgb/ folder.

    If root itself has an rgb/ folder it is treated as a single-view dataset.
    """
    if (root / "rgb").is_dir():
        return [(root.name, root / "rgb")]
    found = []
    with os.scandir(root) as entries:
        for e in entries:
            if e.is_dir() and not e.name.startswith(".") and os.path.isdir(os.path.join(e.path, "rgb")):
                found.append((e.name, Path(e.path) / "rgb"))
    return sorted(found, key=lambda v: natural_key(v[0]))


def folder_signature(root: Path) -> tuple:
    """Cheap fingerprint (rgb folder mtimes) so added or deleted frames trigger a rescan."""
    return tuple((name, str(rgb), rgb.stat().st_mtime_ns) for name, rgb in view_dirs(root))


@dataclass(frozen=True, eq=False)
class Dataset:
    id: str                                   # short hash of folder + signature, used in URLs
    root: str
    views: tuple[str, ...]                    # view folder names in natural order
    frames: tuple[dict[int, str], ...]        # per view: frame number -> png path
    timeline: tuple[int, ...]                 # sorted union of the frame numbers of all views
    image_size: tuple[int, int] | None        # (width, height) of the first image
    name_patterns: tuple[tuple[str, int, str] | None, ...]  # per view: (prefix, digits, suffix)


def _name_pattern(per_view: dict[int, str]) -> tuple[str, int, str] | None:
    """('rgb_', 4, '.png') if every file is named prefix + zero-padded frame + suffix."""
    if not per_view:
        return None
    first = os.path.basename(next(iter(per_view.values())))
    m = FRAME_RE.search(first)
    prefix, digits, suffix = first[:m.start(1)], len(m.group(1)), first[m.end(1):]
    for frame, path in per_view.items():
        if os.path.basename(path) != f"{prefix}{frame:0{digits}d}{suffix}":
            return None
    return prefix, digits, suffix


def index_dataset(root: str, signature: tuple) -> Dataset:
    views, frames = [], []
    for name, rgb, _mtime in signature:
        per_view = {}
        with os.scandir(rgb) as entries:
            for e in entries:
                m = FRAME_RE.search(e.name)
                if m and e.is_file():
                    per_view[int(m.group(1))] = e.path
        views.append(name)
        frames.append(per_view)
    timeline = tuple(sorted(set().union(*frames))) if frames else ()
    size = None
    for per_view in frames:
        if per_view:
            with Image.open(next(iter(per_view.values()))) as im:
                size = im.size
            break
    ds_id = hashlib.sha1(f"{root}|{signature}".encode()).hexdigest()[:12]
    return Dataset(ds_id, root, tuple(views), tuple(frames), timeline, size,
                   tuple(_name_pattern(f) for f in frames))


_registry: OrderedDict[str, Dataset] = OrderedDict()
_registry_lock = threading.Lock()


def register(ds: Dataset) -> None:
    """Allow the image route to serve this dataset's frames."""
    with _registry_lock:
        _registry[ds.id] = ds
        _registry.move_to_end(ds.id)
        while len(_registry) > MAX_DATASETS:
            _registry.popitem(last=False)


def registered(ds_id: str) -> Dataset | None:
    with _registry_lock:
        return _registry.get(ds_id)


def to_runs(frames) -> list[list[int]]:
    """Compress sorted frame numbers into inclusive [first, last] runs: 0..587 -> [[0, 587]]."""
    runs: list[list[int]] = []
    for f in frames:
        if runs and f == runs[-1][1] + 1:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    return runs


def preview_width(cols: int, image_size: tuple[int, int] | None) -> int:
    """Preview resolution: roughly the on-screen cell width on a 2560 px wide screen."""
    native = image_size[0] if image_size else 1280
    return max(320, min(native, math.ceil(2560 / cols)))


# --------------------------------------------------------------------------- preview cache

def render_preview(path: str, width: int) -> bytes:
    """Decode a PNG, downscale it to the preview width and encode it as JPEG."""
    with Image.open(path) as im:
        im = im.convert("RGB")  # Isaac Sim writes RGBA with an opaque alpha channel
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()


class PreviewCache:
    """Thread-safe LRU cache of preview JPEGs.

    Decoding one 1280x720 PNG takes about 35 ms. Pillow releases the GIL while
    decoding, so thread pools decode many images in parallel: one pool serves
    images the browser is waiting for, a second one decodes upcoming frames in
    the background ("warm-up") so they are ready before they are requested.
    """

    def __init__(self, max_bytes: int) -> None:
        self._max = max_bytes
        self._items: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()
        workers = min(16, os.cpu_count() or 4)
        self._now = ThreadPoolExecutor(workers, thread_name_prefix="preview")
        self._bg = ThreadPoolExecutor(workers, thread_name_prefix="warmup")
        self._inflight: dict[tuple[str, int], Future] = {}
        self._warming: dict[tuple[str, int], Future] = {}

    def _build(self, key: tuple[str, int]) -> bytes:
        try:
            data = render_preview(*key)
            with self._lock:
                if key not in self._items:
                    self._items[key] = data
                    self._size += len(data)
                    while self._size > self._max and self._items:
                        _, old = self._items.popitem(last=False)
                        self._size -= len(old)
            return data
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def _future(self, key: tuple[str, int], urgent: bool) -> Future:
        """Caller holds the lock."""
        data = self._items.get(key)
        if data is not None:
            self._items.move_to_end(key)
            done: Future = Future()
            done.set_result(data)
            return done
        fut = self._inflight.get(key)
        # An urgent request must not wait behind queued warm-up work: take the job over.
        if fut is not None and urgent and not fut.running() and fut.cancel():
            fut = None
        if fut is None:
            fut = (self._now if urgent else self._bg).submit(self._build, key)
            self._inflight[key] = fut
            if not urgent:
                self._warming[key] = fut
        return fut

    def get(self, key: tuple[str, int]) -> Future:
        """Future with the JPEG bytes (already resolved when cached)."""
        with self._lock:
            return self._future(key, urgent=True)

    def warm(self, keys: list[tuple[str, int]]) -> None:
        """Decode keys in the background, nearest first; drop queued work that is no longer wanted."""
        wanted = set(keys)
        with self._lock:
            for key, fut in list(self._warming.items()):
                if fut.done():
                    del self._warming[key]
                elif key not in wanted and fut.cancel():
                    del self._warming[key]
                    if self._inflight.get(key) is fut:
                        del self._inflight[key]
            for key in keys:
                if len(self._warming) >= MAX_WARM:
                    break
                if key not in self._items and key not in self._inflight:
                    self._future(key, urgent=False)


CACHE = PreviewCache(CACHE_BYTES)


# --------------------------------------------------------------------------- HTTP routes

def _width(request: Request, ds: Dataset) -> int:
    native = ds.image_size[0] if ds.image_size else 1280
    try:
        w = int(request.query_params.get("w", native))
    except ValueError:
        w = native
    return min(max(w, 64), native)


def _int_list(text: str, limit: int) -> list[int]:
    return [int(x) for x in text.split(",")[:limit] if x.strip()]


async def preview_endpoint(request: Request) -> Response:
    ds = registered(request.path_params["ds"])
    view, frame = request.path_params["view"], request.path_params["frame"]
    path = ds.frames[view].get(frame) if ds and 0 <= view < len(ds.frames) else None
    if path is None:
        return Response(status_code=404)
    try:
        data = await asyncio.wrap_future(CACHE.get((path, _width(request, ds))))
    except Exception as exc:  # corrupt or half-written PNG
        return Response(f"cannot read {os.path.basename(path)}: {exc}", status_code=500, media_type="text/plain")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


async def warm_endpoint(request: Request) -> Response:
    ds = registered(request.path_params["ds"])
    if ds is None:
        return Response(status_code=404)
    try:
        views = _int_list(request.query_params.get("v", ""), 64)
        frames = _int_list(request.query_params.get("f", ""), 128)
    except ValueError:
        return Response(status_code=400)
    width = _width(request, ds)
    CACHE.warm([(ds.frames[v][f], width) for f in frames for v in views
                if 0 <= v < len(ds.frames) and f in ds.frames[v]])
    return Response(status_code=204)
