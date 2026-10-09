"""Ground truth for the preview overlay: 3D boxes of the labelled objects and the cameras.

Reads what the Synthetic Data Recorder (BasicWriter) writes next to rgb/ in every view folder:
    bounding_box_3d/bounding_box_3d_<frame>.npy  (+ bounding_box_3d_labels_ / _prim_paths_ .json)
    camera_params/camera_params_<frame>.json
Boxes are merged across views the way convert_to_mvkubric.py does it: a camera only reports the
objects it sees, so per frame the first view that reports an object supplies its box (in world
coordinates, so all views agree). The browser fetches the data from two routes (see serve.py):
    /mvk/gt/<dataset id>/cameras              per view: projection matrices
    /mvk/gt/<dataset id>/object/<index>       one object's box corners per frame
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

import previews as pv

BOX_DIR, CAMERA_DIR = "bounding_box_3d", "camera_params"
BOX_RE = re.compile(r"^bounding_box_3d_(\d+)\.npy$")
MAX_CACHED = 4
HEADERS = {"Cache-Control": "private, max-age=3600"}   # dataset ids change when files change

# Corner order of convert_to_mvkubric.box_corners_world: z outer, then y, then x (min before max).
CORNER_PICK = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], dtype=bool)


@dataclass(frozen=True, eq=False)
class TrackedObject:
    path: str              # prim path, e.g. /World/StagingHall/AMRs/ForkliftAMR_1
    label: str             # semantic class, e.g. forklift1
    frames: np.ndarray     # (n,) frame numbers in which some view reported the object
    corners: np.ndarray    # (n, 8, 3) float32 world-space box corners in those frames

    @property
    def name(self) -> str:
        return f"{self.label}  ·  {self.path.rsplit('/', 1)[-1]}"


@dataclass(frozen=True, eq=False)
class GroundTruth:
    objects: tuple[TrackedObject, ...]   # sorted by label, then prim path
    cameras: tuple[list, ...]            # per view: [[first frame, world->clip matrix (16 floats)], ...]


def label_of(entry) -> str:
    """BasicWriter writes {"<semanticId>": {"class": "forklift"}}; fall back to any taxonomy present."""
    if isinstance(entry, dict):
        return str(entry.get("class", ",".join(str(v) for v in entry.values())))
    return str(entry)


def box_corners(rows: np.ndarray) -> np.ndarray:
    """World-space corners (n, 8, 3) of BasicWriter bounding_box_3d rows (USD row-vector transforms)."""
    mins = np.stack([rows["x_min"], rows["y_min"], rows["z_min"]], axis=-1).astype(np.float64)
    maxs = np.stack([rows["x_max"], rows["y_max"], rows["z_max"]], axis=-1).astype(np.float64)
    local = np.where(CORNER_PICK[None], maxs[:, None, :], mins[:, None, :])
    local = np.concatenate([local, np.ones(local.shape[:2] + (1,))], axis=-1)
    world = np.einsum("nci,nij->ncj", local, rows["transform"].astype(np.float64))
    return world[..., :3].astype(np.float32)


def _read_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def load(ds: pv.Dataset) -> GroundTruth:
    wanted = set(ds.timeline)
    found: dict[str, tuple[str, dict[int, np.ndarray]]] = {}
    for view_dir in ds.view_paths:
        folder = os.path.join(view_dir, BOX_DIR)
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            m = BOX_RE.match(name)
            if not m or int(m.group(1)) not in wanted:
                continue
            frame, stem = int(m.group(1)), m.group(1)
            try:
                rows = np.load(os.path.join(folder, name))
            except (OSError, ValueError):
                continue
            if len(rows) == 0:
                continue
            labels = _read_json(os.path.join(folder, f"bounding_box_3d_labels_{stem}.json"), {})
            paths = _read_json(os.path.join(folder, f"bounding_box_3d_prim_paths_{stem}.json"), [])
            corners = box_corners(rows)
            for i, row in enumerate(rows):
                sid = str(int(row["semanticId"]))
                path = paths[i] if i < len(paths) else f"object_{sid}"
                _label, per_frame = found.setdefault(path, (label_of(labels.get(sid, sid)), {}))
                per_frame.setdefault(frame, corners[i])
    objects = []
    for path in sorted(found, key=lambda p: (found[p][0].lower(), p)):
        label, per_frame = found[path]
        frames = sorted(per_frame)
        objects.append(TrackedObject(path, label, np.array(frames, dtype=np.int64),
                                     np.stack([per_frame[f] for f in frames])))
    return GroundTruth(tuple(objects), tuple(_cameras(view_dir, ds.timeline) for view_dir in ds.view_paths))


def _cameras(view_dir: str, timeline) -> list:
    """World->clip matrices (row-vector convention: clip = [x y z 1] @ M) where they change."""
    out: list = []
    for frame in timeline:
        raw = _read_json(os.path.join(view_dir, CAMERA_DIR, f"camera_params_{frame:04d}.json"), None)
        if raw is None:
            continue
        view = np.array(raw["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
        proj = np.array(raw["cameraProjection"], dtype=np.float64).reshape(4, 4)
        matrix = (view @ proj).ravel().tolist()
        if not out or out[-1][1] != matrix:
            out.append([frame, matrix])
    return out


_cache: OrderedDict[str, GroundTruth] = OrderedDict()
_lock = threading.Lock()


def get(ds: pv.Dataset) -> GroundTruth:
    with _lock:
        if ds.id in _cache:
            _cache.move_to_end(ds.id)
            return _cache[ds.id]
    truth = load(ds)   # two concurrent first requests just read the files twice
    with _lock:
        _cache[ds.id] = truth
        while len(_cache) > MAX_CACHED:
            _cache.popitem(last=False)
    return truth


def is_loaded(ds: pv.Dataset) -> bool:
    with _lock:
        return ds.id in _cache


# --------------------------------------------------------------------------- HTTP routes

async def cameras_endpoint(request: Request) -> Response:
    ds = pv.registered(request.path_params["ds"])
    if ds is None:
        return Response(status_code=404)
    truth = await asyncio.to_thread(get, ds)
    return JSONResponse({"views": list(truth.cameras)}, headers=HEADERS)


async def object_endpoint(request: Request) -> Response:
    ds = pv.registered(request.path_params["ds"])
    if ds is None:
        return Response(status_code=404)
    truth = await asyncio.to_thread(get, ds)
    index = request.path_params["index"]
    if not 0 <= index < len(truth.objects):
        return Response(status_code=404)
    obj = truth.objects[index]
    return JSONResponse({
        "path": obj.path,
        "label": obj.label,
        "frames": obj.frames.tolist(),
        "corners": np.round(obj.corners.reshape(len(obj.frames), 24).astype(np.float64), 4).tolist(),
    }, headers=HEADERS)
