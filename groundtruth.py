"""Ground truth for the preview overlay: 3D boxes of the labelled objects and the cameras.

Reads what the Synthetic Data Recorder (BasicWriter) writes next to rgb/ in every view folder:
    bounding_box_3d/bounding_box_3d_<frame>.npy  (+ bounding_box_3d_labels_ / _prim_paths_ .json)
    camera_params/camera_params_<frame>.json
Boxes are merged across views the way convert_to_mvkubric.py does it: a camera only reports the
objects it sees, so per frame the first view that reports an object supplies its box (in world
coordinates, so all views agree).

Tracked points (see surface_points): the 3D boxes only bound an object, and most of a forklift's
box is empty space, so the points are taken from the object's real surface instead: the recorded
depth (distance_to_camera) of every view is turned into 3D points, and the points inside the
object's box and above the floor are kept, in the object's own coordinates. They then move with
the object's pose. Isaac Sim 6.1 records the boxes of keyframe-animated objects with the first
frame's orientation (the 'transform' never changes, only the box extents move), so the exact
poses are read from the scene USD the recording was made from (usd_poses.py, run with Isaac
Sim's Python). Without a scene file the poses come from the boxes: right for objects that only
move, wrong while an object turns.

The browser fetches the data from two routes (see serve.py):
    /mvk/gt/<dataset id>/cameras              per view: projection matrices
    /mvk/gt/<dataset id>/object/<index>       one object's surface points and per-frame poses
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

import previews as pv

BOX_DIR, CAMERA_DIR = "bounding_box_3d", "camera_params"
DEPTH_DIRS = ("distance_to_camera", "distance_to_image_plane")   # either depth annotator works
BOX_RE = re.compile(r"^bounding_box_3d_(\d+)\.npy$")
MAX_CACHED = 4
POINTS_PER_OBJECT = 48     # tracked points per object
SAMPLE_FRAMES = 6          # frames whose depth is used to find each object's surface
VOXEL_M = 0.04             # merge depth points closer than this before choosing the tracked points
BOX_MARGIN_M = 0.03        # depth points this far outside an object's box still count
FLOOR_M = 0.03             # ...but not points this close to the bottom of the box (the floor)
POSE_TOOL = Path(__file__).with_name("usd_poses.py")
CACHE_DIR = Path.home() / ".cache" / "mvkubric_app"
SCENES_FILE = CACHE_DIR / "scenes.json"   # dataset folder -> scene file, remembered across restarts
HEADERS = {"Cache-Control": "private, max-age=3600"}   # dataset ids change when files change

# Corner order of convert_to_mvkubric.box_corners_world: z outer, then y, then x (min before max).
CORNER_PICK = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], dtype=bool)


@dataclass(frozen=True, eq=False)
class TrackedObject:
    path: str              # prim path, e.g. /World/StagingHall/AMRs/ForkliftAMR_1
    label: str             # semantic class, e.g. forklift1
    frames: np.ndarray     # (n,) frame numbers in which some view reported the object
    corners: np.ndarray    # (n, 8, 3) float32 world-space box corners in those frames
    box_min: np.ndarray    # (3,) box in the object's own coordinates, from its first frame (where the
    box_max: np.ndarray    #      recorded transform is the true pose)
    transform0: np.ndarray # (4, 4) recorded transform (= pose in the first frame; never changes)

    @property
    def name(self) -> str:
        return f"{self.label}  ·  {self.path.rsplit('/', 1)[-1]}"


@dataclass(frozen=True, eq=False)
class GroundTruth:
    objects: tuple[TrackedObject, ...]   # sorted by label, then prim path
    cameras: tuple[list, ...]            # per view: [[first frame, world->clip matrix (16 floats)], ...]
    camera_matrices: tuple[list, ...]    # per view: [(first frame, view (4x4), projection (4x4), width, height)]


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
                per_frame.setdefault(frame, (corners[i], row))
    objects = []
    for path in sorted(found, key=lambda p: (found[p][0].lower(), p)):
        label, per_frame = found[path]
        frames = sorted(per_frame)
        first = per_frame[frames[0]][1]
        objects.append(TrackedObject(
            path, label, np.array(frames, dtype=np.int64), np.stack([per_frame[f][0] for f in frames]),
            np.array([first["x_min"], first["y_min"], first["z_min"]], dtype=np.float64),
            np.array([first["x_max"], first["y_max"], first["z_max"]], dtype=np.float64),
            first["transform"].astype(np.float64)))
    matrices = tuple(_cameras(view_dir, ds.timeline) for view_dir in ds.view_paths)
    clip = tuple([[frame, (view @ proj).ravel().tolist()] for frame, view, proj, _w, _h in cams] for cams in matrices)
    return GroundTruth(tuple(objects), clip, matrices)


def _cameras(view_dir: str, timeline) -> list:
    """Per view: (first frame, view, projection, width, height) wherever the camera changes.
    Row-vector convention: clip = [x y z 1] @ view @ projection."""
    out: list = []
    for frame in timeline:
        raw = _read_json(os.path.join(view_dir, CAMERA_DIR, f"camera_params_{frame:04d}.json"), None)
        if raw is None:
            continue
        view = np.array(raw["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
        proj = np.array(raw["cameraProjection"], dtype=np.float64).reshape(4, 4)
        width, height = raw["renderProductResolution"]
        if not out or not (np.array_equal(out[-1][1], view) and np.array_equal(out[-1][2], proj)):
            out.append((frame, view, proj, int(width), int(height)))
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


# --------------------------------------------------------------------------- scene file setting

_scene_lock = threading.Lock()
_use_scene: dict[str, str] = {}      # dataset id -> scene file chosen in the app


def remembered_scene(root: str) -> str:
    try:
        return json.loads(SCENES_FILE.read_text()).get(root, "")
    except (OSError, ValueError):
        return ""


def remember_scene(root: str, scene: str) -> None:
    with _scene_lock:
        try:
            data = json.loads(SCENES_FILE.read_text())
        except (OSError, ValueError):
            data = {}
        if scene:
            data[root] = scene
        else:
            data.pop(root, None)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        SCENES_FILE.write_text(json.dumps(data, indent=2) + "\n")


def use_scene(ds: pv.Dataset, scene: str) -> None:
    """Scene file the HTTP routes should take poses from for this dataset ('' = none)."""
    _use_scene[ds.id] = scene


# --------------------------------------------------------------------------- object poses

@dataclass(frozen=True, eq=False)
class Poses:
    key: str                             # changes whenever the source changes (part of the URLs)
    source: str                          # "scene" or "boxes"
    transforms: dict[str, np.ndarray]    # prim path -> (timeline frames, 4, 4) object->world, NaN if unknown
    error: str | None = None             # why the scene file could not be used


_poses: OrderedDict[tuple, Poses] = OrderedDict()


def box_poses(ds: pv.Dataset, truth: GroundTruth) -> dict[str, np.ndarray]:
    """Poses from the recorded boxes: first-frame orientation, moved with the box centre."""
    index = {f: i for i, f in enumerate(ds.timeline)}
    out = {}
    for obj in truth.objects:
        transforms = np.full((len(ds.timeline), 4, 4), np.nan)
        centres = obj.corners.astype(np.float64).mean(axis=1)
        for j, frame in enumerate(obj.frames):
            m = obj.transform0.copy()
            m[3, :3] += centres[j] - centres[0]          # row-vector convention: translation in the last row
            transforms[index[int(frame)]] = m
        out[obj.path] = transforms
    return out


def _scene_poses(ds: pv.Dataset, truth: GroundTruth, scene: str) -> dict[str, np.ndarray]:
    """Exact poses read from the scene USD at every recorded frame (frame i = time code i)."""
    first, last = ds.timeline[0], ds.timeline[-1]
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "poses.npz")
        cmd = [sys.executable, str(POSE_TOOL), scene, "--frames", str(first), str(last), "--out", out]
        for obj in truth.objects:
            cmd += ["--prim", obj.path]
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if run.returncode != 0 or not os.path.exists(out):
            lines = [line for line in (run.stderr + run.stdout).splitlines()
                     if line.strip() and not line.startswith("Warning")]
            raise RuntimeError(lines[-1] if lines else f"the pose export stopped with exit code {run.returncode}")
        data = np.load(out)
        rows = {int(f): i for i, f in enumerate(data["frame_ids"])}
        transforms, ids = data["transforms"], [str(x) for x in data["object_ids"]]
    result = {}
    for k, path in enumerate(ids):
        per_frame = np.full((len(ds.timeline), 4, 4), np.nan)
        for i, frame in enumerate(ds.timeline):
            if frame in rows:
                per_frame[i] = transforms[rows[frame], k]
        result[path] = per_frame
    return result


def cached_poses(ds: pv.Dataset) -> Poses | None:
    """The poses for the dataset's current scene setting if they were already read, else None."""
    scene = _use_scene.get(ds.id, "")
    try:
        stamp = os.stat(scene).st_mtime_ns if scene else None
    except OSError:
        stamp = "missing"
    with _lock:
        return _poses.get((ds.id, scene, stamp))


def poses(ds: pv.Dataset, scene: str | None = None) -> Poses:
    """Object poses: from the scene file when one is set and readable, otherwise from the boxes."""
    scene = _use_scene.get(ds.id, "") if scene is None else scene
    try:
        stamp = os.stat(scene).st_mtime_ns if scene else None
    except OSError:
        stamp = "missing"
    key = (ds.id, scene, stamp)
    with _lock:
        if key in _poses:
            return _poses[key]
    truth = get(ds)
    transforms, error = None, None
    if scene and stamp == "missing":
        error = f"Scene file not found: {scene}"
    elif scene:
        try:
            transforms = _scene_poses(ds, truth, scene)
        except (RuntimeError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
            error = f"Could not read poses from {os.path.basename(scene)}: {exc}"
    source = "scene" if transforms is not None else "boxes"
    result = Poses(hashlib.sha1(repr(key).encode()).hexdigest()[:10], source,
                   transforms if transforms is not None else box_poses(ds, truth), error)
    with _lock:
        _poses[key] = result
        while len(_poses) > 8:
            _poses.popitem(last=False)
    return result


# --------------------------------------------------------------------------- points on the surface

_surfaces: OrderedDict[tuple, np.ndarray] = OrderedDict()


def _camera_at(cams: list, frame: int):
    chosen = cams[0]
    for entry in cams:
        if entry[0] <= frame:
            chosen = entry
    return chosen


def _depth_file(view_dir: str, frame: int) -> tuple[str, str] | None:
    for name in DEPTH_DIRS:
        path = os.path.join(view_dir, name, f"{name}_{frame:04d}.npy")
        if os.path.exists(path):
            return name, path
    return None


def _farthest(points: np.ndarray, k: int) -> np.ndarray:
    """k points spread evenly over the cloud (farthest-point sampling, deterministic)."""
    chosen = [0]
    dist = np.linalg.norm(points - points[0], axis=1)
    for _ in range(min(k, len(points)) - 1):
        i = int(dist.argmax())
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(points - points[i], axis=1))
    return points[chosen]


def surface_points(ds: pv.Dataset, truth: GroundTruth, index: int, pose: Poses) -> np.ndarray:
    """POINTS_PER_OBJECT points on the object's surface, in its own coordinates (see module doc)."""
    key = (ds.id, index, pose.key)
    with _lock:
        if key in _surfaces:
            return _surfaces[key]
    obj = truth.objects[index]
    lo, hi = obj.box_min, obj.box_max
    transforms = pose.transforms.get(obj.path)
    usable = [i for i in range(len(ds.timeline)) if transforms is not None and np.isfinite(transforms[i, 0, 0])]
    if pose.source == "boxes":
        # Box poses are only right while the box keeps its first-frame size (the object has not turned).
        local = np.einsum("nci,ij->ncj", np.concatenate(
            [obj.corners.astype(np.float64), np.ones(obj.corners.shape[:2] + (1,))], axis=-1),
            np.linalg.inv(obj.transform0))[..., :3]
        same = np.all(np.abs((local.max(1) - local.min(1)) - (hi - lo)) < 0.01, axis=1)
        steady = {int(f) for f, ok in zip(obj.frames, same) if ok}
        usable = [i for i in usable if ds.timeline[i] in steady]
    picks = [usable[j] for j in np.linspace(0, len(usable) - 1, min(SAMPLE_FRAMES, len(usable))).astype(int)] if usable else []
    box = np.array([[x, y, z, 1.0] for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])])
    found = []
    for i in picks:
        frame, to_world = ds.timeline[i], transforms[i]
        to_object = np.linalg.inv(to_world)
        box_world = box @ to_world
        for view_dir, cams in zip(ds.view_paths, truth.camera_matrices):
            depth_info = _depth_file(view_dir, frame) if cams else None
            if depth_info is None:
                continue
            _first, view, proj, width, height = _camera_at(cams, frame)
            clip = box_world @ view @ proj
            if np.any(clip[:, 3] <= 1e-6):
                continue                               # part of the box is behind the camera
            ndc = clip[:, :2] / clip[:, 3:4]
            u = (ndc[:, 0] + 1) / 2 * width
            v = (1 - ndc[:, 1]) / 2 * height
            u0, u1 = max(int(u.min()), 0), min(int(np.ceil(u.max())), width)
            v0, v1 = max(int(v.min()), 0), min(int(np.ceil(v.max())), height)
            if u1 <= u0 or v1 <= v0:
                continue                               # not in this view
            kind, depth_path = depth_info
            depth = np.squeeze(np.load(depth_path))
            us, vs = np.meshgrid(np.arange(u0, u1, 2), np.arange(v0, v1, 2))
            dist = depth[vs, us].ravel().astype(np.float64)
            nx = (us.ravel() + 0.5) / width * 2 - 1
            ny = 1 - (vs.ravel() + 0.5) / height * 2
            rays = np.stack([nx / proj[0, 0], ny / proj[1, 1], -np.ones_like(nx)], axis=1)   # camera looks down -Z
            if kind == "distance_to_camera":
                rays /= np.linalg.norm(rays, axis=1, keepdims=True)   # distance along the ray
            cam_points = rays * dist[:, None]                         # (image plane depth: z = -depth)
            world = np.concatenate([cam_points, np.ones((len(cam_points), 1))], axis=1) @ np.linalg.inv(view)
            local_pts = (world @ to_object)[:, :3]
            keep = (np.isfinite(dist) & np.all(local_pts >= lo - BOX_MARGIN_M, axis=1)
                    & np.all(local_pts <= hi + BOX_MARGIN_M, axis=1) & (local_pts[:, 2] > lo[2] + FLOOR_M))
            found.append(local_pts[keep])
    cloud = np.concatenate(found) if found else np.zeros((0, 3))
    if len(cloud):
        cloud = cloud[np.sort(np.unique(np.floor(cloud / VOXEL_M).astype(np.int64), axis=0, return_index=True)[1])]
        points = _farthest(cloud, POINTS_PER_OBJECT)
    else:                                              # no depth data: fall back to the box corners
        points = box[:, :3]
    with _lock:
        _surfaces[key] = points
        while len(_surfaces) > 64:
            _surfaces.popitem(last=False)
    return points


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
    pose = await asyncio.to_thread(poses, ds)
    points = await asyncio.to_thread(surface_points, ds, truth, index, pose)
    transforms = pose.transforms.get(obj.path)
    rows = [i for i in range(len(ds.timeline)) if transforms is not None and np.isfinite(transforms[i, 0, 0])]
    return JSONResponse({
        "path": obj.path,
        "label": obj.label,
        "source": pose.source,
        "points": np.round(points, 4).tolist(),                    # object coordinates
        # centre of the box's bottom face: the point whose path is drawn as the object's route
        "anchor": [round(float((obj.box_min[0] + obj.box_max[0]) / 2), 4),
                   round(float((obj.box_min[1] + obj.box_max[1]) / 2), 4), round(float(obj.box_min[2]), 4)],
        "frames": [ds.timeline[i] for i in rows],
        "transforms": np.round(transforms[rows].reshape(len(rows), 16), 6).tolist() if rows else [],
    }, headers=HEADERS)
