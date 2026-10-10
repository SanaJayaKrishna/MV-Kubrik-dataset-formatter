"""Convert one clip of an Isaac Sim recording into an MV-Kubric scene.

The output follows the generator that made MV-Kubric (ethz-vlg/kubric, branch
multiview-point-tracking, challenges/point_tracking_3d/worker.py) and the loader that reads it
(ethz-vlg/mvtracker, mvtracker/datasets/kubric_multiview_dataset.py). One scene per run:

    <dataset>/mv_kubric/<n>/
        cameras.npz                  camera_positions (V, 3), lookat_positions (V, 3)
        tracked_objects.json         {"names": [...], "ids": [...]}; "dome" = the static background
        tracks_3d.npz                tracks_3d (T, N, 3): world positions of N surface points
        tracks_segmentation_ids.npz  tracks_segmentation_ids (N,): object id of each point
        rgbs.mp4                     all views side by side
        tracks.mp4                   all views with the object tracks drawn on them
        view_<i>/
            rgba_00000.png ...       RGBA images
            depth_00000.tiff ...     float32 distance to the camera centre (Kubric "depth")
            segmentation_00000.png   palette PNG: 0 = background, 1..k = visible objects, most visible first
            metadata.json            flags, metadata, camera, instances (Kubric layout)
            object_id_to_segmentation_id.json
            tracks_2d.npz            tracks_2d (T, N, 2) pixel positions, occlusion (T, N)
            tracks_v1.mp4            all tracks; tracks_v2.mp4: tracks on objects only

Not written, because MVTracker makes them itself: cache/ (its track-selection cache, created
when it loads a scene with cached tracks) and duster-views-0123/ (depth predicted by the DUSt3R
network, made with scripts/estimate_depth_with_duster.py in the MVTracker repository).

Isaac Sim gives no meshes, so the points are taken from the recorded depth instead of mesh
surfaces: background points from the first frame, object points from several frames, each kept
in its object's own coordinates and moved with the object's pose (scene file, see groundtruth.py).
Object pixels come from the recorded instance_segmentation when present, otherwise from the depth
and each object's true 3D box.

Run by app.py in a separate process:
    python converter.py --dataset <folder> --start <frame> --fps <n> [--seconds <x>]
                        [--scene-file <usd>] --progress <file.json> [--parent-pid <pid>]
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groundtruth as gtm  # noqa: E402
import previews as pv  # noqa: E402

OUTPUT_DIR = "mv_kubric"       # created inside the dataset folder
N_POINTS = 18000               # tracked points per scene, as in MV-Kubric
BACKGROUND_SHARE = 0.75        # MV-Kubric puts 59-89 % (mean 75 %) of its points on the background
VOXEL_BACKGROUND_M = 0.05      # depth points are thinned to one per voxel, so that density follows
VOXEL_OBJECT_M = 0.015         # surface area (like Kubric's area-weighted sampling)
BOX_MARGIN_M = 0.02            # object pixels: inside the object's true box (plus this margin)...
FLOOR_M = 0.02                 # ...and higher than this above the bottom of the box
MAX_DEPTH = 1000.0             # Kubric/MVTracker treat larger depths as invalid
SOURCE_FPS = 30                # capture rate of the recordings
VIDEO_SEED = 72                # colour permutation of the track videos (as in the generator)
GRAVITY = [0.0, 0.0, -9.81]

STEPS = [  # key, label, weight in the overall progress
    ("folder", "Create the output folder", 1),
    ("clip", "Read the clip: cameras and object poses", 2),
    ("tracks", "Sample 18000 surface points and track them (tracks_3d.npz)", 8),
    ("views", "Write the views: rgba, depth, segmentation, 2D tracks, metadata", 50),
    ("videos", "Write the videos: tracks_v1/v2 per view, rgbs.mp4, tracks.mp4", 35),
    ("check", "Check the scene the way the MVTracker loader reads it", 4),
]
NOT_WRITTEN = [
    ("cache/", "made by MVTracker when it loads the scene with cached tracks"),
    ("duster-views-0123/", "depth predicted by the DUSt3R network (MVTracker's scripts/estimate_depth_with_duster.py)"),
]


# --------------------------------------------------------------------------- progress file

class Progress:
    """Writes the job's state to a JSON file that the app polls."""

    def __init__(self, path: str, parent_pid: int | None):
        self.path = path
        self.parent_pid = parent_pid
        self.lock = threading.Lock()
        self.last_write = 0.0
        self.state = {
            "status": "running", "started": time.time(), "finished": None, "error": None,
            "output": None, "summary": None, "notes": [],
            "steps": [{"key": k, "label": label, "weight": w, "status": "pending", "detail": "",
                       "done": 0, "total": 0} for k, label, w in STEPS],
            "not_written": [{"name": n, "reason": r} for n, r in NOT_WRITTEN],
        }
        self.write(force=True)

    def _step(self, key):
        return next(s for s in self.state["steps"] if s["key"] == key)

    def start(self, key, detail="", total=0):
        with self.lock:
            s = self._step(key)
            s.update(status="running", detail=detail, done=0, total=total)
        self.write(force=True)

    def advance(self, key, n=1, detail=None):
        with self.lock:
            s = self._step(key)
            s["done"] = min(s["total"], s["done"] + n) if s["total"] else s["done"] + n
            if detail is not None:
                s["detail"] = detail
        self.write()

    def finish(self, key, detail=None):
        with self.lock:
            s = self._step(key)
            s["status"] = "done"
            s["done"] = s["total"]
            if detail is not None:
                s["detail"] = detail
        self.write(force=True)

    def note(self, text):
        with self.lock:
            self.state["notes"].append(text)
        self.write(force=True)

    def set(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)
        self.write(force=True)

    def write(self, force=False):
        now = time.time()
        if not force and now - self.last_write < 0.25:
            return
        if self.parent_pid:
            try:
                os.kill(self.parent_pid, 0)
            except ProcessLookupError:   # the app is gone: stop instead of running on unseen
                os._exit(1)
        with self.lock:
            self.last_write = now
            text = json.dumps(self.state)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------- small helpers

def clip_frames(timeline, start: int, fps: int, seconds: float | None) -> list[int]:
    """Recorded frames shown by the preview: start + round(k * 30 / fps), within the clip."""
    last = timeline[-1]
    limit = math.ceil(seconds * SOURCE_FPS - 1e-9) - 1 if seconds else math.inf
    frames, k = [], 0
    while True:
        offset = round(k * SOURCE_FPS / fps)
        if start + offset > last or offset > limit:
            break
        frame = timeline[int(np.argmin(np.abs(np.asarray(timeline) - (start + offset))))]
        if not frames or frame != frames[-1]:
            frames.append(frame)
        k += 1
    return frames


def quaternion_wxyz(r: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a 3x3 rotation matrix (column-vector convention)."""
    m = r / np.cbrt(np.linalg.det(r)) if np.linalg.det(r) > 0 else r
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def rotation_from_wxyz(q: np.ndarray) -> np.ndarray:
    """Same formula as kornia.geometry.quaternion_to_rotation_matrix (used by the MVTracker loader)."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def angular_velocities(rotations: np.ndarray, fps: float) -> np.ndarray:
    """(T, 3) angular velocity (rad/s, world axes) from (T, 3, 3) rotations, by finite differences."""
    out = np.zeros((len(rotations), 3))
    for k in range(len(rotations)):
        a, b = (k - 1, k + 1) if 0 < k < len(rotations) - 1 else ((k, k + 1) if k == 0 else (k - 1, k))
        if b >= len(rotations) or a < 0 or a == b:
            continue
        delta = rotations[b] @ rotations[a].T
        angle = math.acos(max(-1.0, min(1.0, (np.trace(delta) - 1) / 2)))
        if angle < 1e-9:
            continue
        axis = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])
        axis /= 2 * math.sin(angle)
        out[k] = axis * angle * fps / (b - a)
    return out


def hls_palette(n_colors: int) -> np.ndarray:
    """Kubric's palette (plotting.hls_palette): black, then n evenly spaced hues."""
    hues = (np.linspace(0, 1, int(n_colors) + 1)[:-1] + 0.01) % 1
    palette = [(0.0, 0.0, 0.0)] + [colorsys.hls_to_rgb(h, 0.5, 0.7) for h in hues]
    return np.round(np.array(palette) * 255).astype(np.uint8)


def write_palette_png(seg: np.ndarray, path: Path) -> None:
    data = np.ascontiguousarray(seg.astype(np.uint8))
    image = Image.frombytes("P", (data.shape[1], data.shape[0]), data.tobytes())
    image.putpalette(hls_palette(int(data.max()) + 1).ravel().tolist())
    image.save(path)


def write_json(data, path: Path) -> None:
    """Like kubric.file_io.write_json: sorted keys, indent 4, numpy arrays as lists."""
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(data, f, sort_keys=True, indent=4, default=default)


def hsv_colors(values: np.ndarray) -> np.ndarray:
    """matplotlib's 'hsv' colormap for values in [0, 1)."""
    return (np.array([colorsys.hsv_to_rgb(v % 1.0, 1.0, 1.0) for v in values]) * 255).astype(np.uint8)


def rainbow_colors(values: np.ndarray) -> np.ndarray:
    """matplotlib's 'rainbow' colormap (sin-based gnuplot formula) for values in [0, 1]."""
    v = np.clip(values, 0, 1)
    r = np.abs(2 * v - 0.5)
    g = np.sin(v * np.pi)
    b = np.cos(v * np.pi / 2)
    return (np.clip(np.stack([r, g, b], axis=1), 0, 1) * 255).astype(np.uint8)


def _offsets(radius_in: float, radius_out: float) -> np.ndarray:
    r = int(math.ceil(radius_out))
    ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.sqrt(xs ** 2 + ys ** 2)
    keep = (d <= radius_out) & (d >= radius_in)
    return np.stack([ys[keep], xs[keep]], axis=1)


DOT = _offsets(0, 1.6)      # visible point: small filled dot
RING = _offsets(3.2, 4.6)   # occluded point: hollow circle


def draw_tracks(rgb: np.ndarray, xy: np.ndarray, occluded: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Like the generator's plot_tracks: visible points as dots, occluded ones as hollow circles."""
    out = rgb.copy()
    h, w = out.shape[:2]
    inside = (xy[:, 0] > 0) & (xy[:, 0] < w - 1) & (xy[:, 1] > 0) & (xy[:, 1] < h - 1)
    for stamp, select in ((RING, inside & occluded), (DOT, inside & ~occluded)):
        if not select.any():
            continue
        cy = np.round(xy[select, 1] - 0.5).astype(int)
        cx = np.round(xy[select, 0] - 0.5).astype(int)
        ys = (cy[:, None] + stamp[None, :, 0]).ravel()
        xs = (cx[:, None] + stamp[None, :, 1]).ravel()
        col = np.repeat(colors[select], len(stamp), axis=0)
        ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        out[ys[ok], xs[ok]] = col[ok]
    return out


def grid(images: list[np.ndarray], tracks_layout: bool = False) -> np.ndarray:
    """Views side by side: the generator's layouts for 4 and 10 views (tracks.mp4 with 10 views:
    the first 9 in 3 x 3), otherwise 2 rows."""
    n = len(images)
    if tracks_layout and n == 10:
        return np.concatenate([np.concatenate(images[r * 3:(r + 1) * 3], axis=1) for r in range(3)], axis=0)
    if n == 4:
        return np.concatenate([np.concatenate(images[0:2], axis=0), np.concatenate(images[2:4], axis=0)], axis=1)
    rows = 1 if n <= 2 else 2
    cols = math.ceil(n / rows)
    blank = np.zeros_like(images[0])
    cells = images + [blank] * (rows * cols - n)
    return np.concatenate([np.concatenate(cells[r * cols:(r + 1) * cols], axis=1) for r in range(rows)], axis=0)


def video_writer(path: Path, fps: int):
    import imageio.v2 as imageio
    return imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8,
                              macro_block_size=16, ffmpeg_log_level="error")


# --------------------------------------------------------------------------- output folder

def make_output_root(dataset_root: str, progress: Progress) -> Path:
    """<dataset>/mv_kubric; if the folder belongs to the Isaac Sim container's user, the folder
    is created through that container (docker exec as root) and handed to this user."""
    out = Path(dataset_root) / OUTPUT_DIR
    try:
        out.mkdir(exist_ok=True)
        if os.access(out, os.W_OK):
            return out
    except PermissionError:
        pass
    container, inside = container_path(out)
    if container is None:
        raise PermissionError(
            f"Cannot write to {dataset_root}, and no running Docker container mounts it. Make the folder "
            f"writable for this user, for example:  sudo mkdir {out} && sudo chown {os.getuid()}:{os.getgid()} {out}")
    cmd = ["docker", "exec", "-u", "root", container, "sh", "-c",
           f'mkdir -p "{inside}" && chown {os.getuid()}:{os.getgid()} "{inside}"']
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if run.returncode != 0 or not os.access(out, os.W_OK):
        raise PermissionError(f"Could not create {out} through the '{container}' container: "
                              f"{(run.stderr or run.stdout).strip()}")
    progress.note(f"The dataset folder belongs to the Isaac Sim container's user, so {OUTPUT_DIR}/ was created "
                  f"through the '{container}' container and given to this user (only that folder was changed).")
    return out


def container_path(host_path: Path) -> tuple[str | None, str | None]:
    """A running container that mounts host_path, and the path inside it."""
    try:
        names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True,
                               timeout=30).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    target = os.path.realpath(host_path)
    for name in names:
        try:
            mounts = json.loads(subprocess.run(["docker", "inspect", name, "--format", "{{json .Mounts}}"],
                                               capture_output=True, text=True, timeout=30).stdout or "[]")
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
        for m in mounts:
            source = os.path.realpath(m.get("Source", ""))
            if m.get("RW", True) and source and (target == source or target.startswith(source + os.sep)):
                return name, m["Destination"].rstrip("/") + target[len(source):]
    return None, None


def new_scene_folder(root: Path) -> Path:
    """Next free scene number, like the reference: 1, 2, 3, ..."""
    taken = [int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    number = max(taken, default=0) + 1
    while True:
        folder = root / str(number)
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            number += 1


# --------------------------------------------------------------------------- scene model

class Scene:
    """Everything known about the clip: cameras, depth access, object poses and masks."""

    def __init__(self, ds: pv.Dataset, frames: list[int], fps: int, scene_file: str):
        self.ds, self.frames, self.fps = ds, frames, fps
        self.T, self.V = len(frames), len(ds.views)
        self.truth = gtm.get(ds)
        self.pose = gtm.poses(ds, scene_file)
        index = {f: i for i, f in enumerate(ds.timeline)}
        self.rows = [index[f] for f in frames]
        self.objects = list(self.truth.objects)
        self.object_poses = [self.pose.transforms[o.path][self.rows] for o in self.objects]   # (T, 4, 4) row-vector
        self.W, self.H = ds.image_size
        # ids like the generator's asset order: dome 1, cameras 2..V+1, then the objects
        self.dome_id = 1
        self.object_ids = [self.V + 2 + j for j in range(len(self.objects))]
        leaf = [o.path.rsplit("/", 1)[-1] for o in self.objects]
        self.object_names = [n if leaf.count(n) == 1 else o.path for n, o in zip(leaf, self.objects)]
        self.cameras = [self._camera(v) for v in range(self.V)]
        self.recorded_masks = [self._instance_folder(v) for v in range(self.V)]

    # ---- cameras
    def _camera(self, v: int) -> dict:
        view_dir = self.ds.view_paths[v]
        raw = json.load(open(os.path.join(view_dir, gtm.CAMERA_DIR, f"camera_params_{self.frames[0]:04d}.json")))
        cams = self.truth.camera_matrices[v]
        matrices, worlds = [], []
        for f in self.frames:
            _first, view, proj, _w, _h = gtm._camera_at(cams, f)
            matrices.append((view, proj))
            worlds.append(np.linalg.inv(view).T)   # camera->world, column vectors; camera looks down -Z
        proj0 = matrices[0][1]
        sensor_width = float(raw["cameraAperture"][0])
        fx, fy = proj0[0, 0] / 2, proj0[1, 1] / 2  # normalized focal lengths ([0, 1] image coordinates)
        quaternions = np.array([quaternion_wxyz(c[:3, :3]) for c in worlds])
        positions = np.array([c[:3, 3] for c in worlds])
        return {
            "name": self.ds.views[v], "matrices": matrices, "worlds": np.array(worlds),
            "K": np.array([[fx, 0.0, -0.5], [0.0, -fy, -0.5], [0.0, 0.0, -1.0]]),
            "focal_length": fx * sensor_width, "sensor_width": sensor_width,
            "field_of_view": 2 * math.atan2(sensor_width / 2, fx * sensor_width),
            "positions": positions, "quaternions": quaternions,
            # what the MVTracker loader rebuilds from the quaternions (used for the 2D tracks)
            "rotations": np.array([rotation_from_wxyz(q) for q in quaternions]),
            "rays": self._rays(proj0),
        }

    def _rays(self, proj: np.ndarray) -> np.ndarray:
        """Unit view rays (camera coordinates) through every pixel centre, (H, W, 3)."""
        u = (np.arange(self.W) + 0.5) / self.W * 2 - 1
        v = 1 - (np.arange(self.H) + 0.5) / self.H * 2
        nx, ny = np.meshgrid(u / proj[0, 0], v / proj[1, 1])
        rays = np.stack([nx, ny, -np.ones_like(nx)], axis=-1)
        return (rays / np.linalg.norm(rays, axis=-1, keepdims=True)).astype(np.float32)

    def lookat(self, v: int) -> np.ndarray:
        """Where the camera looks: its optical axis meets the floor (z = 0), else 10 m ahead."""
        c = self.cameras[v]["worlds"][0]
        position, forward = c[:3, 3], -c[:3, 2]
        t = -position[2] / forward[2] if forward[2] < -1e-6 else 10.0
        return position + forward * t

    # ---- per-frame data
    def depth(self, v: int, k: int) -> np.ndarray:
        """Distance to the camera centre (Kubric depth), 0 where unknown."""
        kind, path = gtm._depth_file(self.ds.view_paths[v], self.frames[k])
        d = np.squeeze(np.load(path)).astype(np.float32)
        if kind != "distance_to_camera":    # image-plane depth -> distance along the ray
            d = d / np.abs(self.cameras[v]["rays"][..., 2])
        d[~np.isfinite(d)] = 0
        return d

    def world_points(self, v: int, k: int, depth: np.ndarray, ys=None, xs=None) -> np.ndarray:
        rays = self.cameras[v]["rays"] if ys is None else self.cameras[v]["rays"][ys, xs]
        d = depth if ys is None else depth[ys, xs]
        c = self.cameras[v]["worlds"][k]
        return (rays * d[..., None]) @ c[:3, :3].T + c[:3, 3]

    def _instance_folder(self, v: int) -> str | None:
        folder = os.path.join(self.ds.view_paths[v], "instance_segmentation")
        return folder if os.path.isdir(folder) else None

    def segmentation(self, v: int, k: int, depth: np.ndarray) -> np.ndarray:
        """Raw ids like the generator's: dome (background) 1, each object its id."""
        recorded = self._recorded_segmentation(v, k) if self.recorded_masks[v] else None
        return recorded if recorded is not None else self._derived_segmentation(v, k, depth)

    def _recorded_segmentation(self, v: int, k: int) -> np.ndarray | None:
        """From Isaac Sim's instance_segmentation (raw ids or colours + the id/colour -> prim mapping)."""
        folder, frame = self.recorded_masks[v], self.frames[k]
        png = os.path.join(folder, f"instance_segmentation_{frame:04d}.png")
        mapping = os.path.join(folder, f"instance_segmentation_mapping_{frame:04d}.json")
        if not (os.path.exists(png) and os.path.exists(mapping)):
            return None
        try:
            image = Image.open(png)
            labels = json.load(open(mapping))
            seg = np.full((self.H, self.W), self.dome_id, dtype=np.int32)
            if image.mode in ("RGB", "RGBA"):
                rgba = np.asarray(image.convert("RGBA")).astype(np.int64)
                codes = ((rgba[..., 0] << 24) | (rgba[..., 1] << 16) | (rgba[..., 2] << 8) | rgba[..., 3])
                lut = {}
                for key, path in labels.items():
                    parts = [int(float(p)) for p in key.strip("()[] ").split(",")]
                    parts += [255] * (4 - len(parts))
                    lut[(parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3]] = path
            else:
                codes = np.asarray(image).astype(np.int64)
                lut = {int(key): path for key, path in labels.items() if str(key).lstrip("-").isdigit()}
            for code, prim in lut.items():
                prim = prim if isinstance(prim, str) else str(prim)
                for obj, oid in zip(self.objects, self.object_ids):
                    if prim == obj.path or prim.startswith(obj.path + "/"):
                        seg[codes == code] = oid
            return seg
        except (OSError, ValueError, KeyError):
            return None

    def _derived_segmentation(self, v: int, k: int, depth: np.ndarray) -> np.ndarray:
        """Pixels whose depth point lies inside an object's true 3D box (above the floor)."""
        seg = np.full((self.H, self.W), self.dome_id, dtype=np.int32)
        view, proj = self.cameras[v]["matrices"][k]
        for obj, oid, poses in zip(self.objects, self.object_ids, self.object_poses):
            to_world = poses[k]
            if not np.isfinite(to_world).all():
                continue
            lo, hi = obj.box_min, obj.box_max
            corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            clip = corners @ to_world @ view @ proj
            if np.any(clip[:, 3] <= 1e-6):
                continue
            ndc = clip[:, :2] / clip[:, 3:4]
            u = (ndc[:, 0] + 1) / 2 * self.W
            vv = (1 - ndc[:, 1]) / 2 * self.H
            u0, u1 = max(int(u.min()) - 1, 0), min(int(np.ceil(u.max())) + 1, self.W)
            v0, v1 = max(int(vv.min()) - 1, 0), min(int(np.ceil(vv.max())) + 1, self.H)
            if u1 <= u0 or v1 <= v0:
                continue
            ys, xs = np.mgrid[v0:v1, u0:u1]
            world = self.world_points(v, k, depth, ys, xs)
            local = (np.concatenate([world, np.ones(world.shape[:2] + (1,))], axis=-1) @ np.linalg.inv(to_world))[..., :3]
            inside = (np.all(local >= lo - BOX_MARGIN_M, axis=-1) & np.all(local <= hi + BOX_MARGIN_M, axis=-1)
                      & (local[..., 2] > lo[2] + FLOOR_M) & (depth[ys, xs] > 0))
            seg[ys[inside], xs[inside]] = oid
        return seg


# --------------------------------------------------------------------------- points to track

def sample_tracks(scene: Scene, n_points: int, seed: int, progress: Progress):
    """N surface points (background + objects) and their world positions in every clip frame."""
    rng = np.random.default_rng(seed)
    T = scene.T
    sample_frames = sorted({0, T // 3, (2 * T) // 3, T - 1})
    progress.start("tracks", "Reading depth of the sample frames", total=len(sample_frames) * scene.V + 2)
    background: list[np.ndarray] = []
    per_object: list[list[np.ndarray]] = [[] for _ in scene.objects]
    for k in sample_frames:
        for v in range(scene.V):
            depth = scene.depth(v, k)
            seg = scene.segmentation(v, k, depth)
            ys, xs = np.mgrid[0:scene.H:2, 0:scene.W:2]
            ok = (depth[ys, xs] > 0) & (depth[ys, xs] < MAX_DEPTH)
            ys, xs = ys[ok], xs[ok]
            world = scene.world_points(v, k, depth, ys, xs)
            ids = seg[ys, xs]
            if k == 0:
                background.append(world[ids == scene.dome_id])
            for j, (oid, poses) in enumerate(zip(scene.object_ids, scene.object_poses)):
                sel = ids == oid
                if sel.any() and np.isfinite(poses[k]).all():
                    local = np.concatenate([world[sel], np.ones((sel.sum(), 1))], axis=1) @ np.linalg.inv(poses[k])
                    per_object[j].append(local[:, :3])
            progress.advance("tracks", detail=f"Depth of frame {scene.frames[k]}, {scene.ds.views[v]}")

    def thin(points: list[np.ndarray], voxel: float) -> np.ndarray:
        if not points:
            return np.zeros((0, 3))
        cloud = np.concatenate(points)
        keep = np.unique(np.floor(cloud / voxel).astype(np.int64), axis=0, return_index=True)[1]
        return cloud[np.sort(keep)]

    bg = thin(background, VOXEL_BACKGROUND_M)
    objs = [thin(p, VOXEL_OBJECT_M) for p in per_object]
    available = np.array([len(o) for o in objs])
    n_obj = int(min(round(n_points * (1 - BACKGROUND_SHARE)), available.sum()))
    share = available / available.sum() if available.sum() else available.astype(float)
    alloc = np.minimum(np.floor(share * n_obj).astype(int), available)
    for j in np.argsort(-available):                      # hand out what rounding left over
        if alloc.sum() >= n_obj:
            break
        alloc[j] += min(available[j] - alloc[j], n_obj - alloc.sum())
    n_bg = min(n_points - int(alloc.sum()), len(bg))
    progress.advance("tracks", detail="Choosing the points")

    tracks, seg_ids, sources = [], [], []
    pick = rng.choice(len(bg), n_bg, replace=False) if n_bg else np.zeros(0, int)
    tracks.append(np.repeat(bg[pick][None], T, axis=0))
    seg_ids.append(np.full(n_bg, scene.dome_id, dtype=np.int32))
    for j, (obj_points, count) in enumerate(zip(objs, alloc)):
        if count == 0:
            continue
        local = obj_points[rng.choice(len(obj_points), count, replace=False)]
        homo = np.concatenate([local, np.ones((count, 1))], axis=1)
        tracks.append(np.einsum("ni,tij->tnj", homo, scene.object_poses[j])[..., :3])
        seg_ids.append(np.full(count, scene.object_ids[j], dtype=np.int32))
        sources.append(f"{scene.object_names[j]}: {count}")
    tracks_3d = np.concatenate(tracks, axis=1).astype(np.float64)
    tracks_ids = np.concatenate(seg_ids)
    order = rng.permutation(tracks_3d.shape[1])           # mixed order, like Kubric's surface sampling
    tracks_3d, tracks_ids = tracks_3d[:, order], tracks_ids[order]
    progress.finish("tracks", f"{tracks_3d.shape[1]} points: background {n_bg}, " + ", ".join(sources))
    return tracks_3d, tracks_ids


# --------------------------------------------------------------------------- one view

def project(scene: Scene, v: int, k: int, points: np.ndarray):
    """Pixel positions (the generator's K @ inv(matrix_world) @ p, times the resolution),
    z1 (in front of the camera) and z2 (distance to the camera, like the depth)."""
    cam = scene.cameras[v]
    r, t = cam["rotations"][k], cam["positions"][k]
    in_camera = (points - t) @ r                        # inverse of [R | t] applied to the points
    projected = in_camera @ cam["K"].T
    z1 = projected[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = projected[:, :2] / z1[:, None]
    xy = xy * np.array([scene.W, scene.H])
    z2 = np.linalg.norm(points - t, axis=1)
    return xy, z1, z2


def occlusion_by_depth_and_segment(depth, seg, xy, z2, track_ids):
    """The generator's estimate_occlusion_by_depth_and_segment for one frame."""
    h, w = depth.shape
    x = np.nan_to_num(xy[:, 0], nan=-1e6, posinf=1e6, neginf=-1e6) - 0.5
    y = np.nan_to_num(xy[:, 1], nan=-1e6, posinf=1e6, neginf=-1e6) - 0.5
    x0 = np.clip(np.floor(x).astype(np.int64), 0, w - 1)
    x1 = np.clip(np.floor(x).astype(np.int64) + 1, 0, w - 1)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, h - 1)
    y1 = np.clip(np.floor(y).astype(np.int64) + 1, 0, h - 1)
    neighbours = [(y0, x0), (y1, x0), (y0, x1), (y1, x1)]
    d = np.max([depth[a, b] for a, b in neighbours], axis=0)
    depth_occluded = d < z2 * 0.99
    seg_occluded = np.ones(len(track_ids), dtype=bool)
    for a, b in neighbours:
        seg_occluded &= seg[a, b] != track_ids
    return depth_occluded | seg_occluded


def write_view(scene: Scene, v: int, out: Path, tracks_3d: np.ndarray, track_ids: np.ndarray,
               flags: dict, scene_meta: dict, progress: Progress, lock: threading.Lock):
    T, N = tracks_3d.shape[:2]
    folder = out / f"view_{v}"
    folder.mkdir()
    cam = scene.cameras[v]
    tracks_2d = np.zeros((T, N, 2))
    occlusion = np.zeros((T, N), dtype=bool)
    raw_masks = []
    visibility = {oid: np.zeros(T, dtype=np.int64) for oid in scene.object_ids}
    for k, frame in enumerate(scene.frames):
        rgb_path = scene.ds.frames[v].get(frame)
        rgba = Image.open(rgb_path).convert("RGBA") if rgb_path else Image.new("RGBA", (scene.W, scene.H))
        rgba.save(folder / f"rgba_{k:05d}.png", compress_level=3)
        depth = scene.depth(v, k)
        Image.fromarray(depth).save(folder / f"depth_{k:05d}.tiff")          # float32 -> mode "F"
        seg = scene.segmentation(v, k, depth)
        raw_masks.append(seg.astype(np.uint16))
        counts = np.bincount(seg.ravel(), minlength=max(scene.object_ids) + 1)
        for oid in scene.object_ids:
            visibility[oid][k] = counts[oid]
        xy, z1, z2 = project(scene, v, k, tracks_3d[k])
        occluded = (z1 <= 0) | (z1 > MAX_DEPTH) | (z2 <= 0) | (z2 > MAX_DEPTH) | ~np.isfinite(xy).all(axis=1)
        occluded |= (xy[:, 1] < 0) | (xy[:, 0] < 0) | (xy[:, 1] > scene.H - 1) | (xy[:, 0] > scene.W - 1)
        occluded |= occlusion_by_depth_and_segment(depth, seg, xy, z2, track_ids)
        tracks_2d[k], occlusion[k] = xy, occluded
        with lock:
            progress.advance("views", detail=f"{scene.ds.views[v]} -> view_{v}: frame {k + 1}/{T}")

    # Segmentation ids per view: visible objects 1..n, most visible first; background 0.
    visible = sorted([oid for oid in scene.object_ids if visibility[oid].sum() > 0],
                     key=lambda oid: visibility[oid].sum(), reverse=True)
    new_id = {oid: i + 1 for i, oid in enumerate(visible)}
    lut = np.zeros(max(scene.object_ids) + 1, dtype=np.uint8)
    for oid, nid in new_id.items():
        lut[oid] = nid
    boxes = {oid: ([], []) for oid in visible}
    for k, raw in enumerate(raw_masks):
        seg = lut[raw]
        write_palette_png(seg, folder / f"segmentation_{k:05d}.png")
        for oid, nid in new_id.items():
            ys, xs = np.nonzero(seg == nid)
            if ys.size:
                boxes[oid][0].append((float(ys.min() / scene.H), float(xs.min() / scene.W),
                                      float((ys.max() + 1) / scene.H), float((xs.max() + 1) / scene.W)))
                boxes[oid][1].append(k)

    np.savez(folder / "tracks_2d.npz", tracks_2d=tracks_2d, occlusion=occlusion)
    mapping = {"dome": 0}
    mapping.update({name: 0 for name in scene.ds.views})                       # cameras
    mapping.update({name: new_id.get(oid, 0) for name, oid in zip(scene.object_names, scene.object_ids)})
    write_json(mapping, folder / "object_id_to_segmentation_id.json")

    instances = []
    for oid in visible:
        j = scene.object_ids.index(oid)
        obj, poses = scene.objects[j], scene.object_poses[j]
        rotations = np.array([p[:3, :3].T for p in poses])                     # row-vector -> column-vector
        positions = np.array([p[3, :3] for p in poses])
        lo, hi = obj.box_min, obj.box_max
        corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        image_positions = []
        for k in range(T):
            xy, _z1, _z2 = project(scene, v, k, positions[k][None])
            image_positions.append(xy[0] / np.array([scene.W, scene.H]))
        moved = float(np.nanmax(np.linalg.norm(positions - positions[0], axis=1))) if T else 0.0
        instances.append({
            "asset_id": obj.path,
            "category": obj.label,
            "description": f"{obj.label} ({obj.path})",
            "is_dynamic": bool(moved > 0.01),
            "mass": None, "friction": None, "restitution": None,                 # not known from the recording
            "nr_faces": None, "nr_vertices": None, "surface_area": None, "volume": None,
            "scale": 1.0,
            "positions": positions,
            "quaternions": np.array([quaternion_wxyz(r) for r in rotations]),
            "velocities": np.gradient(positions, axis=0) * scene.fps if T > 1 else np.zeros((T, 3)),
            "angular_velocities": angular_velocities(rotations, scene.fps),
            "image_positions": np.array(image_positions, dtype=np.float32),
            "bboxes_3d": np.einsum("ci,tij->tcj", corners, poses)[..., :3],
            "bboxes": boxes[oid][0],
            "bbox_frames": boxes[oid][1],
            "visibility": visibility[oid].tolist(),
        })
    write_json({
        "flags": flags,
        "metadata": {**scene_meta, "num_instances": len(visible)},
        "camera": {
            "focal_length": cam["focal_length"], "sensor_width": cam["sensor_width"],
            "field_of_view": cam["field_of_view"], "positions": cam["positions"],
            "quaternions": cam["quaternions"], "K": cam["K"], "R": cam["worlds"][0],
        },
        "instances": instances,
    }, folder / "metadata.json")
    return tracks_2d, occlusion


# --------------------------------------------------------------------------- videos

def write_videos(scene: Scene, out: Path, tracks_3d, track_ids, views_2d, progress: Progress):
    T, N = tracks_3d.shape[:2]
    progress.start("videos", "Drawing the tracks", total=T)
    group = np.sqrt((tracks_3d[0] ** 2).sum(axis=1))                        # tracks_v1: distance from the origin
    colors_v1 = hsv_colors(group / (group.max() + 1))
    perm = np.random.default_rng(VIDEO_SEED).permutation(N)                  # tracks_v2: random colours
    colors_v2_all = rainbow_colors(perm / max(N, 1))
    selectors = []
    for tracks_2d, occlusion in views_2d:
        selector = (track_ids != scene.dome_id) & ((~occlusion).sum(0) > 0)
        selector &= np.cumsum(selector) <= 18000
        selectors.append(selector)
    writers_v1 = [video_writer(out / f"view_{v}" / "tracks_v1.mp4", scene.fps) for v in range(scene.V)]
    writers_v2 = [video_writer(out / f"view_{v}" / "tracks_v2.mp4", scene.fps) for v in range(scene.V)]
    writer_rgbs = video_writer(out / "rgbs.mp4", scene.fps)
    writer_tracks = video_writer(out / "tracks.mp4", scene.fps)
    try:
        for k in range(T):
            rgbs, v2_frames = [], []
            for v, ((tracks_2d, occlusion), selector) in enumerate(zip(views_2d, selectors)):
                rgb = np.asarray(Image.open(out / f"view_{v}" / f"rgba_{k:05d}.png").convert("RGB"))
                rgbs.append(rgb)
                writers_v1[v].append_data(draw_tracks(rgb, tracks_2d[k], occlusion[k], colors_v1))
                v2 = draw_tracks(rgb, tracks_2d[k][selector], occlusion[k][selector], colors_v2_all[selector])
                writers_v2[v].append_data(v2)
                v2_frames.append(v2)
            writer_rgbs.append_data(grid(rgbs))
            writer_tracks.append_data(grid(v2_frames, tracks_layout=True))
            progress.advance("videos", detail=f"Frame {k + 1}/{T}")
    finally:
        for w in writers_v1 + writers_v2 + [writer_rgbs, writer_tracks]:
            w.close()
    progress.finish("videos", f"{2 * scene.V + 2} videos at {scene.fps} fps")


# --------------------------------------------------------------------------- loader-style check

def check_scene(out: Path, progress: Progress) -> str:
    """The checks of MVTracker's getitem_raw_datapoint, done with numpy."""
    progress.start("check", "Reading the scene back")
    tracks_3d = np.load(out / "tracks_3d.npz")["tracks_3d"]
    ids = np.load(out / "tracks_segmentation_ids.npz")["tracks_segmentation_ids"]
    json.load(open(out / "tracked_objects.json"))
    cams = np.load(out / "cameras.npz")
    T, N = tracks_3d.shape[:2]
    V = cams["camera_positions"].shape[0]
    assert tracks_3d.shape == (T, N, 3) and ids.shape == (N,)
    assert cams["lookat_positions"].shape == (V, 3)
    views = sorted([p for p in out.iterdir() if p.is_dir() and p.name.startswith("view_")],
                   key=lambda p: int(p.name.split("_")[-1]))
    assert len(views) == V, f"{len(views)} view folders for {V} cameras"
    worst = 0.0
    for view in views:
        names = sorted(os.listdir(view))
        assert sum(n.startswith("rgba_") for n in names) == T
        assert sum(n.startswith("depth_") for n in names) == T
        t2 = np.load(view / "tracks_2d.npz")
        assert t2["tracks_2d"].shape == (T, N, 2) and t2["occlusion"].shape == (T, N)
        meta = json.load(open(view / "metadata.json"))
        json.load(open(view / "object_id_to_segmentation_id.json"))
        K = np.array(meta["camera"]["K"])
        positions = np.array(meta["camera"]["positions"])
        quats = np.array(meta["camera"]["quaternions"])
        assert K.shape == (3, 3) and positions.shape == (T, 3) and quats.shape == (T, 4)
        w, h = meta["metadata"]["resolution"]
        extr_inv = np.eye(4)
        extr_inv[:3, :3], extr_inv[:3, 3] = rotation_from_wxyz(quats[0]), positions[0]
        extr = np.diag([1, -1, -1]) @ np.linalg.inv(extr_inv)[:3, :]
        intr = np.diag([w, h, 1]) @ K @ np.diag([1, -1, -1])
        p = intr @ extr @ np.append(tracks_3d[0, 0], 1.0)
        error = float(np.abs(p[:2] / p[2] - t2["tracks_2d"][0, 0]).max())
        worst = max(worst, error)
        assert error < 1e-3, f"{view.name}: projection of point 0 is off by {error} px"
        depth = np.asarray(Image.open(view / "depth_00000.tiff"))
        assert depth.dtype == np.float32 and depth.shape == (h, w)
        progress.advance("check")
    detail = f"{V} views x {T} frames, {N} tracks; point projection matches to {worst:.1e} px"
    progress.finish("check", detail)
    return detail


# --------------------------------------------------------------------------- main

def convert(args, progress: Progress) -> None:
    root = os.path.abspath(args.dataset)
    ds = pv.index_dataset(root, pv.folder_signature(Path(root)))

    progress.start("folder", f"{root}/{OUTPUT_DIR}")
    out_root = make_output_root(root, progress)
    out = new_scene_folder(out_root)
    progress.set(output=str(out))
    progress.finish("folder", str(out))

    progress.start("clip", "Reading cameras and poses")
    frames = clip_frames(ds.timeline, args.start, args.fps, args.seconds)
    if len(frames) < 2:
        raise ValueError("The clip has fewer than 2 frames; choose an earlier start frame or more seconds.")
    scene = Scene(ds, frames, args.fps, args.scene_file or "")
    if scene.pose.source != "scene":
        progress.note("No exact poses from a scene file: object motion comes from the recorded 3D boxes, "
                      "which keep the first frame's orientation (objects that turn are tracked wrongly). "
                      + (scene.pose.error or ""))
    if not all(scene.recorded_masks):
        progress.note("No instance_segmentation in the recording: object pixels come from the depth and each "
                      "object's true 3D box. Record instance_segmentation for exact masks.")
    if len(frames) < 24:
        progress.note(f"The clip has {len(frames)} frames; MVTracker trains and evaluates on 24-frame windows "
                      "(12 fps x 2 s in the preview gives exactly 24).")
    progress.finish("clip", f"{len(frames)} frames ({frames[0]} to {frames[-1]}) at {args.fps} fps from "
                            f"{scene.V} views; poses from {'the scene file' if scene.pose.source == 'scene' else 'the 3D boxes'}")

    seed = int(out.name) if out.name.isdigit() else 0
    tracks_3d, track_ids = sample_tracks(scene, N_POINTS, seed, progress)
    np.savez(out / "tracks_3d.npz", tracks_3d=tracks_3d)
    np.savez(out / "tracks_segmentation_ids.npz", tracks_segmentation_ids=track_ids)
    write_json({"names": ["dome"] + scene.object_names, "ids": [scene.dome_id] + scene.object_ids},
               out / "tracked_objects.json")
    np.savez(out / "cameras.npz",
             camera_positions=np.array([c["positions"][0] for c in scene.cameras]),
             lookat_positions=np.array([scene.lookat(v) for v in range(scene.V)]))

    flags = {
        # the generator's flags, with this conversion's values (None where they do not apply)
        "backgrounds_split": None, "camera": "fixed", "floor_friction": None, "floor_restitution": None,
        "frame_end": len(frames), "frame_rate": args.fps, "frame_start": 1, "gso_assets": None,
        "hdri_assets": None, "job_dir": str(out), "kubasic_assets": None, "logging_level": "INFO",
        "max_camera_movement": 0.0, "max_motion_blur": 0.0, "max_num_dynamic_objects": None,
        "max_num_static_objects": None, "min_num_dynamic_objects": None, "min_num_static_objects": None,
        "n_cameras": scene.V, "n_points": int(tracks_3d.shape[1]), "objects_split": None,
        "resolution": [scene.W, scene.H], "save_state": False, "scratch_dir": None, "seed": seed,
        "show_debug_plots": False, "step_rate": SOURCE_FPS,
        # where the scene came from
        "source": "isaac_sim", "source_dataset": root, "source_views": list(ds.views),
        "source_frames": frames, "source_fps": SOURCE_FPS, "scene_file": args.scene_file or None,
        "pose_source": scene.pose.source,
        "segmentation_source": "instance_segmentation" if all(scene.recorded_masks) else "depth_and_3d_boxes",
    }
    scene_meta = {
        "background": Path(args.scene_file).stem if args.scene_file else Path(root).name,
        "frame_rate": args.fps, "gravity": GRAVITY, "motion_blur": 0.0, "num_frames": len(frames),
        "resolution": [scene.W, scene.H], "seed": seed, "step_rate": SOURCE_FPS,
    }

    progress.start("views", f"{scene.V} views", total=scene.V * len(frames))
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=min(scene.V, os.cpu_count() or 4)) as pool:
        jobs = [pool.submit(write_view, scene, v, out, tracks_3d, track_ids, flags, scene_meta, progress, lock)
                for v in range(scene.V)]
        views_2d = [job.result() for job in jobs]
    progress.finish("views", f"{scene.V} view folders, {len(frames)} frames each")

    write_videos(scene, out, tracks_3d, track_ids, views_2d, progress)
    check = check_scene(out, progress)
    objects = [n for n, oid in zip(scene.object_names, scene.object_ids) if (track_ids == oid).any()]
    progress.set(status="done", finished=time.time(), summary={
        "scene": out.name, "frames": len(frames), "views": scene.V, "tracks": int(tracks_3d.shape[1]),
        "objects": objects, "fps": args.fps, "first_frame": frames[0], "last_frame": frames[-1],
        "check": check,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--fps", type=int, default=SOURCE_FPS)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--scene-file", default="")
    ap.add_argument("--progress", required=True)
    ap.add_argument("--parent-pid", type=int, default=None)
    args = ap.parse_args()
    progress = Progress(args.progress, args.parent_pid)
    try:
        convert(args, progress)
    except Exception as exc:  # reported in the app's conversion pane
        for step in progress.state["steps"]:
            if step["status"] == "running":
                step["status"] = "failed"
        error = f"{type(exc).__name__}: {exc}"
        if remove_incomplete(progress.state.get("output")):
            error += " (the incomplete scene folder was removed)"
        progress.set(status="failed", finished=time.time(), error=error, traceback=traceback.format_exc()[-4000:])
        sys.exit(1)


def remove_incomplete(output: str | None) -> bool:
    """Delete a half-written scene folder (it would break the MVTracker loader)."""
    if not output or not os.path.isdir(output) or not Path(output).name.isdigit():
        return False
    if Path(output).parent.name != OUTPUT_DIR:
        return False
    import shutil
    shutil.rmtree(output, ignore_errors=True)
    return True


if __name__ == "__main__":
    main()
