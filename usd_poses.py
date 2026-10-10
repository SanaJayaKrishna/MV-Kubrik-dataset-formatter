#!/usr/bin/env python3
"""Export exact per-frame world poses of labelled prims straight from a USD stage.

Why: for keyframe-animated prims, the Synthetic Data Recorder's bounding_box_3d 'transform' stays at
its frame-0 value (only the box extents move), so it cannot give the object's orientation over time.
The stage itself has the truth: this reads it at every frame.

Usage:
    python3 usd_object_poses.py <stage.usd> [--frames START END] [--prim /World/Path]... [--out poses.npz]

    Without --prim, every prim that carries a semantic label is exported.
    Without --frames, the stage's start..end time codes are used (frame i == time code i when the
    recorder's "Control Timeline" is on and Play started at time code 0).
    Runs itself with Isaac Sim's bundled Python if 'pxr' is not importable (looks in ~/isaacsim,
    $ISAACSIM_PATH or /isaac-sim).

Output npz:
    frame_ids       (T,)        time codes
    object_ids      (N,)        prim paths
    object_labels   (N,)        semantic class ('' if none)
    transforms      (T, N, 4, 4) object->world, USD row-vector convention: p_world_h = p_local_h @ M
    positions       (T, N, 3)   world position of the prim origin
    headings_deg    (T, N)      rotation about world Z in degrees (yaw), for ground vehicles
    fps                          stage timeCodesPerSecond
"""
import argparse
import glob
import os
import sys


def _reexec_with_isaacsim_python() -> None:
    for root in (os.environ.get("ISAACSIM_PATH", ""), os.path.expanduser("~/isaacsim"), "/isaac-sim"):
        libs = sorted(glob.glob(os.path.join(root, "extscache", "omni.usd.libs-*"))) if root else []
        py = os.path.join(root, "kit", "python", "bin", "python3")
        if libs and os.path.exists(py):
            env = dict(os.environ)
            # numpy lives in Kit's pip prebundle, not in the interpreter's own site-packages
            prebundles = sorted(glob.glob(os.path.join(root, "extscache", "omni.kit.pip_archive-*", "pip_prebundle")))
            prebundles += sorted(glob.glob(os.path.join(root, "exts", "omni.isaac.core_archive", "pip_prebundle")))
            env["PYTHONPATH"] = os.pathsep.join([libs[-1]] + prebundles + [env.get("PYTHONPATH", "")])
            env["LD_LIBRARY_PATH"] = os.path.join(libs[-1], "bin") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            env["_USD_POSES_REEXEC"] = "1"
            os.execve(py, [py, os.path.abspath(__file__)] + sys.argv[1:], env)
    sys.exit("The 'pxr' module is not available and no Isaac Sim install was found "
             "(set ISAACSIM_PATH to the folder that contains kit/ and extscache/).")


try:
    from pxr import Usd, UsdGeom, Gf  # noqa: E402
except ImportError:
    if os.environ.get("_USD_POSES_REEXEC"):
        raise
    _reexec_with_isaacsim_python()

import math
import numpy as np


def semantic_label(prim) -> str:
    labels = []
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if name.startswith("semantics:labels:") or (name.startswith("semantic:") and name.endswith(":params:semanticData")):
            value = attr.Get()
            if value is None:
                continue
            labels += list(value) if hasattr(value, "__iter__") and not isinstance(value, str) else [str(value)]
    return ",".join(str(v) for v in labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage")
    ap.add_argument("--frames", nargs=2, type=int, metavar=("START", "END"), help="inclusive time-code range")
    ap.add_argument("--prim", action="append", default=[], help="prim path to export (repeatable)")
    ap.add_argument("--out", default=None, help="output .npz (default: <stage>_poses.npz next to the stage)")
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.stage)
    if stage is None:
        sys.exit(f"Could not open {args.stage}")
    fps = stage.GetTimeCodesPerSecond()
    start, end = args.frames if args.frames else (int(stage.GetStartTimeCode()), int(stage.GetEndTimeCode()))
    frame_ids = list(range(start, end + 1))

    if args.prim:
        prims = [stage.GetPrimAtPath(p) for p in args.prim]
        missing = [p for p, pr in zip(args.prim, prims) if not pr]
        if missing:
            sys.exit(f"Prim(s) not found: {missing}")
    else:
        prims = [p for p in stage.Traverse() if semantic_label(p)]
        if not prims:
            sys.exit("No semantically labelled prims found; pass --prim explicitly")

    N, T = len(prims), len(frame_ids)
    transforms = np.zeros((T, N, 4, 4))
    positions = np.zeros((T, N, 3))
    headings = np.zeros((T, N))
    xf_cache = UsdGeom.XformCache()
    for t, fid in enumerate(frame_ids):
        xf_cache.SetTime(Usd.TimeCode(fid))
        for k, prim in enumerate(prims):
            m = xf_cache.GetLocalToWorldTransform(prim)
            M = np.array([[m[i][j] for j in range(4)] for i in range(4)])
            transforms[t, k] = M
            positions[t, k] = M[3, :3]
            headings[t, k] = math.degrees(math.atan2(M[0, 1], M[0, 0]))  # local +X axis direction in world XY
    object_ids = [str(p.GetPath()) for p in prims]
    object_labels = [semantic_label(p) for p in prims]
    out = args.out or os.path.splitext(args.stage)[0] + "_poses.npz"
    np.savez(out, frame_ids=np.array(frame_ids), object_ids=np.array(object_ids), object_labels=np.array(object_labels),
             transforms=transforms, positions=positions, headings_deg=headings, fps=fps)
    print(f"{out}: {T} frames ({start}..{end} @ {fps:g} fps) x {N} prim(s)")
    for k, (oid, lab) in enumerate(zip(object_ids, object_labels)):
        path_len = float(np.linalg.norm(np.diff(positions[:, k], axis=0), axis=1).sum())
        print(f"  {lab or '(no label)':<12} {oid}: start {positions[0, k].round(3).tolist()} end {positions[-1, k].round(3).tolist()}, "
              f"path {path_len:.2f} m, heading {headings[0, k]:.0f}° -> {headings[-1, k]:.0f}°")


if __name__ == "__main__":
    main()
