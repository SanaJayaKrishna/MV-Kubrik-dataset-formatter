"""Background MV-Kubric conversions: converter.py runs in its own process and reports its progress
in a JSON file (~/.cache/mvkubric_app/jobs/<id>.json) that the app's conversion pane polls.

Kept out of app.py on purpose: Streamlit re-runs app.py on every interaction, while imported
modules (and the process handles kept here) stay alive.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

JOBS_DIR = Path.home() / ".cache" / "mvkubric_app" / "jobs"
CONVERTER = Path(__file__).with_name("converter.py")

_processes: dict[str, subprocess.Popen] = {}


def start(dataset_root: str, start_frame: int, fps: int, seconds: float | None, scene_file: str,
          clip: dict) -> dict:
    """Launch a conversion; returns the job record the app keeps in its session state."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{len(_processes)}"
    progress = JOBS_DIR / f"{job_id}.json"
    log = JOBS_DIR / f"{job_id}.log"
    cmd = [sys.executable, str(CONVERTER), "--dataset", dataset_root, "--start", str(start_frame),
           "--fps", str(fps), "--progress", str(progress), "--parent-pid", str(os.getpid())]
    if seconds:
        cmd += ["--seconds", str(seconds)]
    if scene_file:
        cmd += ["--scene-file", scene_file]
    with open(log, "w") as log_file:
        _processes[job_id] = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                              cwd=str(CONVERTER.parent))
    return {"id": job_id, "progress": str(progress), "log": str(log), "dataset": dataset_root,
            "clip": clip, "started": time.time()}


def read(job: dict) -> dict:
    """The job's current state (from its progress file), with crashes and stops filled in."""
    try:
        state = json.loads(Path(job["progress"]).read_text())
    except (OSError, ValueError):
        state = {"status": "starting", "steps": [], "notes": [], "not_written": [], "started": job["started"]}
    process = _processes.get(job["id"])
    if state.get("status") in ("starting", "running") and process is not None and process.poll() is not None:
        state = {**state, "status": "failed",
                 "error": f"The converter stopped unexpectedly (exit code {process.returncode}). "
                          f"Last lines of its log ({job['log']}):\n{_tail(job['log'])}"}
    return state


def is_running(job: dict | None) -> bool:
    return bool(job) and read(job).get("status") in ("starting", "running")


def stop(job: dict) -> None:
    """Stop a running conversion and remove its incomplete scene folder."""
    process = _processes.get(job["id"])
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(10)
        except subprocess.TimeoutExpired:
            process.kill()
    state = read(job)
    sys.path.insert(0, str(CONVERTER.parent))
    import converter
    removed = converter.remove_incomplete(state.get("output")) if state.get("status") != "done" else False
    for step in state.get("steps", []):
        if step["status"] == "running":
            step["status"] = "stopped"
    state.update(status="stopped", finished=time.time(),
                 error="Stopped." + (" The incomplete scene folder was removed." if removed else ""))
    Path(job["progress"]).write_text(json.dumps(state))


def _tail(path: str, lines: int = 12) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""
