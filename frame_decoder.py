"""PNG -> preview JPEG conversion, run in worker processes by previews.PreviewCache.

Kept in its own small module (Pillow only) so worker processes start quickly. Decoding a
1280x720 PNG takes ~26 ms and Pillow's PNG reader holds Python's global interpreter lock
for part of it, so threads top out near 150 images/s; separate processes scale with cores
(16 processes: ~390 images/s, enough for 6 views at 60 fps).
"""

import io

from PIL import Image

JPEG_QUALITY = 85


def render_preview(path: str, width: int) -> bytes:
    """Decode a PNG, downscale it to the preview width and encode it as JPEG."""
    with Image.open(path) as im:
        im = im.convert("RGB")  # Isaac Sim writes RGBA with an opaque alpha channel
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_QUALITY)
        return buf.getvalue()


def watch_server(server_pid: int) -> None:
    """Worker initializer: exit as soon as the app server process is gone.

    Python's process pools only stop their workers when the server exits normally; if it
    is killed (closed terminal, SIGTERM, SIGKILL) the workers would linger forever.
    """
    import os
    import threading
    import time

    def run() -> None:
        while True:
            try:
                os.kill(server_pid, 0)
            except ProcessLookupError:
                os._exit(0)
            time.sleep(1)

    threading.Thread(target=run, name="watch-server", daemon=True).start()
