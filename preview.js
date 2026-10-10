// Preview player (Streamlit components v2 module; app.py prepends common.js).
//
// Plays the frames of all arranged views as one video on a canvas, in the tile arrangement
// the viewer had when Preview was clicked. The recording has `srcFps` frames per second
// (30 for the Isaac Sim captures). At a playback rate of n FPS the preview still runs in
// real time but shows n frames per second, picked at regular intervals from the recorded
// ones: output frame k shows recorded frame  start + round(k * srcFps / n)
// (15 FPS -> every 2nd frame, 10 FPS -> every 3rd, 12 FPS -> frames 0, 3, 5, 8, 10, ...).
//
// Seconds (optional): length of the clip. X seconds at n FPS show n * X frames, covering X
// seconds of the recording from the start frame; empty plays until the last frame. If the
// recording ends sooner, playback simply stops at its last frame.
//
// Play button: click = play / pause, double-click = stop and go back to the start frame.
// Preview toggle (on): switches the pane back to the multi-view viewer.
//
// Ground truth overlay (data.gt, set by the object picker in app.py): each selected object's
// tracked points lie on its real surface (taken from the recorded depth) and move with its
// per-frame pose (from the scene file, or from the 3D boxes); they are projected into every view
// with each point's trajectory over the last `gt.trail` recorded frames (see groundtruth.py).

const GAP = 8;                // canvas pixels between tiles
const PRELOAD_SECONDS = 1.5;  // browser preloads this much of the upcoming playback
const WARM_SECONDS = 4;       // the server decodes this much ahead (at most 128 frames)
const WIDTHS = [160, 240, 320, 480, 640, 854, 1280];   // image widths requested from the server
const PREROLL_SECONDS = 0.6; // Play starts once this much upcoming video is loaded...
const PREROLL_MAX_MS = 2000;  // ...or after this long at most

preloader.limit = 6;          // while the preview plays the viewer is idle: use every browser connection

const savedFps = new Map();   // dataset folder -> playback FPS
const savedSeconds = new Map(); // dataset folder -> clip length in seconds (null = until the end)
const jsonCache = new Map();  // url -> Promise<object | null>

function fetchJson(url) {
  if (!jsonCache.has(url)) {
    jsonCache.set(url, fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null));
  }
  return jsonCache.get(url);
}

const PLAY_ICON = '<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path d="M8 5v14l11-7z" fill="currentColor"/></svg>';
const PAUSE_ICON = '<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path d="M7 5h4v14H7zM13 5h4v14h-4z" fill="currentColor"/></svg>';

function ordinal(n) {
  const v = n % 100;
  return n + (["th", "st", "nd", "rd"][(v - 20) % 10] || ["th", "st", "nd", "rd"][v] || "th");
}

function buildPlayer(root, data) {
  root.replaceChildren();
  const timeline = expandRuns(data.runs);
  const last = timeline[timeline.length - 1];
  const missing = data.views.map((_, view) => new Set(expandRuns(data.missing[String(view)] || [])));
  const [imgW, imgH] = data.size;
  const src = data.srcFps;
  const cols = data.cols;
  const rows = Math.ceil(data.tiles.length / cols);
  // Canvas pixels per tile: the on-screen size (x device pixel ratio), at most the preview
  // resolution; resizeCanvas() updates them when the pane changes size.
  let tileW = data.w;
  let tileH = Math.round((data.w * imgH) / imgW);
  // Images are requested at the smallest standard width that covers a tile, so a small pane
  // does not make the browser decode far more pixels than it shows.
  const widthFor = (w) => Math.min(data.w, WIDTHS.find((x) => x >= w) || data.w);
  let imageW = data.w;
  const shownViews = data.tiles.filter((view) => view >= 0);

  const stage = el("div", "mvk-stage", root);          // canvas + view labels on top of it
  const canvas = el("canvas", "mvk-canvas", stage);
  canvas.width = cols * tileW + (cols - 1) * GAP;
  canvas.height = rows * tileH + (rows - 1) * GAP;
  const ctx = canvas.getContext("2d");
  const tags = data.tiles.map((view) => {   // HTML labels stay readable however small the video is drawn
    if (view < 0) return null;
    const tag = el("div", "mvk-tile-label", stage);
    tag.textContent = data.views[view];
    return tag;
  });

  function placeTags() {
    tags.forEach((tag, t) => {
      if (!tag) return;
      const x = (t % cols) * (tileW + GAP);
      const yBottom = Math.floor(t / cols) * (tileH + GAP) + tileH;
      tag.style.left = `calc(${(100 * x) / canvas.width}% + 6px)`;
      tag.style.bottom = `calc(${(100 * (canvas.height - yBottom)) / canvas.height}% + 6px)`;
    });
  }

  function resizeCanvas() {   // match the canvas to its size on screen; redraw if it changed
    const shown = stage.getBoundingClientRect().width;
    if (!shown) return;
    const scale = window.devicePixelRatio || 1;
    const w = Math.max(64, Math.min(data.w, Math.round((shown * scale - (cols - 1) * GAP) / cols)));
    if (w === tileW) return;
    tileW = w;
    tileH = Math.round((w * imgH) / imgW);
    canvas.width = cols * tileW + (cols - 1) * GAP;
    canvas.height = rows * tileH + (rows - 1) * GAP;
    placeTags();
    const nextImageW = widthFor(tileW);
    if (nextImageW !== imageW) {           // new image size: reload the frame on screen, then preload
      imageW = nextImageW;
      const frame = p.shown;
      if (frame !== null) request(frame).then(() => { if (p.shown === frame) draw(frame); });
      prefetch();
    } else if (p.shown !== null) {
      draw(p.shown);
    }
  }
  placeTags();

  const legend = el("div", "mvk-legend", root);
  legend.hidden = true;
  const bar = el("div", "mvk-nav", root);
  const play = el("button", "mvk-btn mvk-play", bar);
  play.title = "Click: play / pause.  Double-click: stop and go back to the start frame";
  const fpsLabel = el("label", "mvk-label", bar);
  fpsLabel.append("FPS");
  const fpsInput = el("input", "mvk-input mvk-fps", fpsLabel);
  fpsInput.type = "number";
  fpsInput.min = "1";
  fpsInput.max = String(src);
  fpsInput.step = "1";
  fpsInput.title = `Frames shown per second (1 to ${src}); playback stays real time`;
  const secondsLabel = el("label", "mvk-label", bar);
  secondsLabel.append("Seconds");
  const secondsInput = el("input", "mvk-input mvk-seconds", secondsLabel);
  secondsInput.type = "number";
  secondsInput.min = "0";
  secondsInput.step = "any";
  secondsInput.placeholder = "all";
  secondsInput.title = "How long to play, in seconds of the recording (empty: until the last frame)";
  const status = el("span", "mvk-status mvk-player-status", bar);
  const note = el("div", "mvk-player-note", root);
  const clipNote = el("div", "mvk-player-note mvk-player-clip", root);
  const actions = el("div", "mvk-actions", root);
  const toggle = makeToggle(actions, "Preview", true);
  toggle.title = "Switch back to the multi-view viewer";

  const host = root.getRootNode().host;
  const theme = host ? getComputedStyle(host) : null;
  const tileBg = (theme && theme.getPropertyValue("--st-secondary-background-color").trim()) || "#f0f2f6";
  const mutedText = (theme && theme.getPropertyValue("--st-gray-text-color").trim()) || "#808495";
  const font = (theme && theme.getPropertyValue("--st-font").trim()) || "sans-serif";

  const saved = savedFps.get(data.root) ?? parseInt(storageGet(`mvk:fps:${data.root}`), 10);
  const savedSecs = savedSeconds.has(data.root) ? savedSeconds.get(data.root)
    : parseFloat(storageGet(`mvk:seconds:${data.root}`));
  const origin = timeline[nearestIndex(timeline, data.start)];
  const p = {
    origin,                 // start frame (from the viewer)
    anchor: origin,         // output frame k shows recorded frame anchor + round(k * src / fps)
    k: 0,                   // next output frame to show
    fps: Number.isInteger(saved) && saved >= 1 && saved <= src ? saved : src,
    seconds: Number.isFinite(savedSecs) && savedSecs > 0 ? savedSecs : null,   // clip length; null = to the end
    secondsTimer: 0,
    playing: false,
    ended: false,
    raf: 0,
    nextDue: 0,
    shown: null,            // recorded frame number on the canvas
    preroll: 0,             // time Play was pressed while still waiting for the first frames
    warmAt: 0,
    stillToken: 0,
    setStateValue: null,
    setTriggerValue: null,
    syncTimer: 0,
    observer: null,
    gt: { on: false, objects: [], trail: src },   // overlay settings from app.py
    cams: null,             // per view: [[first frame, world->clip matrix], ...]
    tracks: new Map(),      // object id -> undefined (not requested), null (loading), {pts, n} or {failed}
    poseKey: null,          // pose source the tracks were loaded for
    stats: { draws: 0, drawMs: 0, overlayMs: 0, waits: 0 },   // for diagnosing playback speed
  };
  const warm = makeWarmer();
  const tlIndex = new Map(timeline.map((f, i) => [f, i]));

  const url = (view, frame) => `${data.base}/preview/${data.ds}/${view}/${frame}?w=${imageW}`;
  const urlsFor = (frame) => data.tiles.map((view) =>
    (view >= 0 && !missing[view].has(frame) ? url(view, frame) : null));

  function target(k) {      // recorded frame for output frame k, or null past the end of the clip
    const f = p.anchor + Math.round((k * src) / p.fps);
    if (f > last || f - p.origin > clipOffset()) return null;
    return timeline[nearestIndex(timeline, f)];
  }

  // ---- clip length (Seconds field)

  const ratio = () => src / p.fps;              // recorded frames per shown frame
  // Largest offset from the start frame inside the clip: X seconds = offsets below X * src.
  const clipOffset = () => (p.seconds ? Math.ceil(p.seconds * src - 1e-9) - 1 : Infinity);
  // Shown frames k = 0, 1, ... whose recorded offset round(k * ratio) is at most `offset`.
  const framesUpTo = (offset) => (offset < 0 ? 0 : Math.ceil((offset + 0.5) / ratio() - 1e-9));

  function clipPlan() {
    const available = last - p.origin;
    const total = framesUpTo(Math.min(available, clipOffset()));
    return {
      total,                                                    // frames that will be shown
      requested: p.seconds ? framesUpTo(clipOffset()) : total,  // n * X
      cut: p.seconds !== null && clipOffset() > available,      // the recording ends first
      lastFrame: p.origin + Math.round((total - 1) * ratio()),
    };
  }

  const formatSeconds = (x) => String(Math.round(x * 100) / 100);

  function ready(frame) {
    return urlsFor(frame).every((u) => u === null || ["ok", "error"].includes(loadState(u)));
  }

  function request(frame) {
    return Promise.all(urlsFor(frame).map((u) => (u ? load(u, "high") : true)));
  }

  function draw(frame) {    // all tiles of one recorded frame, in a single canvas update
    const started = performance.now();
    const urls = urlsFor(frame);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    data.tiles.forEach((view, t) => {
      const x = (t % cols) * (tileW + GAP);
      const y = Math.floor(t / cols) * (tileH + GAP);
      ctx.fillStyle = tileBg;
      ctx.fillRect(x, y, tileW, tileH);
      if (view < 0) return;
      const entry = urls[t] && pool.get(urls[t]);
      if (entry && entry.state === "ok") {
        ctx.drawImage(entry.img, x, y, tileW, tileH);
      } else if (urls[t] && (!entry || entry.state === "loading")) {
        // still loading (the pane was just resized): leave the tile blank for a moment
      } else {
        ctx.fillStyle = mutedText;
        ctx.font = `${Math.max(14, Math.round(tileH * 0.05))}px ${font}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(urls[t] ? `cannot load frame ${frame}` : `no frame ${frame}`, x + tileW / 2, y + tileH / 2);
      }
    });
    const overlayStart = performance.now();
    drawOverlay(frame);
    p.shown = frame;
    updateStatus();
    const now = performance.now();
    p.stats.draws++;
    p.stats.drawMs += now - started;
    p.stats.overlayMs += now - overlayStart;
  }

  // ---- ground-truth overlay

  // World positions of the object's n surface points in every timeline frame (NaN = unknown):
  // point p (object coordinates) at frame f is [p 1] @ T(f), T row-major with translation last.
  function prepareTrack(t) {
    const n = t.points.length;
    const pts = new Float32Array(timeline.length * n * 3).fill(NaN);
    t.frames.forEach((frame, j) => {
      const ti = tlIndex.get(frame);
      if (ti === undefined) return;
      const m = t.transforms[j];
      t.points.forEach(([x, y, z], q) => {
        const o = (ti * n + q) * 3;
        pts[o] = x * m[0] + y * m[4] + z * m[8] + m[12];
        pts[o + 1] = x * m[1] + y * m[5] + z * m[9] + m[13];
        pts[o + 2] = x * m[2] + y * m[6] + z * m[10] + m[14];
      });
    });
    return { pts, n, source: t.source };
  }

  function cameraAt(cams, frame) {
    for (let i = cams.length - 1; i > 0; i--) if (cams[i][0] <= frame) return cams[i][1];
    return cams[0][1];
  }

  // Canvas position of point o of `pts` seen by camera matrix m in the tile at (x0, y0), or null.
  function project(m, pts, o, x0, y0) {
    const x = pts[o];
    if (Number.isNaN(x)) return null;
    const y = pts[o + 1], z = pts[o + 2];
    const w = x * m[3] + y * m[7] + z * m[11] + m[15];
    if (w <= 1e-6) return null;                         // behind the camera
    const u = (x * m[0] + y * m[4] + z * m[8] + m[12]) / w;
    const v = (x * m[1] + y * m[5] + z * m[9] + m[13]) / w;
    return [x0 + ((u + 1) / 2) * tileW, y0 + ((1 - v) / 2) * tileH];
  }

  function drawOverlay(frame) {
    if (!p.gt.on || !p.cams || !p.gt.objects.length) return;
    const ti = tlIndex.get(frame);
    if (ti === undefined) return;
    const k = canvas.width / Math.max(1, canvas.getBoundingClientRect().width);   // canvas px per screen px
    const first = Math.max(0, ti - p.gt.trail);
    data.tiles.forEach((view, t) => {
      const cams = view >= 0 && p.cams[view];
      if (!cams || !cams.length) return;
      const x0 = (t % cols) * (tileW + GAP);
      const y0 = Math.floor(t / cols) * (tileH + GAP);
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0, y0, tileW, tileH);
      ctx.clip();
      for (const obj of p.gt.objects) {
        const track = p.tracks.get(obj.id);
        if (!track || !track.pts) continue;
        const n = track.n;
        ctx.beginPath();                                 // trajectories over the last second
        for (let q = 0; q < n; q++) {
          let pen = false;
          for (let i = first; i <= ti; i++) {
            const s = project(cameraAt(cams, timeline[i]), track.pts, (i * n + q) * 3, x0, y0);
            if (!s) { pen = false; continue; }
            if (pen) ctx.lineTo(s[0], s[1]); else ctx.moveTo(s[0], s[1]);
            pen = true;
          }
        }
        ctx.globalAlpha = 0.65;
        ctx.strokeStyle = obj.color;
        ctx.lineWidth = 1.5 * k;
        ctx.lineJoin = "round";
        ctx.stroke();
        ctx.globalAlpha = 1;
        const m = cameraAt(cams, frame);                 // the points now
        const r = 2.6 * k;
        ctx.beginPath();
        for (let q = 0; q < n; q++) {
          const s = project(m, track.pts, (ti * n + q) * 3, x0, y0);
          if (!s) continue;
          ctx.moveTo(s[0] + r, s[1]);
          ctx.arc(s[0], s[1], r, 0, 2 * Math.PI);
        }
        ctx.fillStyle = obj.color;
        ctx.fill();
        ctx.lineWidth = 0.8 * k;
        ctx.strokeStyle = "rgba(0, 0, 0, 0.75)";
        ctx.stroke();
      }
      ctx.restore();
    });
  }

  function renderLegend() {
    legend.replaceChildren();
    legend.hidden = !(p.gt.on && p.gt.objects.length);
    for (const obj of p.gt.objects) {
      const chip = el("span", "mvk-chip", legend);
      el("span", "mvk-dot", chip).style.background = obj.color;
      const track = p.tracks.get(obj.id);
      chip.append(obj.name + (track === null ? "  (loading)" : track && track.failed ? "  (no data)" : ""));
    }
  }

  function redraw() {          // paused: draw the frame on screen again with the new overlay
    if (!p.playing && p.shown !== null) draw(p.shown);
  }

  p.setOverlay = (gt) => {
    p.gt = gt && gt.on ? gt : { on: false, objects: [], trail: src };
    if (p.gt.on && p.gt.poseKey !== p.poseKey) {   // other pose source (scene file changed): reload
      p.poseKey = p.gt.poseKey;
      p.tracks.clear();
    }
    const jobs = [];
    if (p.gt.on && p.gt.objects.length) {
      if (!p.cams) {
        jobs.push(fetchJson(`${data.base}/gt/${data.ds}/cameras`).then((c) => { p.cams = c ? c.views : []; }));
      }
      for (const obj of p.gt.objects) {
        if (p.tracks.get(obj.id) !== undefined) continue;
        p.tracks.set(obj.id, null);
        jobs.push(fetchJson(`${data.base}/gt/${data.ds}/object/${obj.id}?poses=${p.poseKey}`)
          .then((t) => p.tracks.set(obj.id, t ? prepareTrack(t) : { failed: true })));
      }
    }
    renderLegend();
    redraw();
    if (jobs.length) Promise.all(jobs).then(() => { renderLegend(); redraw(); });
  };

  function updateStatus() {
    const f = p.shown ?? p.origin;
    const seconds = ((f - p.origin) / src).toFixed(2);
    const plan = clipPlan();
    const position = Math.max(1, Math.min(plan.total, Math.round((f - p.origin) / ratio()) + 1));
    status.textContent = `frame ${f}  ·  ${seconds} s  ·  ${position} of ${plan.total}` + (p.ended ? "  ·  end" : "");
    const range = `recorded frames ${p.origin} to ${plan.lastFrame}`;
    const length = `${(plan.total / p.fps).toFixed(2)} s`;
    clipNote.textContent = !p.seconds
      ? `Plays to the last frame: ${plan.total} frames at ${p.fps} fps (${range}, ${length})`
      : !plan.cut
        ? `Plays ${formatSeconds(p.seconds)} s × ${p.fps} fps = ${plan.total} frames (${range})`
        : `${formatSeconds(p.seconds)} s × ${p.fps} fps would be ${plan.requested} frames, but the recording ` +
          `ends at frame ${last}: plays ${plan.total} frames (${range}, ${length})`;
    const pick = p.fps === src ? "every recorded frame"
      : src % p.fps === 0 ? `every ${ordinal(src / p.fps)} recorded frame`
      : `${p.fps} of every ${src} recorded frames, evenly spaced`;
    note.textContent = `Start: frame ${p.origin}  ·  recorded at ${src} fps  ·  ${p.fps} fps shows ${pick}`;
  }

  function prefetch() {
    const upcoming = [];
    const horizon = Math.ceil(p.fps * WARM_SECONDS);
    for (let j = 0; j < Math.min(128, horizon); j++) {
      const f = target(p.k + j);
      if (f === null) break;
      upcoming.push(f);
    }
    const urls = [];
    for (const f of upcoming.slice(0, Math.ceil(p.fps * PRELOAD_SECONDS))) {
      for (const u of urlsFor(f)) if (u) urls.push(u);
    }
    preloader.set(urls);
    const now = performance.now();
    if (upcoming.length && now - p.warmAt > 400) {
      p.warmAt = now;
      warm(`${data.base}/warm/${data.ds}?w=${imageW}&v=${shownViews.join(",")}&f=${upcoming.join(",")}`);
    }
  }

  function showStill(k) {   // paused: show output frame k as soon as its images are loaded
    const frame = target(k);
    if (frame === null) return;
    p.k = k + 1;
    const token = ++p.stillToken;
    request(frame).then(() => { if (token === p.stillToken && !p.playing) draw(frame); });
    prefetch();
  }

  function tick(now) {
    if (!p.playing) return;
    if (!root.isConnected) { pause(); return; }
    p.raf = requestAnimationFrame(tick);
    if (now < p.nextDue) return;
    const frame = target(p.k);
    if (frame === null) { finish(); return; }
    if (p.preroll) {          // just pressed Play: start once the next frames are loaded
      const upcoming = [];
      const count = Math.max(4, Math.ceil(p.fps * PREROLL_SECONDS));
      for (let j = 0; j < count; j++) { const f = target(p.k + j); if (f !== null) upcoming.push(f); }
      if (!upcoming.every(ready) && now - p.preroll < PREROLL_MAX_MS) {
        upcoming.forEach(request);
        status.classList.add("mvk-loading");
        return;
      }
      p.preroll = 0;
      status.classList.remove("mvk-loading");
      p.nextDue = now;
    }
    if (!ready(frame)) { p.stats.waits++; request(frame); return; }   // wait for the images, never skip a frame
    draw(frame);
    p.k++;
    const interval = 1000 / p.fps;
    p.nextDue = now - p.nextDue > interval ? now + interval : p.nextDue + interval;
    prefetch();
  }

  function setIcon() {
    play.innerHTML = p.playing ? PAUSE_ICON : PLAY_ICON;
    play.setAttribute("aria-label", p.playing ? "Pause" : "Play");
  }

  function start() {
    const restart = p.ended || target(p.k) === null;   // at the end: play again from the start frame
    if (restart) { p.anchor = p.origin; p.k = 0; p.ended = false; }
    p.stillToken++;
    p.playing = true;
    p.preroll = performance.now();
    p.nextDue = performance.now() + (restart ? 0 : 1000 / p.fps);
    setIcon();
    updateStatus();
    p.raf = requestAnimationFrame(tick);
    prefetch();
  }

  function pause() {
    p.playing = false;
    cancelAnimationFrame(p.raf);
    setIcon();
  }

  function finish() {
    pause();
    p.ended = true;
    updateStatus();
  }

  function stop() {         // double-click: back to the start frame, paused
    pause();
    p.ended = false;
    p.anchor = p.origin;
    showStill(0);
    updateStatus();
  }

  function setFps(value) {
    const fps = Math.max(1, Math.min(src, Math.round(value)));
    fpsInput.value = String(fps);
    if (fps === p.fps) return;
    p.anchor = p.shown ?? p.origin;   // keep the current picture; new spacing from here on
    p.k = 1;
    p.fps = fps;
    p.nextDue = performance.now() + 1000 / fps;
    savedFps.set(data.root, fps);
    storageSet(`mvk:fps:${data.root}`, String(fps));
    clearTimeout(p.syncTimer);
    p.syncTimer = setTimeout(() => { if (p.setStateValue) p.setStateValue("fps", fps); }, 300);
    updateStatus();
    prefetch();
  }

  toggle.addEventListener("click", () => {
    if (!p.setTriggerValue) return;
    toggle.setChecked(false);
    pause();
    p.setTriggerValue("close", true);
  });
  function setSeconds(value) {
    const seconds = Number.isFinite(value) && value > 0 ? Math.round(value * 100) / 100 : null;
    secondsInput.value = seconds === null ? "" : String(seconds);
    if (seconds === p.seconds) return;
    p.seconds = seconds;
    savedSeconds.set(data.root, seconds);
    storageSet(`mvk:seconds:${data.root}`, seconds === null ? "" : String(seconds));
    clearTimeout(p.secondsTimer);
    p.secondsTimer = setTimeout(() => { if (p.setStateValue) p.setStateValue("seconds", seconds); }, 300);
    updateStatus();
    prefetch();
  }

  secondsInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); secondsInput.blur(); }
    else if (e.key === "Escape") { secondsInput.value = p.seconds === null ? "" : String(p.seconds); secondsInput.blur(); }
  });
  secondsInput.addEventListener("change", () => setSeconds(parseFloat(secondsInput.value)));

  play.addEventListener("click", (e) => {
    if (e.detail === 2) stop();                      // second click of a double-click
    else if (e.detail <= 1) (p.playing ? pause : start)();
  });
  fpsInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); fpsInput.blur(); }
    else if (e.key === "Escape") { fpsInput.value = String(p.fps); fpsInput.blur(); }
  });
  fpsInput.addEventListener("change", () => {
    const value = parseFloat(fpsInput.value);
    if (Number.isFinite(value)) setFps(value); else fpsInput.value = String(p.fps);
  });

  // With a fixed panel height (bottom or corner handle) the canvas scales to fit inside it.
  const panel = host ? host.closest(".st-key-mvk_panel") : null;   // the pane shared with the viewer
  const fit = () => {
    resizeCanvas();
    const room = roomForContent(panel, "--mvk-panel-h", root, stage);
    const width = room === null ? null : Math.max(160, (room * canvas.width) / canvas.height);
    const next = width !== null && width < root.getBoundingClientRect().width ? `${Math.floor(width)}px` : "";
    if (stage.style.width !== next) { stage.style.width = next; resizeCanvas(); }
  };
  let fitQueued = false;
  p.observer = new ResizeObserver(() => {
    if (fitQueued) return;
    fitQueued = true;
    requestAnimationFrame(() => { fitQueued = false; fit(); });
  });
  if (panel) p.observer.observe(panel);
  p.observer.observe(root);

  p.destroy = () => { pause(); p.observer.disconnect(); };
  fpsInput.value = String(p.fps);
  secondsInput.value = p.seconds === null ? "" : String(p.seconds);
  setIcon();
  ctx.fillStyle = tileBg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  updateStatus();
  resizeCanvas();           // pick the canvas and image size before the first image request
  showStill(0);
  return p;
}

export default function (component) {
  const { data, parentElement, setStateValue, setTriggerValue } = component;
  if (!data || !data.runs || !data.runs.length) return;
  let root = parentElement.querySelector(".mvk-root");
  if (!root) root = el("div", "mvk-root", parentElement);
  const { gt, ...base } = data;            // the overlay changes without restarting playback
  const config = JSON.stringify(base);
  if (!root.__player || root.__config !== config) {
    if (root.__player) root.__player.destroy();
    root.__player = buildPlayer(root, base);
    root.__config = config;
    root.__gt = undefined;
  }
  root.__player.setStateValue = setStateValue;
  root.__player.setTriggerValue = setTriggerValue;
  const gtConfig = JSON.stringify(gt || null);
  if (root.__gt !== gtConfig) {
    root.__gt = gtConfig;
    root.__player.setOverlay(gt);
  }
  // Playback stops by itself once the pane is closed (tick() checks root.isConnected).
}
