// Preview player (Streamlit components v2 module; app.py prepends common.js).
//
// Plays the frames of all arranged views as one video on a canvas, in the tile arrangement
// the viewer had when Preview was clicked. The recording has `srcFps` frames per second
// (30 for the Isaac Sim captures). At a playback rate of n FPS the preview still runs in
// real time but shows n frames per second, picked at regular intervals from the recorded
// ones: output frame k shows recorded frame  start + round(k * srcFps / n)
// (15 FPS -> every 2nd frame, 10 FPS -> every 3rd, 12 FPS -> frames 0, 3, 5, 8, 10, ...).
//
// Play button: click = play / pause, double-click = stop and go back to the start frame.

const GAP = 8;                // canvas pixels between tiles
const PRELOAD_SECONDS = 1.5;  // browser preloads this much of the upcoming playback
const WARM_SECONDS = 4;       // the server decodes this much ahead (at most 128 frames)

const savedFps = new Map();   // dataset folder -> playback FPS

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
  const tileW = data.w;
  const tileH = Math.round((data.w * imgH) / imgW);
  const shownViews = data.tiles.filter((view) => view >= 0);

  const stage = el("div", "mvk-stage", root);          // canvas + view labels on top of it
  const canvas = el("canvas", "mvk-canvas", stage);
  canvas.width = cols * tileW + (cols - 1) * GAP;
  canvas.height = rows * tileH + (rows - 1) * GAP;
  const ctx = canvas.getContext("2d");
  data.tiles.forEach((view, t) => {   // HTML labels stay readable however small the video is drawn
    if (view < 0) return;
    const x = (t % cols) * (tileW + GAP);
    const yBottom = Math.floor(t / cols) * (tileH + GAP) + tileH;
    const tag = el("div", "mvk-tile-label", stage);
    tag.textContent = data.views[view];
    tag.style.left = `calc(${(100 * x) / canvas.width}% + 6px)`;
    tag.style.bottom = `calc(${(100 * (canvas.height - yBottom)) / canvas.height}% + 6px)`;
  });

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
  const status = el("span", "mvk-status mvk-player-status", bar);
  const note = el("div", "mvk-player-note", root);

  const host = root.getRootNode().host;
  const theme = host ? getComputedStyle(host) : null;
  const tileBg = (theme && theme.getPropertyValue("--st-secondary-background-color").trim()) || "#f0f2f6";
  const mutedText = (theme && theme.getPropertyValue("--st-gray-text-color").trim()) || "#808495";
  const font = (theme && theme.getPropertyValue("--st-font").trim()) || "sans-serif";

  const saved = savedFps.get(data.root) ?? parseInt(storageGet(`mvk:fps:${data.root}`), 10);
  const origin = timeline[nearestIndex(timeline, data.start)];
  const p = {
    origin,                 // start frame (from the viewer)
    anchor: origin,         // output frame k shows recorded frame anchor + round(k * src / fps)
    k: 0,                   // next output frame to show
    fps: Number.isInteger(saved) && saved >= 1 && saved <= src ? saved : src,
    playing: false,
    ended: false,
    raf: 0,
    nextDue: 0,
    shown: null,            // recorded frame number on the canvas
    warmAt: 0,
    stillToken: 0,
    setStateValue: null,
    syncTimer: 0,
    observer: null,
  };
  const warm = makeWarmer();

  const url = (view, frame) => `${data.base}/preview/${data.ds}/${view}/${frame}?w=${data.w}`;
  const urlsFor = (frame) => data.tiles.map((view) =>
    (view >= 0 && !missing[view].has(frame) ? url(view, frame) : null));

  function target(k) {      // recorded frame for output frame k, or null past the end
    const f = p.anchor + Math.round((k * src) / p.fps);
    return f > last ? null : timeline[nearestIndex(timeline, f)];
  }

  function ready(frame) {
    return urlsFor(frame).every((u) => u === null || ["ok", "error"].includes(loadState(u)));
  }

  function request(frame) {
    return Promise.all(urlsFor(frame).map((u) => (u ? load(u, "high") : true)));
  }

  function draw(frame) {    // all tiles of one recorded frame, in a single canvas update
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
      } else {
        ctx.fillStyle = mutedText;
        ctx.font = `${Math.max(14, Math.round(tileH * 0.05))}px ${font}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(urls[t] ? `cannot load frame ${frame}` : `no frame ${frame}`, x + tileW / 2, y + tileH / 2);
      }
    });
    p.shown = frame;
    updateStatus();
  }

  function updateStatus() {
    const f = p.shown ?? p.origin;
    const seconds = ((f - p.origin) / src).toFixed(2);
    status.textContent = p.ended ? `frame ${f}  ·  ${seconds} s  ·  end` : `frame ${f}  ·  ${seconds} s`;
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
      warm(`${data.base}/warm/${data.ds}?w=${data.w}&v=${shownViews.join(",")}&f=${upcoming.join(",")}`);
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
    if (!ready(frame)) { request(frame); return; }   // wait for the images rather than skip a frame
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
  const panel = host ? host.closest(".st-key-mvk_preview") : null;
  const fit = () => {
    const room = roomForContent(panel, "--mvk-preview-h", root, stage);
    const width = room === null ? null : Math.max(160, (room * canvas.width) / canvas.height);
    const next = width !== null && width < root.getBoundingClientRect().width ? `${Math.floor(width)}px` : "";
    if (stage.style.width !== next) stage.style.width = next;
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
  setIcon();
  ctx.fillStyle = tileBg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  updateStatus();
  showStill(0);
  return p;
}

export default function (component) {
  const { data, parentElement, setStateValue } = component;
  if (!data || !data.runs || !data.runs.length) return;
  let root = parentElement.querySelector(".mvk-root");
  if (!root) root = el("div", "mvk-root", parentElement);
  const config = JSON.stringify(data);
  if (!root.__player || root.__config !== config) {
    if (root.__player) root.__player.destroy();
    root.__player = buildPlayer(root, data);
    root.__config = config;
  }
  root.__player.setStateValue = setStateValue;
  // Playback stops by itself once the pane is closed (tick() checks root.isConnected).
}
