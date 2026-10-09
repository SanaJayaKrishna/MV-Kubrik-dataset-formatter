// Synchronized multi-view frame grid (Streamlit components v2 module).
//
// Navigation runs entirely in the browser, so stepping never waits for a
// Streamlit rerun. Images come from the app's /mvk/preview route; the next
// frames are preloaded, the server is asked to decode further ahead, and all
// views of a frame are swapped together once every image is decoded.
//
// Mouse and keys:
//   < > buttons, Left/Right keys    step one frame; hold to keep stepping (Shift: 10)
//   frame box + Enter               jump to the nearest existing frame
//   left-button drag on a view      drop it on another tile to swap the two

const AHEAD = 12;             // frames preloaded by the browser in the direction of travel
const BEHIND = 4;             // ...and behind
const WARM_AHEAD = 40;        // frames the server decodes in advance
const WARM_BEHIND = 8;
const PRELOAD_PARALLEL = 4;   // leaves browser connections free for the frame on screen
const POOL_MAX = 800;         // preloaded images kept alive (about 90 KB each)
const SYNC_DELAY_MS = 400;    // report the frame to Python once navigation pauses
const HOLD_DELAY_MS = 350;    // press and hold: repeating starts after this...
const HOLD_INTERVAL_MS = 33;  // ...then about 30 steps per second, each shown before the next
const DRAG_THRESHOLD_PX = 6;  // a press on a view becomes a drag after moving this far

const savedFrame = new Map(); // dataset folder -> current frame number (survives reruns)
const savedSlots = new Map(); // dataset folder -> tile arrangement as view names (null = empty tile)

// ------------------------------------------------------------------ image pool

const pool = new Map();       // url -> {img, ready: Promise<boolean>}

function load(url, priority) {
  let entry = pool.get(url);
  if (entry) {                // LRU touch
    pool.delete(url);
    pool.set(url, entry);
    return entry.ready;
  }
  const img = new Image();
  img.decoding = "async";
  img.fetchPriority = priority;
  const ready = new Promise((resolve) => {
    img.onload = () => img.decode().then(() => resolve(true), () => resolve(true));
    img.onerror = () => {
      pool.delete(url);       // retry next time (e.g. a frame that was still being written)
      resolve(false);
    };
  });
  img.src = url;
  entry = { img, ready };
  pool.set(url, entry);
  while (pool.size > POOL_MAX) pool.delete(pool.keys().next().value);
  return ready;
}

const preloader = {
  queue: [],
  active: 0,
  set(urls) {
    this.queue = urls.filter((u) => !pool.has(u));
    this.pump();
  },
  pump() {
    while (this.active < PRELOAD_PARALLEL && this.queue.length) {
      const url = this.queue.shift();
      if (pool.has(url)) continue;
      this.active++;
      load(url, "low").finally(() => {
        this.active--;
        this.pump();
      });
    }
  },
};

// ------------------------------------------------------------------ helpers

function expandRuns(runs) {
  const out = [];
  for (const [a, b] of runs) for (let f = a; f <= b; f++) out.push(f);
  return out;
}

function nearestIndex(timeline, frame) {
  let lo = 0, hi = timeline.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (timeline[mid] < frame) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && frame - timeline[lo - 1] <= timeline[lo] - frame) return lo - 1;
  return lo;
}

function el(tag, cls, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (parent) parent.appendChild(e);
  return e;
}

function storageGet(key) {
  try { return localStorage.getItem(key); } catch (e) { return null; }
}

function storageSet(key, value) {
  try { localStorage.setItem(key, value); } catch (e) { /* storage blocked */ }
}

function restoreFrame(root) {        // frame shown before a browser reload, if any
  try {
    const s = sessionStorage.getItem(`mvk:${root}`);
    return s === null ? -Infinity : parseInt(s, 10) || 0;
  } catch (e) {
    return -Infinity;
  }
}

function restoreSlots(root) {        // saved tile arrangement (view names), if any
  if (savedSlots.has(root)) return savedSlots.get(root);
  try { return JSON.parse(storageGet(`mvk:layout:${root}`)) || null; } catch (e) { return null; }
}

// Tile arrangement: slots[t] = view index shown on tile t, or -1 for an empty tile.
// Every view appears exactly once; views beyond the visible tiles are hidden.
function normalizeSlots(slots, nViews, nTiles) {
  const seen = new Set();
  const out = slots.map((s) => {
    if (Number.isInteger(s) && s >= 0 && s < nViews && !seen.has(s)) { seen.add(s); return s; }
    return -1;
  });
  for (let view = 0; view < nViews; view++) {
    if (seen.has(view)) continue;
    const hole = out.indexOf(-1);
    if (hole >= 0) out[hole] = view; else out.push(view);
  }
  const length = Math.max(nViews, nTiles);
  while (out.length < length) out.push(-1);
  while (out.length > length) {        // drop empty tiles so hidden views move up into view
    const hole = out.lastIndexOf(-1);
    if (hole < 0) break;
    out.splice(hole, 1);
  }
  return out;
}

const CHEVRON = (d) =>
  `<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="${d}" fill="none" ` +
  `stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

// ------------------------------------------------------------------ viewer

function buildViewer(root, data) {
  root.replaceChildren();
  const timeline = expandRuns(data.runs);
  const n = timeline.length;
  const nViews = data.views.length;
  const missing = data.views.map((_, view) => new Set(expandRuns(data.missing[String(view)] || [])));
  const [imgW, imgH] = data.size;
  const byName = new Map(data.views.map((name, i) => [name, i]));
  const saved = restoreSlots(data.root);
  let slots = normalizeSlots(
    saved ? saved.map((name) => (byName.has(name) ? byName.get(name) : -1)) : data.views.map((_, i) => i),
    nViews, data.cells);

  const grid = el("div", "mvk-grid", root);
  grid.style.gridTemplateColumns = `repeat(${data.cols}, minmax(0, 1fr))`;
  const tiles = [];
  for (let t = 0; t < data.cells; t++) {
    const fig = el("figure", "mvk-cell", grid);
    const box = el("div", "mvk-frame", fig);
    box.style.aspectRatio = `${imgW} / ${imgH}`;
    const img = el("img", "mvk-img", box);
    img.alt = "";
    img.draggable = false;
    const note = el("div", "mvk-note", box);
    const cap = el("figcaption", "mvk-cap", fig);
    tiles.push({ fig, img, note, cap });
  }

  const nav = el("div", "mvk-nav", root);
  const prev = el("button", "mvk-btn", nav);
  prev.innerHTML = CHEVRON("M15 5l-7 7 7 7");
  prev.title = "Previous frame (Left arrow). Hold to keep stepping, Shift for 10";
  prev.setAttribute("aria-label", "Previous frame");
  const label = el("label", "mvk-label", nav);
  label.textContent = "Frame";
  const input = el("input", "mvk-input", label);
  input.type = "number";
  input.step = "1";
  input.min = String(timeline[0]);
  input.max = String(timeline[n - 1]);
  input.title = "Type a frame number and press Enter";
  const next = el("button", "mvk-btn", nav);
  next.innerHTML = CHEVRON("M9 5l7 7-7 7");
  next.title = "Next frame (Right arrow). Hold to keep stepping, Shift for 10";
  next.setAttribute("aria-label", "Next frame");
  const status = el("span", "mvk-status", nav);

  const v = {
    idx: nearestIndex(timeline, savedFrame.get(data.root) ?? restoreFrame(data.root)),
    seq: 0,                 // number of the latest show() request
    shownSeq: 0,            // request number of the frame on screen
    last: null,             // what is on screen: {frame, urls: Map(view -> url|null), ok: Map(view -> bool)}
    hold: null,             // press-and-hold state
    press: null,            // left button down on a view, not yet moved far enough to drag
    drag: null,             // drag state while rearranging tiles
    observer: null,         // ResizeObserver that refits the grid to the panel height
    setStateValue: null,
    syncTimer: 0,
    warmBusy: false,
    warmNext: null,
  };

  const url = (view, frame) => `${data.base}/preview/${data.ds}/${view}/${frame}?w=${data.w}`;
  const has = (view, frame) => !missing[view].has(frame);
  const visibleViews = () => slots.slice(0, data.cells).filter((view) => view >= 0);
  const fileName = (view, frame) => {
    const p = data.names[view];
    return p ? `${p[0]}${String(frame).padStart(p[1], "0")}${p[2]}` : "";
  };

  function caption(view, frame, state) {
    const parts = [data.views[view], `frame ${frame}`];
    if (state === "missing") parts.push("missing");
    else {
      const name = fileName(view, frame);
      if (name) parts.push(name);
      if (state === "error") parts.push("unreadable");
    }
    return parts.join("  ·  ");
  }

  function updateNav() {
    const frame = timeline[v.idx];
    if (root.getRootNode().activeElement !== input) input.value = String(frame);  // not while typing
    prev.disabled = v.idx <= 0;
    next.disabled = v.idx >= n - 1;
    status.textContent = `${v.idx + 1} of ${n}`;
  }

  function paint() {                       // draw v.last into the tiles, all in the same task
    const shown = v.last;
    queueMicrotask(() => tiles.forEach((tile) => { tile.cap.title = tile.cap.textContent.trim(); }));
    tiles.forEach((tile, t) => {
      const view = slots[t];
      tile.fig.classList.toggle("mvk-empty", view < 0);
      if (view < 0) {
        tile.img.hidden = true;
        tile.note.hidden = false;
        tile.note.textContent = "empty";
        tile.cap.textContent = " ";
        return;
      }
      const u = shown && shown.urls.get(view);
      if (shown && u && shown.ok.get(view)) {
        tile.img.src = u;
        tile.img.hidden = false;
        tile.note.hidden = true;
        tile.cap.textContent = caption(view, shown.frame, "ok");
      } else {
        tile.img.hidden = true;
        tile.note.hidden = false;
        const frame = shown ? shown.frame : timeline[v.idx];
        tile.note.textContent = !shown ? "" : u === null ? `no frame ${frame}` : `cannot load frame ${frame}`;
        tile.cap.textContent = shown ? caption(view, frame, u === null ? "missing" : "error") : data.views[view];
      }
    });
  }

  async function show(i) {
    const seq = ++v.seq;
    const frame = timeline[i];
    const views = visibleViews();
    const urls = new Map(views.map((view) => [view, has(view, frame) ? url(view, frame) : null]));
    const slow = setTimeout(() => status.classList.add("mvk-loading"), 150);
    const ok = await Promise.all([...urls.values()].map((u) => (u ? load(u, "high") : false)));
    clearTimeout(slow);
    if (seq < v.shownSeq) return;          // a newer frame is already on screen
    v.shownSeq = seq;
    if (seq === v.seq) status.classList.remove("mvk-loading");
    v.last = { frame, urls, ok: new Map(views.map((view, k) => [view, ok[k]])) };
    paint();
  }

  function warm(frames) {                  // ask the server to decode upcoming frames
    v.warmNext = `${data.base}/warm/${data.ds}?w=${data.w}&v=${visibleViews().join(",")}&f=${frames.join(",")}`;
    if (!v.warmBusy) sendWarm();
  }

  function sendWarm() {                    // at most one request in flight, latest one wins
    const query = v.warmNext;
    v.warmNext = null;
    if (!query) return;
    v.warmBusy = true;
    fetch(query, { cache: "no-store" }).catch(() => {}).finally(() => {
      v.warmBusy = false;
      sendWarm();
    });
  }

  function prefetch(i, delta) {           // delta: last step (+-1, or +-10 with Shift)
    const ranked = [];
    for (let k = 1; k <= WARM_AHEAD; k++) ranked.push([k, i + delta * k]);
    for (let k = 1; k <= WARM_BEHIND; k++) ranked.push([2.5 * k, i - delta * k]);
    ranked.sort((a, b) => a[0] - b[0]);
    const order = ranked.filter((r) => r[1] >= 0 && r[1] < n);
    warm(order.map((r) => timeline[r[1]]));
    const near = order.filter((r) => r[0] <= AHEAD).map((r) => r[1]);   // the nearest steps both ways
    const views = visibleViews();
    const urls = [];
    for (const j of near) for (const view of views) {
      if (has(view, timeline[j])) urls.push(url(view, timeline[j]));
    }
    preloader.set(urls);
  }

  function go(i, delta) {
    i = Math.max(0, Math.min(n - 1, i));
    const changed = i !== v.idx;
    v.idx = i;
    savedFrame.set(data.root, timeline[i]);
    updateNav();
    show(i);
    prefetch(i, delta || 1);
    if (changed) scheduleSync();
  }

  function scheduleSync() {
    clearTimeout(v.syncTimer);
    v.syncTimer = setTimeout(() => {
      try { sessionStorage.setItem(`mvk:${data.root}`, String(timeline[v.idx])); } catch (e) { /* blocked */ }
      if (v.setStateValue) v.setStateValue("frame", timeline[v.idx]);
    }, SYNC_DELAY_MS);
  }

  function jumpToInput() {
    const value = parseInt(input.value, 10);
    if (Number.isNaN(value)) { updateNav(); return; }
    const target = nearestIndex(timeline, value);
    go(target, target >= v.idx ? 1 : -1);
    input.value = String(timeline[target]);
  }

  v.step = (delta) => go(v.idx + delta, delta);

  // ---- press and hold (buttons and arrow keys share one paced repeater)

  function holdLoop(now) {
    const h = v.hold;
    if (!h) return;
    if (now >= h.next && v.shownSeq === v.seq) {   // only step once the previous frame is on screen
      if ((h.delta > 0 && v.idx >= n - 1) || (h.delta < 0 && v.idx <= 0)) { v.stopHold(); return; }
      v.step(h.delta);
      h.next = now + HOLD_INTERVAL_MS - 4;
    }
    h.raf = requestAnimationFrame(holdLoop);
  }

  v.startHold = (delta, key) => {
    v.stopHold();
    v.step(delta);
    v.hold = { delta, key, next: performance.now() + HOLD_DELAY_MS, raf: requestAnimationFrame(holdLoop) };
  };

  v.stopHold = () => {
    if (!v.hold) return;
    cancelAnimationFrame(v.hold.raf);
    v.hold = null;
  };

  for (const [button, dir] of [[prev, -1], [next, 1]]) {
    button.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      button.setPointerCapture(e.pointerId);
      v.startHold(e.shiftKey ? 10 * dir : dir, null);
    });
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
      button.addEventListener(type, () => { if (v.hold && v.hold.key === null) v.stopHold(); });
    }
    button.addEventListener("click", (e) => { if (e.detail === 0) v.step(dir); });  // Enter/Space when focused
  }

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); jumpToInput(); input.blur(); }
    else if (e.key === "Escape") { updateNav(); input.blur(); }
  });
  input.addEventListener("change", jumpToInput);   // spinner arrows, or leaving the box after typing
  input.addEventListener("blur", () => updateNav());

  // ---- left-button drag: rearrange the tiles

  function tileAt(x, y) {
    return tiles.findIndex(({ fig }) => {
      const r = fig.getBoundingClientRect();
      return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
    });
  }

  function startDrag(t, x, y) {
    const tile = tiles[t];
    const r = tile.fig.getBoundingClientRect();
    const scale = Math.min(1, 260 / r.width);
    const ghost = el("div", "mvk-ghost", root);
    ghost.style.width = `${r.width * scale}px`;
    const thumb = el("div", "mvk-ghost-frame", ghost);
    thumb.style.aspectRatio = `${imgW} / ${imgH}`;
    if (!tile.img.hidden) { const gi = el("img", "", thumb); gi.src = tile.img.src; gi.draggable = false; }
    el("div", "mvk-ghost-label", ghost).textContent = data.views[slots[t]];
    v.drag = { from: t, over: -1, ghost, dx: (x - r.left) * scale, dy: (y - r.top) * scale };
    tile.fig.classList.add("mvk-drag-source");
    grid.classList.add("mvk-dragging");
  }

  function moveDrag(x, y) {
    const d = v.drag;
    d.ghost.style.transform = `translate(${x - d.dx}px, ${y - d.dy}px)`;
    const over = tileAt(x, y);
    if (over === d.over) return;
    if (d.over >= 0) tiles[d.over].fig.classList.remove("mvk-drop-target");
    d.over = over;
    if (over >= 0 && over !== d.from) tiles[over].fig.classList.add("mvk-drop-target");
  }

  v.endDrag = (commit) => {
    v.press = null;
    const d = v.drag;
    if (!d) return;
    v.drag = null;
    d.ghost.remove();
    grid.classList.remove("mvk-dragging");
    tiles[d.from].fig.classList.remove("mvk-drag-source");
    if (d.over >= 0) tiles[d.over].fig.classList.remove("mvk-drop-target");
    if (!commit || d.over < 0 || d.over === d.from) return;
    [slots[d.from], slots[d.over]] = [slots[d.over], slots[d.from]];
    const names = slots.map((view) => (view >= 0 ? data.views[view] : null));
    savedSlots.set(data.root, names);
    storageSet(`mvk:layout:${data.root}`, JSON.stringify(names));
    if (v.setStateValue) v.setStateValue("layout", names);
    paint();                               // instant: both views are already loaded
    show(v.idx);                           // a view that just became visible loads its image
  };

  tiles.forEach((tile, t) => {
    tile.fig.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || slots[t] < 0 || v.drag) return;
      e.preventDefault();                  // no text selection or native image drag
      tile.fig.setPointerCapture(e.pointerId);
      v.press = { tile: t, x: e.clientX, y: e.clientY };
    });
    tile.fig.addEventListener("pointermove", (e) => {
      const p = v.press;
      if (p && !v.drag) {
        if (Math.hypot(e.clientX - p.x, e.clientY - p.y) < DRAG_THRESHOLD_PX) return;
        startDrag(p.tile, p.x, p.y);
      }
      if (v.drag) moveDrag(e.clientX, e.clientY);
    });
    tile.fig.addEventListener("pointerup", (e) => { if (e.button === 0) v.endDrag(true); });
    tile.fig.addEventListener("pointercancel", () => v.endDrag(false));
  });

  // ---- fit the grid into the panel when the panel has a fixed height (bottom/corner handle)

  function fit() {
    const panel = v.panel;
    const fixed = panel && getComputedStyle(document.documentElement).getPropertyValue("--mvk-panel-h").trim();
    if (!fixed || !root.isConnected) {
      if (grid.style.width) grid.style.width = "";
      return;
    }
    const ps = getComputedStyle(panel);
    const bottom = panel.getBoundingClientRect().bottom - parseFloat(ps.paddingBottom) - parseFloat(ps.borderBottomWidth);
    const navSpace = nav.getBoundingClientRect().height + parseFloat(getComputedStyle(nav).marginTop);
    const gs = getComputedStyle(grid);
    const rowGap = parseFloat(gs.rowGap) || 0;
    const colGap = parseFloat(gs.columnGap) || 0;
    const cap = tiles[0].cap;
    const capSpace = cap.getBoundingClientRect().height + parseFloat(getComputedStyle(cap).marginTop);
    const rows = Math.ceil(data.cells / data.cols);
    const room = bottom - grid.getBoundingClientRect().top - navSpace - 4;
    const tileH = (room - (rows - 1) * rowGap) / rows - capSpace;
    const width = Math.max(data.cols * 48, data.cols * (tileH * imgW) / imgH + (data.cols - 1) * colGap);
    const next = width < root.getBoundingClientRect().width ? `${Math.floor(width)}px` : "";
    if (grid.style.width !== next) grid.style.width = next;
  }

  let fitQueued = false;
  v.queueFit = () => {
    if (fitQueued) return;
    fitQueued = true;
    requestAnimationFrame(() => { fitQueued = false; fit(); });
  };
  const host = root.getRootNode().host;
  v.panel = host ? host.closest(".st-key-mvk_panel") : null;
  v.observer = new ResizeObserver(v.queueFit);
  if (v.panel) v.observer.observe(v.panel);
  v.observer.observe(root);
  v.destroy = () => {
    v.stopHold();
    v.endDrag(false);
    v.observer.disconnect();
  };

  paint();
  go(v.idx, 1);
  return v;
}

function isTyping(e) {
  const t = e.composedPath()[0];
  return !!t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
}

export default function (component) {
  const { data, parentElement, setStateValue } = component;
  if (!data || !data.runs || !data.runs.length) return;

  let root = parentElement.querySelector(".mvk-root");
  if (!root) root = el("div", "mvk-root", parentElement);
  const config = JSON.stringify(data);
  if (!root.__viewer || root.__config !== config) {
    if (root.__viewer) root.__viewer.destroy();
    root.__viewer = buildViewer(root, data);
    root.__config = config;
  }
  const viewer = root.__viewer;
  viewer.setStateValue = setStateValue;

  const onKeyDown = (e) => {
    if (e.key === "Escape" && viewer.drag) { viewer.endDrag(false); return; }
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || isTyping(e)) return;
    if (!root.isConnected || document.querySelector('[role="dialog"]')) return;
    e.preventDefault();
    if (e.repeat) return;                  // holding the key is handled by the paced repeater
    const dir = e.key === "ArrowRight" ? 1 : -1;
    viewer.startHold(e.shiftKey ? 10 * dir : dir, e.key);
  };
  const onKeyUp = (e) => { if (viewer.hold && e.key === viewer.hold.key) viewer.stopHold(); };
  const onBlur = () => viewer.stopHold();
  document.addEventListener("keydown", onKeyDown);
  document.addEventListener("keyup", onKeyUp);
  window.addEventListener("blur", onBlur);
  return () => {
    document.removeEventListener("keydown", onKeyDown);
    document.removeEventListener("keyup", onKeyUp);
    window.removeEventListener("blur", onBlur);
    viewer.stopHold();
  };
}
