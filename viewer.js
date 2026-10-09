// Synchronized multi-view frame grid (Streamlit components v2 module).
//
// Navigation runs entirely in the browser, so stepping never waits for a
// Streamlit rerun. Images come from the app's /mvk/preview route; the next
// frames are preloaded, the server is asked to decode further ahead, and all
// views of a frame are swapped together once every image is decoded.

const AHEAD = 12;           // frames preloaded by the browser in the direction of travel
const BEHIND = 4;           // ...and behind
const WARM_AHEAD = 40;      // frames the server decodes in advance
const WARM_BEHIND = 8;
const PRELOAD_PARALLEL = 6;
const POOL_MAX = 800;       // preloaded images kept alive (about 90 KB each)
const SYNC_DELAY_MS = 400;  // report the frame to Python once navigation pauses

const savedFrame = new Map();   // dataset folder -> current frame number, survives reruns

// ------------------------------------------------------------------ image pool

const pool = new Map();     // url -> {img, ready: Promise<boolean>}

function load(url, priority) {
  let entry = pool.get(url);
  if (entry) {               // LRU touch
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
      pool.delete(url);      // retry next time (e.g. a frame that was still being written)
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

const CHEVRON = (d) =>
  `<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="${d}" fill="none" ` +
  `stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

// ------------------------------------------------------------------ viewer

function buildViewer(root, data) {
  root.replaceChildren();
  const timeline = expandRuns(data.runs);
  const n = timeline.length;
  const missing = data.views.map((_, v) => new Set(expandRuns(data.missing[String(v)] || [])));
  const [imgW, imgH] = data.size;

  const grid = el("div", "mvk-grid", root);
  grid.style.gridTemplateColumns = `repeat(${data.cols}, minmax(0, 1fr))`;
  const cells = data.views.map(() => {
    const fig = el("figure", "mvk-cell", grid);
    const box = el("div", "mvk-frame", fig);
    box.style.aspectRatio = `${imgW} / ${imgH}`;
    const img = el("img", "mvk-img", box);
    img.alt = "";
    const note = el("div", "mvk-note", box);
    const cap = el("figcaption", "mvk-cap", fig);
    return { img, note, cap };
  });

  const nav = el("div", "mvk-nav", root);
  const prev = el("button", "mvk-btn", nav);
  prev.innerHTML = CHEVRON("M15 5l-7 7 7 7");
  prev.title = "Previous frame (Left arrow, Shift for 10)";
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
  next.title = "Next frame (Right arrow, Shift for 10)";
  next.setAttribute("aria-label", "Next frame");
  const status = el("span", "mvk-status", nav);

  const v = {
    ds: data.ds,
    idx: nearestIndex(timeline, savedFrame.get(data.root) ?? restoreFrame(data.root)),
    seq: 0,
    shownSeq: 0,
    setStateValue: null,
    syncTimer: 0,
    warmBusy: false,
    warmNext: null,
  };

  const url = (view, frame) =>
    `${data.base}/preview/${data.ds}/${data.viewIdx[view]}/${frame}?w=${data.w}`;
  const has = (view, frame) => !missing[view].has(frame);
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

  async function show(i) {
    const seq = ++v.seq;
    const frame = timeline[i];
    const urls = data.views.map((_, view) => (has(view, frame) ? url(view, frame) : null));
    const slow = setTimeout(() => status.classList.add("mvk-loading"), 150);
    const ok = await Promise.all(urls.map((u) => (u ? load(u, "high") : false)));
    clearTimeout(slow);
    if (seq < v.shownSeq) return;          // a newer frame is already on screen
    v.shownSeq = seq;
    if (seq === v.seq) status.classList.remove("mvk-loading");
    cells.forEach((c, view) => {           // swap every view in the same task
      if (urls[view] && ok[view]) {
        c.img.src = urls[view];
        c.img.hidden = false;
        c.note.hidden = true;
        c.cap.textContent = caption(view, frame, "ok");
      } else {
        c.img.hidden = true;
        c.note.hidden = false;
        c.note.textContent = urls[view] ? `cannot load frame ${frame}` : `no frame ${frame}`;
        c.cap.textContent = caption(view, frame, urls[view] ? "error" : "missing");
      }
    });
  }

  function warm(frames) {                  // ask the server to decode upcoming frames
    v.warmNext = `${data.base}/warm/${data.ds}?w=${data.w}&v=${data.viewIdx.join(",")}&f=${frames.join(",")}`;
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

  function prefetch(i, dir) {
    const ranked = [];
    for (let k = 1; k <= WARM_AHEAD; k++) ranked.push([k, i + dir * k]);
    for (let k = 1; k <= WARM_BEHIND; k++) ranked.push([2.5 * k, i - dir * k]);
    ranked.sort((a, b) => a[0] - b[0]);
    const order = ranked.map((r) => r[1]).filter((j) => j >= 0 && j < n);
    warm(order.map((j) => timeline[j]));
    const near = order.filter((j) => Math.abs(j - i) <= (Math.sign(j - i) === dir ? AHEAD : BEHIND));
    const urls = [];
    for (const j of near) data.views.forEach((_, view) => {
      if (has(view, timeline[j])) urls.push(url(view, timeline[j]));
    });
    preloader.set(urls);
  }

  function go(i, dir) {
    i = Math.max(0, Math.min(n - 1, i));
    const changed = i !== v.idx;
    v.idx = i;
    savedFrame.set(data.root, timeline[i]);
    updateNav();
    show(i);
    prefetch(i, dir || 1);
    if (changed) scheduleSync();
  }

  function scheduleSync() {
    clearTimeout(v.syncTimer);
    v.syncTimer = setTimeout(() => {
      try { sessionStorage.setItem(`mvk:${data.root}`, String(timeline[v.idx])); } catch (e) { /* storage blocked */ }
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

  v.step = (delta) => go(v.idx + delta, Math.sign(delta));
  prev.addEventListener("click", () => v.step(-1));
  next.addEventListener("click", () => v.step(1));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); jumpToInput(); input.blur(); }
    else if (e.key === "Escape") { updateNav(); input.blur(); }
  });
  input.addEventListener("change", jumpToInput);   // spinner arrows, or leaving the box after typing
  input.addEventListener("blur", () => updateNav());

  go(v.idx, 1);
  return v;
}

function restoreFrame(root) {       // frame shown before a browser reload, if any
  try {
    const s = sessionStorage.getItem(`mvk:${root}`);
    return s === null ? -Infinity : parseInt(s, 10) || 0;
  } catch (e) {
    return -Infinity;
  }
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
    root.__viewer = buildViewer(root, data);
    root.__config = config;
  }
  const viewer = root.__viewer;
  viewer.setStateValue = setStateValue;

  const onKey = (e) => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || isTyping(e)) return;
    if (!root.isConnected || document.querySelector('[role="dialog"]')) return;
    if (e.key === "ArrowRight") { e.preventDefault(); viewer.step(e.shiftKey ? 10 : 1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); viewer.step(e.shiftKey ? -10 : -1); }
  };
  document.addEventListener("keydown", onKey);
  return () => document.removeEventListener("keydown", onKey);
}
