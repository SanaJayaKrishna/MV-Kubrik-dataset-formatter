// Shared browser code for viewer.js and preview.js. app.py prepends this file to each of them,
// so every component module gets its own copy (own image pool), while the browser's HTTP cache
// is shared.

const PRELOAD_PARALLEL = 4;   // leaves browser connections free for the frame on screen
const POOL_MAX = 800;         // preloaded images kept alive (about 90 KB each)
const RETRY_MS = 5000;        // a failed image is tried again after this long

// ------------------------------------------------------------------ image pool

const pool = new Map();       // url -> {img, state: "loading" | "ok" | "error", time, ready: Promise<boolean>}

function load(url, priority) {
  let entry = pool.get(url);
  if (entry && entry.state === "error" && performance.now() - entry.time > RETRY_MS) {
    pool.delete(url);         // e.g. a frame that was still being written: try again
    entry = null;
  }
  if (entry) {                // LRU touch
    pool.delete(url);
    pool.set(url, entry);
    return entry.ready;
  }
  const img = new Image();
  img.decoding = "async";
  img.fetchPriority = priority;
  entry = { img, state: "loading", time: 0, ready: null };
  const e = entry;
  e.ready = new Promise((resolve) => {
    const ok = () => { e.state = "ok"; resolve(true); };
    img.onload = () => img.decode().then(ok, ok);
    img.onerror = () => { e.state = "error"; e.time = performance.now(); resolve(false); };
  });
  img.src = url;
  pool.set(url, entry);
  while (pool.size > POOL_MAX) pool.delete(pool.keys().next().value);
  return entry.ready;
}

function loadState(url) {     // null (never requested), "loading", "ok" or "error"
  const entry = pool.get(url);
  return entry ? entry.state : null;
}

const preloader = {
  limit: PRELOAD_PARALLEL,    // requests in flight at most (the preview player raises it)
  queue: [],
  active: 0,
  set(urls) {
    this.queue = urls.filter((u) => !pool.has(u));
    this.pump();
  },
  pump() {
    while (this.active < this.limit && this.queue.length) {
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

// Asks the server to decode frames in advance; at most one request in flight, latest one wins.
function makeWarmer() {
  let busy = false;
  let next = null;
  const send = () => {
    const query = next;
    next = null;
    if (!query) return;
    busy = true;
    fetch(query, { cache: "no-store" }).catch(() => {}).finally(() => {
      busy = false;
      send();
    });
  };
  return (query) => {
    next = query;
    if (!busy) send();
  };
}

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

// Height available to `content` inside a panel whose height was fixed with the bottom or
// corner handle (everything below `content` in the panel, such as the controls under the
// grid or a button under the component, must still fit), or null when it is not fixed.
function roomForContent(panel, heightVar, root, content) {
  if (!panel || !root.isConnected) return null;
  if (!getComputedStyle(document.documentElement).getPropertyValue(heightVar).trim()) return null;
  const ps = getComputedStyle(panel);
  const bottom = panel.getBoundingClientRect().bottom - parseFloat(ps.paddingBottom) - parseFloat(ps.borderBottomWidth);
  const box = content.getBoundingClientRect();
  let end = root.getBoundingClientRect().bottom;
  for (const child of panel.children) end = Math.max(end, child.getBoundingClientRect().bottom);
  return bottom - box.top - (end - box.bottom) - 4;
}

// A switch that looks like Streamlit's st.toggle (styles in viewer.css).
function makeToggle(parent, label, checked) {
  const button = el("button", "mvk-toggle", parent);
  button.type = "button";
  button.setAttribute("role", "switch");
  el("span", "mvk-switch", button);
  el("span", "mvk-toggle-label", button).textContent = label;
  button.setChecked = (on) => button.setAttribute("aria-checked", on ? "true" : "false");
  button.setChecked(checked);
  return button;
}
