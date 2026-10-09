// Resize handles for the viewer panel (Streamlit components v2 module).
//   right edge: width    bottom edge: height    corner: both
// Sizes are kept in CSS variables (--mvk-panel-w / --mvk-panel-h, read by the page CSS in
// app.py) and in localStorage; double-click a handle to go back to the default size.
// With a fixed height the grid (viewer.js) scales itself to fit inside the panel.

const AXES = {
  w: { prop: "--mvk-panel-w", key: "mvk:panelWidth", min: 380 },
  h: { prop: "--mvk-panel-h", key: "mvk:panelHeight", min: 300 },
};
const HANDLES = [
  { cls: "mvk-resize-x", axes: ["w"], title: "Drag to change the width, double-click to reset" },
  { cls: "mvk-resize-y", axes: ["h"], title: "Drag to change the height, double-click to reset" },
  { cls: "mvk-resize-xy", axes: ["w", "h"], title: "Drag to resize, double-click to reset" },
];

function setSize(axis, px) {
  const style = document.documentElement.style;
  if (px) style.setProperty(AXES[axis].prop, `${Math.round(px)}px`);
  else style.removeProperty(AXES[axis].prop);
}

function store(axis, px) {
  try {
    if (px) localStorage.setItem(AXES[axis].key, String(Math.round(px)));
    else localStorage.removeItem(AXES[axis].key);
  } catch (e) { /* storage blocked */ }
}

export default function ({ data, parentElement }) {
  const host = parentElement.host || parentElement;
  const panel = host.closest(`.st-key-${data.panel}`);
  if (!panel) return;
  const item = panel.parentElement;          // flex item that holds the panel
  const layout = item.parentElement;          // horizontal container (wraps)

  for (const axis of Object.keys(AXES)) {
    try { const px = parseFloat(localStorage.getItem(AXES[axis].key)); if (px > 0) setSize(axis, px); } catch (e) { /* blocked */ }
  }
  const accent = getComputedStyle(host).getPropertyValue("--st-primary-color").trim();
  const cleanups = [];

  for (const spec of HANDLES) {
    let handle = item.querySelector(`:scope > .${spec.cls}`);
    if (!handle) {
      handle = document.createElement("div");
      handle.className = `mvk-resize ${spec.cls}`;
      handle.title = spec.title;
      item.appendChild(handle);
    }
    if (accent) handle.style.setProperty("--mvk-accent", accent);

    let drag = null;
    const onDown = (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      handle.setPointerCapture(e.pointerId);
      const r = item.getBoundingClientRect();
      drag = { x: e.clientX, y: e.clientY, w: r.width, h: r.height, maxW: layout.getBoundingClientRect().width };
      handle.classList.add("mvk-active");
      document.body.classList.add("mvk-resizing", `${spec.cls}-active`);
    };
    const onMove = (e) => {
      if (!drag) return;
      if (spec.axes.includes("w")) {
        setSize("w", Math.max(Math.min(AXES.w.min, drag.maxW), Math.min(drag.maxW, drag.w + e.clientX - drag.x)));
      }
      if (spec.axes.includes("h")) setSize("h", Math.max(AXES.h.min, drag.h + e.clientY - drag.y));
    };
    const onUp = () => {
      if (!drag) return;
      drag = null;
      handle.classList.remove("mvk-active");
      document.body.classList.remove("mvk-resizing", `${spec.cls}-active`);
      const r = item.getBoundingClientRect();
      if (spec.axes.includes("w")) store("w", r.width);
      if (spec.axes.includes("h")) store("h", r.height);
    };
    const onReset = () => {
      for (const axis of spec.axes) { setSize(axis, null); store(axis, null); }
    };
    const events = [["pointerdown", onDown], ["pointermove", onMove], ["pointerup", onUp],
                    ["pointercancel", onUp], ["dblclick", onReset]];
    for (const [type, fn] of events) handle.addEventListener(type, fn);
    cleanups.push(() => {
      for (const [type, fn] of events) handle.removeEventListener(type, fn);
      handle.remove();
    });
  }
  return () => cleanups.forEach((fn) => fn());
}
