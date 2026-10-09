// Adds a drag handle to the right edge of the viewer panel (Streamlit components v2 module).
// The width is stored in localStorage; double-click the handle to go back to the default.
// Page CSS (app.py) sizes the panel from --mvk-panel-w and lets the side panel wrap beside or below it.

const KEY = "mvk:panelWidth";
const MIN_WIDTH = 380;

function setWidth(px) {
  const style = document.documentElement.style;
  if (px) style.setProperty("--mvk-panel-w", `${Math.round(px)}px`);
  else style.removeProperty("--mvk-panel-w");
}

export default function ({ data, parentElement }) {
  const host = parentElement.host || parentElement;
  const panel = host.closest(`.st-key-${data.panel}`);
  if (!panel) return;
  const item = panel.parentElement;          // flex item that holds the panel
  const layout = item.parentElement;          // horizontal container (wraps)

  try { const w = parseFloat(localStorage.getItem(KEY)); if (w > 0) setWidth(w); } catch (e) { /* blocked */ }

  let handle = item.querySelector(":scope > .mvk-resize-handle");
  if (!handle) {
    handle = document.createElement("div");
    handle.className = "mvk-resize-handle";
    handle.title = "Drag to resize the viewer, double-click to reset";
    item.appendChild(handle);
  }
  const accent = getComputedStyle(host).getPropertyValue("--st-primary-color").trim();
  if (accent) handle.style.setProperty("--mvk-accent", accent);

  let drag = null;
  const onDown = (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, width: item.getBoundingClientRect().width, max: layout.getBoundingClientRect().width };
    handle.classList.add("mvk-active");
    document.body.classList.add("mvk-resizing");
  };
  const onMove = (e) => {
    if (!drag) return;
    setWidth(Math.max(Math.min(MIN_WIDTH, drag.max), Math.min(drag.max, drag.width + e.clientX - drag.x)));
  };
  const onUp = () => {
    if (!drag) return;
    drag = null;
    handle.classList.remove("mvk-active");
    document.body.classList.remove("mvk-resizing");
    try { localStorage.setItem(KEY, String(item.getBoundingClientRect().width)); } catch (e) { /* blocked */ }
  };
  const onReset = () => {
    setWidth(null);
    try { localStorage.removeItem(KEY); } catch (e) { /* blocked */ }
  };

  handle.addEventListener("pointerdown", onDown);
  handle.addEventListener("pointermove", onMove);
  handle.addEventListener("pointerup", onUp);
  handle.addEventListener("pointercancel", onUp);
  handle.addEventListener("dblclick", onReset);
  return () => {
    handle.removeEventListener("pointerdown", onDown);
    handle.removeEventListener("pointermove", onMove);
    handle.removeEventListener("pointerup", onUp);
    handle.removeEventListener("pointercancel", onUp);
    handle.removeEventListener("dblclick", onReset);
    handle.remove();
  };
}
