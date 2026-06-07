"""Build a static HTML explorer for an OME-Zarr output directory.

Walks ``out_dir`` for ``*.ome.zarr`` directories, reads each one's
``.zattrs`` (the ``cv8000`` namespace + ``multiscales``/``omero``), and
emits ``out_dir/index.html``: a single-page-app that renders a
microtiter-plate map (auto-detected 96 / 384), filters a dataset list
by clicked well, and shows one shared Vizarr CDN iframe for the
selected dataset. Designed to scale to thousands of fieldstacks without
spawning thousands of iframes.

If ``{stem}.analysis.html`` exists next to a ``{stem}.ome.zarr``, it is
loaded into a second iframe under the viewer when that dataset is
selected.

Usage
-----
::

    python experiments/build_zarr_explorer.py /path/to/out_dir

Then serve the directory and open the explorer::

    python -m http.server 8000 --directory /path/to/out_dir
    # open http://localhost:8000/

Vizarr is loaded from the public CDN
(``https://hms-dbmi.github.io/vizarr/``) and fetches the zarr chunks
via same-origin relative URLs — no CORS configuration needed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import typer

app = typer.Typer(
    name="build-zarr-explorer",
    help="Generate index.html for an OME-Zarr output directory.",
)

VIZARR_BASE = "https://hms-dbmi.github.io/vizarr/"

WELL_RE = re.compile(r"^([A-Pa-p])(\d{1,2})$")


def _read_zattrs(zarr_dir: Path) -> dict:
    zattrs_path = zarr_dir / ".zattrs"
    if not zattrs_path.is_file():
        return {}
    try:
        return json.loads(zattrs_path.read_text())
    except json.JSONDecodeError:
        return {}


def _read_zarray_shape(zarr_dir: Path) -> tuple[list[int], str]:
    z0 = zarr_dir / "0" / ".zarray"
    if not z0.is_file():
        return [], ""
    try:
        zarray = json.loads(z0.read_text())
        return list(zarray.get("shape", []) or []), str(zarray.get("dtype", "") or "")
    except json.JSONDecodeError:
        return [], ""


def _parse_well(well_id: str) -> tuple[str, int] | None:
    """Split a well id like ``A01`` or ``P24`` into (row letter, col int)."""
    if not well_id:
        return None
    m = WELL_RE.match(well_id.strip())
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2))


def _summarize(zarr_dir: Path, out_dir: Path) -> dict:
    attrs = _read_zattrs(zarr_dir)
    cv = attrs.get("cv8000", {}) or {}
    multiscales = attrs.get("multiscales") or [{}]
    ms0 = multiscales[0] if multiscales else {}
    axes = [a.get("name", "?") for a in ms0.get("axes", [])]
    n_levels = len(ms0.get("datasets", []))

    shape, dtype = _read_zarray_shape(zarr_dir)

    omero = attrs.get("omero", {}) or {}
    channels = omero.get("channels", []) or []

    rel_path = zarr_dir.relative_to(out_dir).as_posix()
    stem = (
        zarr_dir.name[: -len(".ome.zarr")]
        if zarr_dir.name.endswith(".ome.zarr")
        else zarr_dir.stem
    )
    analysis_html = zarr_dir.parent / f"{stem}.analysis.html"
    analysis_rel = (
        analysis_html.relative_to(out_dir).as_posix()
        if analysis_html.is_file()
        else None
    )

    return {
        "rel_path": rel_path,
        "stem": stem,
        "well": str(cv.get("WellID") or "unknown"),
        "action": str(cv.get("Action") or ""),
        "action_index": cv.get("ActionIndex"),
        "field_index": cv.get("FieldIndex"),
        "partial_tile_index": cv.get("PartialTileIndex"),
        "acquisition_index": cv.get("AcquisitionIndex"),
        "z_mode": cv.get("ZMode"),
        "stitched": bool(cv.get("Stitched")) if "Stitched" in cv else None,
        "tile_grid": [cv.get("TileGridY"), cv.get("TileGridX")] if cv.get("TileGridX") else None,
        "axes": axes,
        "shape": shape,
        "dtype": dtype,
        "n_levels": n_levels,
        "n_channels": len(channels),
        "channel_names": [c.get("label", f"Ch{i}") for i, c in enumerate(channels)],
        "analysis_rel": analysis_rel,
        "framerate_hz": cv.get("FramerateHz"),
    }


def _detect_plate(entries: list[dict]) -> dict:
    """Decide rows/cols of the plate from the WellIDs we saw."""
    rows: set[str] = set()
    cols: set[int] = set()
    for e in entries:
        parsed = _parse_well(e["well"])
        if parsed is None:
            continue
        rows.add(parsed[0])
        cols.add(parsed[1])

    if not rows:
        return {"rows": [], "cols": [], "n_rows": 0, "n_cols": 0, "label": "no plate"}

    max_row_idx = max(ord(r) - ord("A") for r in rows)
    max_col = max(cols)

    # snap to the smallest standard plate ≥ what we observed
    if max_row_idx < 8 and max_col <= 12:
        n_rows, n_cols, label = 8, 12, "96-well"
    elif max_row_idx < 16 and max_col <= 24:
        n_rows, n_cols, label = 16, 24, "384-well"
    elif max_row_idx < 32 and max_col <= 48:
        n_rows, n_cols, label = 32, 48, "1536-well"
    else:
        n_rows = max_row_idx + 1
        n_cols = max_col
        label = f"{n_rows}×{n_cols} (custom)"

    row_letters = [chr(ord("A") + i) for i in range(n_rows)]
    col_numbers = list(range(1, n_cols + 1))
    return {
        "rows": row_letters,
        "cols": col_numbers,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "label": label,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OME-Zarr explorer — __TITLE__</title>
<style>
  :root { color-scheme: light dark; --accent: #3b82f6; --accent-soft: #dbeafe; }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body { font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
         display: grid; grid-template-rows: auto 1fr;
         grid-template-columns: 1fr; min-height: 100vh; }
  header { padding: 0.6rem 1rem; border-bottom: 1px solid #ccc;
           display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; }
  header h1 { margin: 0; font-size: 1.05rem; }
  header .summary { color: #666; font-size: 0.85rem; }
  header .legend { font-size: 0.75rem; color: #666; display: flex; gap: 0.5rem;
                   align-items: center; flex-wrap: wrap; }
  header .legend .swatch { display: inline-block; width: 0.9rem; height: 0.9rem;
                           border-radius: 2px; vertical-align: middle;
                           margin-right: 0.25rem; border: 1px solid rgba(0,0,0,0.2); }
  main { display: grid; grid-template-columns: minmax(320px, 28%) minmax(280px, 22%) 1fr;
         min-height: 0; }
  @media (max-width: 1100px) {
    main { grid-template-columns: 1fr; grid-auto-rows: minmax(0, auto); }
  }
  .panel { padding: 0.6rem 0.75rem; min-height: 0; overflow: auto;
           border-right: 1px solid #ccc; }
  .panel:last-child { border-right: none; }
  .panel h2 { font-size: 0.9rem; margin: 0 0 0.5rem; color: #333;
              text-transform: uppercase; letter-spacing: 0.04em; }
  .plate-wrap { display: flex; justify-content: center; }
  .plate { display: grid; gap: 2px; user-select: none; }
  .plate .corner, .plate .col-h, .plate .row-h {
    font-size: 0.65rem; color: #888; text-align: center; line-height: 1;
    display: flex; align-items: center; justify-content: center;
  }
  .plate .row-h { justify-content: flex-end; padding-right: 0.25rem; }
  .well { width: 100%; aspect-ratio: 1; border-radius: 50%; cursor: pointer;
          background: #eee; border: 1px solid #ccc; position: relative;
          transition: transform 0.05s; }
  .well.has-data { background: var(--accent-soft); border-color: #93c5fd; }
  .well.selected { background: var(--accent); border-color: #1e3a8a;
                   transform: scale(1.15); z-index: 2; }
  .well:hover { border-color: #000; }
  .well .count { font-size: 0.55rem; color: #1e3a8a; position: absolute;
                 inset: 0; display: flex; align-items: center;
                 justify-content: center; font-weight: 600; pointer-events: none; }
  .actions-row { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.5rem;
                 font-size: 0.75rem; }
  .action-pill { padding: 1px 0.4rem; border-radius: 999px;
                 border: 1px solid #ccc; cursor: pointer; background: white;
                 color: #333; }
  .action-pill.active { background: var(--accent); color: white; border-color: var(--accent); }
  .action-pill .swatch { display: inline-block; width: 0.6rem; height: 0.6rem;
                         border-radius: 2px; margin-right: 0.25rem;
                         vertical-align: middle; border: 1px solid rgba(0,0,0,0.2); }
  .ds-list { display: flex; flex-direction: column; gap: 2px; font-size: 0.8rem; }
  .ds-group { margin-top: 0.5rem; }
  .ds-group h3 { font-size: 0.75rem; margin: 0 0 0.2rem; color: #555;
                 text-transform: uppercase; letter-spacing: 0.04em;
                 padding-bottom: 0.15rem; border-bottom: 1px solid #ddd; }
  .ds { padding: 0.25rem 0.4rem; border-radius: 3px; cursor: pointer;
        border: 1px solid transparent; }
  .ds:hover { background: rgba(127,127,127,0.1); }
  .ds.selected { background: var(--accent-soft); border-color: var(--accent); }
  .ds .ds-stem { font-family: ui-monospace, monospace; font-size: 0.72rem;
                 word-break: break-all; }
  .ds .ds-meta { color: #777; font-size: 0.7rem; }
  .viewer-pane { display: flex; flex-direction: column; gap: 0.5rem; min-height: 0; }
  .viewer-meta { font-size: 0.78rem; color: #555; }
  .viewer-meta code { background: rgba(127,127,127,0.12); padding: 0 0.25rem;
                      border-radius: 3px; font-size: 0.95em; }
  .viewer-frame { flex: 1; min-height: 360px; width: 100%;
                  border: 1px solid #ccc; border-radius: 4px; }
  .analysis-frame { width: 100%; height: 320px;
                    border: 1px solid #ccc; border-radius: 4px; }
  .empty { color: #888; font-size: 0.85rem; padding: 0.5rem 0; }
  .controls { display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center;
              font-size: 0.78rem; }
  .controls input[type=text] { padding: 0.15rem 0.4rem; border-radius: 3px;
                                border: 1px solid #bbb; min-width: 8rem; }
  button.reset { background: white; border: 1px solid #bbb;
                 padding: 0.15rem 0.5rem; border-radius: 3px;
                 cursor: pointer; font-size: 0.78rem; }
  button.reset:hover { background: #f3f4f6; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="summary" id="summary"></div>
  <div class="legend" id="legend"></div>
</header>
<main>
  <section class="panel" id="plate-panel">
    <h2>Plate (<span id="plate-label"></span>)</h2>
    <div class="plate-wrap"><div class="plate" id="plate"></div></div>
    <div class="controls" style="margin-top:0.6rem">
      <button class="reset" id="reset">Show all wells</button>
      <input type="text" id="search" placeholder="Filter datasets…">
    </div>
    <div class="actions-row" id="action-pills"></div>
  </section>
  <section class="panel">
    <h2>Datasets <span id="ds-count" style="color:#888;font-weight:400"></span></h2>
    <div class="ds-list" id="ds-list"></div>
  </section>
  <section class="panel viewer-pane">
    <h2>Viewer</h2>
    <div class="viewer-meta" id="viewer-meta">Pick a dataset.</div>
    <iframe class="viewer-frame" id="viewer" src="about:blank"
            allow="cross-origin-isolated" loading="lazy"></iframe>
    <div id="analysis-slot"></div>
  </section>
</main>
<script>
const DATA = __DATA__;
const PLATE = __PLATE__;
const VIZARR_BASE = "__VIZARR_BASE__";

// --- action color palette (stable per action name)
const PALETTE = ["#3b82f6","#ef4444","#10b981","#f59e0b","#8b5cf6",
                 "#ec4899","#14b8a6","#f97316","#84cc16","#6366f1"];
function actionColor(action) {
  if (!action) return "#9ca3af";
  let h = 0;
  for (let i = 0; i < action.length; i++) h = (h * 31 + action.charCodeAt(i)) >>> 0;
  return PALETTE[h % PALETTE.length];
}

const allActions = [...new Set(DATA.map(d => d.action).filter(Boolean))].sort();

// --- index by well
const byWell = {};
for (const d of DATA) {
  (byWell[d.well] ||= []).push(d);
}
function sortKey(d) {
  return [d.action || "", d.action_index || 0,
          d.field_index || d.partial_tile_index || 0,
          d.acquisition_index || 0, d.stem];
}
for (const w of Object.keys(byWell)) {
  byWell[w].sort((a, b) => {
    const [aa, bb] = [sortKey(a), sortKey(b)];
    for (let i = 0; i < aa.length; i++) {
      if (aa[i] < bb[i]) return -1;
      if (aa[i] > bb[i]) return 1;
    }
    return 0;
  });
}

// --- header summary + legend
document.getElementById("summary").textContent =
  `${DATA.length} dataset(s) · ${Object.keys(byWell).filter(w => w !== "unknown").length} well(s) · ${allActions.length} action(s)`;
document.getElementById("plate-label").textContent = PLATE.label;
const legend = document.getElementById("legend");
for (const a of allActions) {
  const span = document.createElement("span");
  span.innerHTML = `<span class="swatch" style="background:${actionColor(a)}"></span>${a}`;
  legend.appendChild(span);
}

// --- plate grid
const plate = document.getElementById("plate");
plate.style.gridTemplateColumns = `auto repeat(${PLATE.n_cols}, minmax(14px, 1fr))`;
function makeCell(cls, text) {
  const d = document.createElement("div");
  d.className = cls;
  if (text != null) d.textContent = text;
  return d;
}
plate.appendChild(makeCell("corner"));
for (const c of PLATE.cols) plate.appendChild(makeCell("col-h", c));
const wellEls = {};
for (const r of PLATE.rows) {
  plate.appendChild(makeCell("row-h", r));
  for (const c of PLATE.cols) {
    const wid = `${r}${String(c).padStart(2, "0")}`;
    const entries = byWell[wid] || [];
    const el = document.createElement("div");
    el.className = "well" + (entries.length ? " has-data" : "");
    el.dataset.wellId = wid;
    el.title = entries.length
      ? `${wid}: ${entries.length} dataset(s)\n` +
        [...new Set(entries.map(e => e.action).filter(Boolean))].join(", ")
      : `${wid}: empty`;
    if (entries.length) {
      const stroke = actionColor(entries[0].action);
      el.style.boxShadow = `inset 0 0 0 2px ${stroke}`;
      const cnt = document.createElement("span");
      cnt.className = "count";
      cnt.textContent = entries.length;
      el.appendChild(cnt);
      el.addEventListener("click", () => selectWell(wid));
    }
    wellEls[wid] = el;
    plate.appendChild(el);
  }
}

// --- action filter pills
const pillEls = {};
const activeActions = new Set();
const actionPillsRow = document.getElementById("action-pills");
for (const a of allActions) {
  const b = document.createElement("button");
  b.className = "action-pill";
  b.innerHTML = `<span class="swatch" style="background:${actionColor(a)}"></span>${a}`;
  b.addEventListener("click", () => {
    if (activeActions.has(a)) activeActions.delete(a);
    else activeActions.add(a);
    b.classList.toggle("active");
    render();
  });
  pillEls[a] = b;
  actionPillsRow.appendChild(b);
}

// --- search
const searchInput = document.getElementById("search");
searchInput.addEventListener("input", () => render());

// --- state
let selectedWell = null;
let selectedDs = null;

function selectWell(wid) {
  if (selectedWell === wid) { selectedWell = null; }
  else { selectedWell = wid; }
  for (const [w, el] of Object.entries(wellEls))
    el.classList.toggle("selected", w === selectedWell);
  render();
}
document.getElementById("reset").addEventListener("click", () => {
  selectedWell = null;
  for (const el of Object.values(wellEls)) el.classList.remove("selected");
  activeActions.clear();
  for (const b of Object.values(pillEls)) b.classList.remove("active");
  searchInput.value = "";
  render();
});

// --- dataset list rendering
function visibleEntries() {
  const q = searchInput.value.trim().toLowerCase();
  let entries = DATA;
  if (selectedWell) entries = byWell[selectedWell] || [];
  if (activeActions.size) entries = entries.filter(d => activeActions.has(d.action));
  if (q) entries = entries.filter(d =>
    d.stem.toLowerCase().includes(q) ||
    (d.action || "").toLowerCase().includes(q) ||
    d.well.toLowerCase().includes(q)
  );
  return entries.slice().sort((a, b) => {
    const [aa, bb] = [sortKey(a), sortKey(b)];
    if (a.well !== b.well) return a.well < b.well ? -1 : 1;
    for (let i = 0; i < aa.length; i++) {
      if (aa[i] < bb[i]) return -1;
      if (aa[i] > bb[i]) return 1;
    }
    return 0;
  });
}

const dsList = document.getElementById("ds-list");
const dsCount = document.getElementById("ds-count");

function shapeStr(d) {
  if (!d.shape || !d.shape.length) return "";
  const ax = d.axes.map(a => a.toUpperCase()).join("");
  return `${ax}=${d.shape.join("×")}`;
}

function render() {
  const entries = visibleEntries();
  dsCount.textContent = `(${entries.length}${
    selectedWell ? ` in well ${selectedWell}` : " across all wells"})`;
  dsList.innerHTML = "";
  if (!entries.length) {
    dsList.innerHTML = '<div class="empty">No datasets match.</div>';
    return;
  }
  // Group by well first (when not filtered to one well), then by action
  const groups = new Map();
  for (const d of entries) {
    const key = selectedWell ? (d.action || "—") : `${d.well} · ${d.action || "—"}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(d);
  }
  for (const [key, group] of groups.entries()) {
    const g = document.createElement("div");
    g.className = "ds-group";
    const h = document.createElement("h3");
    h.innerHTML = `<span class="swatch" style="display:inline-block;width:0.6rem;height:0.6rem;border-radius:2px;background:${actionColor(group[0].action)};vertical-align:middle;margin-right:0.3rem"></span>${key} <span style="color:#999;font-weight:400">(${group.length})</span>`;
    g.appendChild(h);
    for (const d of group) {
      const row = document.createElement("div");
      row.className = "ds";
      if (selectedDs && selectedDs.rel_path === d.rel_path) row.classList.add("selected");
      const meta = [];
      if (d.field_index != null) meta.push(`F${d.field_index}`);
      if (d.partial_tile_index != null) meta.push(`M${d.partial_tile_index}`);
      if (d.action_index != null) meta.push(`A${d.action_index}`);
      if (d.acquisition_index != null) meta.push(`Acq${d.acquisition_index}`);
      const sh = shapeStr(d);
      if (sh) meta.push(sh);
      row.innerHTML = `<div class="ds-stem">${d.stem}</div><div class="ds-meta">${meta.join(" · ")}</div>`;
      row.addEventListener("click", () => selectDs(d));
      g.appendChild(row);
    }
    dsList.appendChild(g);
  }
}

const viewer = document.getElementById("viewer");
const viewerMeta = document.getElementById("viewer-meta");
const analysisSlot = document.getElementById("analysis-slot");

function selectDs(d) {
  selectedDs = d;
  for (const el of dsList.querySelectorAll(".ds")) el.classList.remove("selected");
  // Re-render to update selected styling
  render();
  const src = `${VIZARR_BASE}?source=./${d.rel_path}`;
  viewer.src = src;
  const bits = [];
  bits.push(`<b>${d.stem}</b>`);
  bits.push(`well <code>${d.well}</code>`);
  if (d.action) bits.push(`action <code>${d.action}</code>`);
  if (d.shape && d.shape.length) bits.push(`<code>${shapeStr(d)}</code>`);
  if (d.dtype) bits.push(`<code>${d.dtype}</code>`);
  bits.push(`<code>${d.n_levels}</code> levels`);
  if (d.channel_names && d.channel_names.length)
    bits.push(`channels: ${d.channel_names.join(", ")}`);
  bits.push(`<code>./${d.rel_path}</code>`);
  viewerMeta.innerHTML = bits.join(" · ");
  analysisSlot.innerHTML = "";
  if (d.analysis_rel) {
    const lbl = document.createElement("div");
    lbl.className = "viewer-meta";
    lbl.innerHTML = `Analysis: <code>${d.analysis_rel}</code>`;
    const af = document.createElement("iframe");
    af.className = "analysis-frame";
    af.src = `./${d.analysis_rel}`;
    af.loading = "lazy";
    analysisSlot.appendChild(lbl);
    analysisSlot.appendChild(af);
  }
}

render();
</script>
</body>
</html>
"""


@app.command()
def main(
    out_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Output directory containing *.ome.zarr datasets.",
    ),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Path for the generated HTML (default: <out_dir>/index.html).",
    ),
):
    """Walk OUT_DIR for *.ome.zarr directories and generate an HTML explorer.

    Serve the resulting page with::

        python -m http.server 8000 --directory <OUT_DIR>

    then open http://localhost:8000/. The viewer fetches zarr chunks via
    relative same-origin URLs — opening the HTML file directly (file://)
    will not work because Vizarr cannot load the data.
    """
    out_dir = out_dir.resolve()
    output_path = output.resolve() if output is not None else out_dir / "index.html"

    zarr_dirs = sorted(p for p in out_dir.rglob("*.ome.zarr") if p.is_dir())
    if not zarr_dirs:
        typer.secho(
            f"No *.ome.zarr directories found under {out_dir}.",
            fg=typer.colors.YELLOW,
        )

    entries = [_summarize(z, out_dir) for z in zarr_dirs]
    plate = _detect_plate(entries)

    html_text = (
        HTML_TEMPLATE
        .replace("__TITLE__", out_dir.name)
        .replace("__VIZARR_BASE__", VIZARR_BASE)
        .replace("__DATA__", json.dumps(entries, separators=(",", ":")))
        .replace("__PLATE__", json.dumps(plate, separators=(",", ":")))
    )
    output_path.write_text(html_text)

    typer.secho(
        f"Wrote {output_path} ({len(entries)} dataset(s), plate: {plate['label']})",
        fg=typer.colors.GREEN,
    )
    typer.echo(
        f"Serve with:  python -m http.server 8000 --directory {out_dir}\n"
        f"Then open:   http://localhost:8000/{output_path.name}"
    )


if __name__ == "__main__":
    app()
