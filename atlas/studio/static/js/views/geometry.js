// Geometry (plan §3): the tool rail, the feature tree, the canvas, the inspector, the
// start screen, materials, and the Check view.  The canvas is `geo/editor.js`; this
// module lays the view out and turns what the editor does into edits.

/* global paper */
import { store, subscribe, send, edit, family } from '../store.js';
import { h, clear, svgIcon, cssVar, fmt, dialog, toast } from '../ui.js';
import { Stage, SWATCHES } from '../canvas.js';
import { Editor } from '../geo/editor.js';
import { itemOf, byIdOf, topFeatures, describe, areaOf, domainItem, consumed, translated, nextId, blank } from '../geo/model.js';
import { readDxf, readSvg, toFeatures } from '../geo/imports.js';

let root, stage, editor;
let els = {};
let checkMode = false;
let check = null;            // the last Check result
let checkN = null;
let showStart = false;
let dismissedStart = false;  // a drawing tool was chosen on an empty drawing
let busy = 0;
let paletteMaterial = null;
let lastFiles = [];

const RAIL = [
  ['select', 'Select', 'Select · V. Click a shape; drag handles to reshape.', 'M5 3l13 8-6 1.5L9.5 19z'],
  ['pen', 'Pen', 'Pen · P. Click for a corner, drag for a curve.', 'M4 18C8 4 16 20 20 6M3 17h2v2H3zM19 5h2v2h-2z'],
  ['rect', 'Rectangle', 'Rectangle · R. Drag a box; Shift for a square.', 'M4 6h16v12H4z'],
  ['ellipse', 'Ellipse', 'Ellipse · E. Drag a box; Shift for a circle.', 'M4 12a8 6.5 0 1 0 16 0a8 6.5 0 1 0 -16 0'],
  ['arc', 'Arc', 'Arc · A. Start, end, then pull the bulge.', 'M4 18A9 9 0 0 1 20 18'],
  ['channel', 'Channel', 'Channel · C. Draw a centre line; it gets a width. For thin high-contrast strips.', 'M3 9c5-3 13 3 18 0M3 15c5-3 13 3 18 0'],
  ['dimension', 'Dimension', 'Dimension · D. Click an edge or two points to set a size.', 'M4 8v8M20 8v8M4 12h16M7 10l-3 2 3 2M17 10l3 2-3 2'],
  ['measure', 'Measure', 'Measure · M. Read a distance or angle without changing anything.', 'M3 17L17 3l4 4L7 21zM7 13l2 2M10 10l2 2M13 7l2 2'],
];
const UTIL = [
  ['hand', 'Pan', 'Pan · H, or hold Space and drag.', 'M8 13V5a1.5 1.5 0 0 1 3 0v6M11 11V4a1.5 1.5 0 0 1 3 0v7M14 11V6a1.5 1.5 0 0 1 3 0v8a6 6 0 0 1-6 6h-1a5 5 0 0 1-4-2l-3-4a1.5 1.5 0 0 1 2.4-1.8L8 14'],
  ['import', 'Import', 'Import · I. CAD (STEP, DXF), Gmsh mesh, or SVG.', 'M12 3v12M7 10l5 5 5-5M4 19h16'],
];
const LATER = { channel: 'P2', dimension: 'P2' };

const unit = () => store.state?.meta?.unit || 'm';
const geomOf = () => store.state?.geometry || { parameters: [], features: [] };
const hasShape = () => topFeatures(geomOf()).some((f) => f.role === 'add');

// -- edits ----------------------------------------------------------------------------
async function commit(ops, label, defaults = false) {
  busy++; renderBusy();
  const r = await edit(ops, label, defaults);
  busy--; renderBusy();
  if (!r.ok && store.state) editor.setGeom(store.state.geometry, store.state.drawing, store.state.case.materials);
  editor.flushPending();
  return r;
}

function setField(fid, field, value, label) {
  const i = geomOf().features.findIndex((f) => f.id === fid);
  if (i < 0) return;
  return commit([{ op: 'set', path: ['geometry', 'features', i, field], value }], label);
}

function removeFeatures(ids) {
  const g = geomOf();
  const all = new Set(ids);
  // a combine's inputs go with it
  const grow = (id) => { const f = g.features.find((x) => x.id === id); if (f && f.kind === 'combine') f.inputs.forEach((i) => { all.add(i); grow(i); }); };
  ids.forEach(grow);
  const used = consumed(g);
  for (const id of all) {
    if (used.has(id)) {
      const parent = g.features.find((f) => f.kind === 'combine' && f.inputs.includes(id));
      if (parent && !all.has(parent.id)) { toast('info', `${id} is part of ${parent.name || parent.id}: delete that, or ungroup it first.`); return; }
    }
  }
  const idx = g.features.map((f, i) => (all.has(f.id) ? i : -1)).filter((i) => i >= 0).sort((a, b) => b - a);
  editor.sel.clear();
  commit(idx.map((i) => ({ op: 'remove', path: ['geometry', 'features', i] })), `deleted ${all.size} ${all.size === 1 ? 'shape' : 'shapes'}`);
}

function duplicate(ids) {
  const g = JSON.parse(JSON.stringify(geomOf()));
  const step = 0.04 * Math.max(editor.grid.width, editor.grid.height);
  const ops = [];
  const fresh = [];
  for (const id of ids) {
    const f = g.features.find((x) => x.id === id);
    if (!f || f.kind === 'combine') continue;
    const c = translated(f, step, -step);
    c.id = nextId(g); c.name = `${f.name || f.id} copy`;
    g.features.push(c);
    fresh.push(c.id);
    ops.push({ op: 'insert', path: ['geometry', 'features'], value: c });
  }
  if (!ops.length) { toast('info', 'Duplicate a plain shape (a combined one: its inputs).'); return; }
  editor.sel = new Set(fresh);
  commit(ops, `duplicated ${ops.length}`);
}

function combine(op) {
  const g = geomOf();
  const order = g.features.map((f) => f.id);
  const ids = [...editor.sel].sort((a, b) => order.indexOf(a) - order.indexOf(b));
  if (ids.length < 2) return;
  const base = g.features.find((f) => f.id === ids[0]);
  const id = nextId(g);
  const names = ids.map((i) => g.features.find((f) => f.id === i)).map((f) => f.name || f.id);
  const name = op === 'union' ? names.join(' ∪ ') : op === 'subtract' ? `${names[0]} − (${names.slice(1).join(', ')})` : names.join(' ∩ ');
  const f = blank('combine', id, base.role === 'cut' ? 'cut' : base.role, { op, inputs: ids, name: name.slice(0, 60), material: base.material });
  const at = Math.max(...ids.map((i) => order.indexOf(i))) + 1;
  editor.sel = new Set([id]);
  commit([{ op: 'insert', path: ['geometry', 'features'], index: at, value: f }], `${op}: ${name}`);
}

function ungroup(fid) {
  const g = geomOf();
  const i = g.features.findIndex((f) => f.id === fid);
  if (i < 0) return;
  editor.sel = new Set(g.features[i].inputs);
  commit([{ op: 'remove', path: ['geometry', 'features', i] }], `ungrouped ${g.features[i].name || fid}`);
}

// -- the rail -----------------------------------------------------------------------------
function setTool(t) {
  if (t === 'import') { importFile(); return; }
  if (LATER[t]) { toast('info', `${RAIL.find((r) => r[0] === t)[1]} arrives in phase ${LATER[t]}.`); return; }
  editor.fillMaterial = null;
  editor.setTool(t);
  if (t !== 'select' && t !== 'hand') { dismissedStart = true; showStart = false; renderStart(); }
  renderRail(); renderInspector();
}

function renderRail() {
  for (const b of els.rail.querySelectorAll('.tool[data-tool]')) {
    b.setAttribute('aria-pressed', String(b.dataset.tool === editor.tool && !editor.fillMaterial));
  }
}

function railButton([key, name, tip, d]) {
  const b = h('button', { class: 'tool', 'data-tool': key, 'aria-label': name, 'data-tip': tip + (LATER[key] ? ` Arrives in ${LATER[key]}.` : ''),
    'aria-pressed': 'false', style: LATER[key] ? { opacity: 0.45 } : {} }, svgIcon(d));
  b.addEventListener('click', () => setTool(key));
  return b;
}

// -- the feature tree ----------------------------------------------------------------------
function swatch(f) {
  if (f.role === 'cut') return h('span', { class: 'sw cut' });
  if (f.role === 'guide') return h('span', { class: 'sw', style: { border: '1.5px dashed var(--muted)' } });
  const bg = f.role === 'region' ? editor.materialColour(f.material) : 'var(--accent)';
  return h('span', { class: 'sw', style: { background: bg } });
}

function treeRow(f, depth, list) {
  const sel = editor.sel.has(f.id);
  const eye = h('button', { class: 'btn small', style: { padding: '0 4px', border: 0, background: 'transparent', color: 'var(--muted)' },
    'data-tip': f.hidden ? 'Hidden: click to show' : 'Click to hide', 'aria-label': f.hidden ? 'Show' : 'Hide' }, f.hidden ? '◌' : '●');
  eye.addEventListener('click', (e) => { e.stopPropagation(); setField(f.id, 'hidden', !f.hidden, `${f.hidden ? 'showed' : 'hid'} ${f.name || f.id}`); });
  const row = h('div', { class: 'item', role: 'button', tabindex: '0', 'aria-selected': String(sel), draggable: depth === 0 ? 'true' : 'false',
    style: { paddingLeft: `${7 + 14 * depth}px`, opacity: f.hidden ? 0.55 : 1 },
    'data-tip': `${{ add: 'Adds to the domain', cut: 'Cuts a hole', region: 'A material region', guide: 'A guide: never part of the domain' }[f.role]}${f.material ? ` · ${f.material}` : ''}` },
    swatch(f), h('span', { class: 'name' }, f.name || f.id), h('span', { class: 'meta' }, describe(f, fmt)), eye);
  row.addEventListener('click', (e) => {
    if (e.shiftKey) { if (sel) editor.sel.delete(f.id); else editor.sel.add(f.id); }
    else { editor.sel.clear(); editor.sel.add(f.id); }
    editor.redraw(); renderTree(); renderInspector();
  });
  row.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); row.click(); } });
  row.addEventListener('dragstart', (e) => { e.dataTransfer.setData('text/plain', f.id); e.dataTransfer.effectAllowed = 'move'; });
  row.addEventListener('dragover', (e) => { e.preventDefault(); row.style.boxShadow = 'inset 0 2px 0 var(--accent)'; });
  row.addEventListener('dragleave', () => { row.style.boxShadow = ''; });
  row.addEventListener('drop', (e) => {
    e.preventDefault(); row.style.boxShadow = '';
    const from = geomOf().features.findIndex((x) => x.id === e.dataTransfer.getData('text/plain'));
    const to = geomOf().features.findIndex((x) => x.id === f.id);
    if (from >= 0 && to >= 0 && from !== to) commit([{ op: 'move', path: ['geometry', 'features'], from, to }], 'reordered the features');
  });
  list.append(row);
  if (f.kind === 'combine') {
    const byId = byIdOf(geomOf());
    for (const i of f.inputs) if (byId[i]) treeRow(byId[i], depth + 1, list);
  }
}

function renderTree() {
  const list = clear(els.tree);
  const g = geomOf();
  const used = consumed(g);
  if (!g.features.length) list.append(h('div', { class: 'empty' }, 'Nothing drawn yet.'));
  for (const f of g.features) if (!used.has(f.id)) treeRow(f, 0, list);
}

// -- the inspector --------------------------------------------------------------------------
function numField(label, value, onCommit, tip = null) {
  const input = h('input', { class: 'field num', value: fmt(value, 6), inputmode: 'decimal', style: { width: '78px' } });
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') input.blur(); if (e.key === 'Escape') { input.value = fmt(value, 6); input.blur(); } });
  input.addEventListener('change', () => {
    const x = Number(input.value);
    if (!Number.isFinite(x)) { input.value = fmt(value, 6); return; }
    onCommit(x);
  });
  return h('label', { class: 'row', 'data-tip': tip }, label, h('span', { style: { display: 'flex', gap: '4px', alignItems: 'center' } }, input, h('span', { class: 'muted', style: { fontSize: '11px' } }, unit())));
}

function rolePicker(f) {
  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Role' });
  for (const [r, label, tip] of [['add', 'Add', 'Adds to the domain'], ['cut', 'Cut', 'Cuts a hole'], ['region', 'Region', 'A material region inside the domain'], ['guide', 'Guide', 'Construction: never part of the domain']]) {
    const b = h('button', { 'aria-pressed': String(f.role === r), 'data-tip': tip, disabled: (f.kind === 'path' && !f.closed && r !== 'guide') || null }, label);
    b.addEventListener('click', () => {
      if (f.role === r) return;
      const i = geomOf().features.findIndex((x) => x.id === f.id);
      const ops = [{ op: 'set', path: ['geometry', 'features', i, 'role'], value: r }];
      if ((r === 'add' || r === 'region') && !f.material) ops.push({ op: 'set', path: ['geometry', 'features', i, 'material'], value: defaultMaterial(r) });
      commit(ops, `${f.name || f.id} now ${r === 'cut' ? 'cuts' : (r === 'add' ? 'adds' : `a ${r}`)}`);
    });
    seg.append(b);
  }
  return seg;
}

function materialsOf() {
  const fam = family();
  return { ...(fam?.materials || {}), ...(store.state?.case?.materials || {}) };
}

function defaultMaterial(role) {
  if (paletteMaterial) return paletteMaterial;
  const mats = materialsOf();
  if (role === 'region') {
    const best = Object.entries(mats).sort((a, b) => ((b[1].k ?? b[1].sigma ?? 0) - (a[1].k ?? a[1].sigma ?? 0)))[0];
    return best ? best[0] : null;
  }
  return store.state?.meta?.default_material || Object.keys(mats)[0] || null;
}

function propsLine(m) {
  const p = materialsOf()[m] || {};
  if (p.k !== undefined) return `κ ${fmt(p.k)}` + (p.rho && p.cp ? ` · ρc ${fmt(p.rho * p.cp, 2)}` : '');
  if (p.sigma !== undefined) return `σ ${fmt(p.sigma)}`;
  const k0 = Object.keys(p)[0];
  return k0 ? `${k0} ${fmt(p[k0])}` : '';
}

function contrast() {
  const g = geomOf();
  const used = new Set(topFeatures(g).filter((f) => f.material && f.role !== 'cut').map((f) => f.material));
  const mats = materialsOf();
  const ks = [...used].map((m) => (mats[m] || {}).k ?? (mats[m] || {}).sigma).filter((v) => v > 0);
  if (ks.length < 1) return null;
  return Math.max(...ks) / Math.min(...ks);
}

function palette(f) {
  const box = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '4px' } });
  box.append(h('div', { class: 'label' }, 'Materials', h('span', { class: 'q', 'data-tip': f ? 'Click a material to fill the selected shape' : 'Pick a material, click a region to fill it' }, '?')));
  box.append(h('div', { class: 'row' }, h('span', {}, 'colour by'), colourBy()));
  const mats = materialsOf();
  for (const m of Object.keys(mats)) {
    const on = f ? f.material === m : paletteMaterial === m;
    const b = h('button', { class: 'item', 'aria-selected': String(on), 'data-tip': `${m} · ${propsLine(m)}` },
      h('span', { class: 'sw', style: { background: editor.materialColour(m), width: '14px', height: '14px', borderRadius: '4px' } }),
      h('span', { class: 'name' }, m[0].toUpperCase() + m.slice(1)), h('span', { class: 'meta' }, propsLine(m).split(' · ')[0]));
    b.addEventListener('click', () => {
      if (f) { setField(f.id, 'material', m, `${f.name || f.id}: ${m}`); return; }
      paletteMaterial = paletteMaterial === m ? null : m;
      editor.fillMaterial = paletteMaterial;
      editor.stage.canvas.style.cursor = paletteMaterial ? 'cell' : 'default';
      renderInspector(); renderRail();
    });
    box.append(b);
  }
  const add = h('button', { class: 'btn small', 'data-tip': 'New material from the library or by name' }, '+ New');
  add.addEventListener('click', newMaterial);
  box.append(add);
  const c = contrast();
  if (c) box.append(h('div', { class: 'row', 'data-tip': 'Largest κ over smallest κ in the domain' }, h('span', { class: 'muted' }, 'contrast'), h('b', { class: 'mono' }, c >= 1000 ? c.toExponential(0).replace('e+', ' × 10^') : fmt(c, 3))));
  return box;
}

async function newMaterial() {
  const fam = family();
  const lib = fam?.materials || {};
  const props = fam?.material_props || [];
  const name = h('input', { class: 'field', placeholder: 'name, e.g. conductor', style: { width: '100%' } });
  const base = h('select', { class: 'field' }, ...Object.keys(lib).map((m) => h('option', { value: m }, m)));
  const fields = {};
  const grid = h('div', { style: { display: 'grid', gridTemplateColumns: 'minmax(0,1fr) auto auto', gap: '6px 8px', alignItems: 'center' } });
  const fill = () => {
    clear(grid);
    for (const p of props) {
      const v = (lib[base.value] || {})[p.name] ?? p.default;
      fields[p.name] = h('input', { class: 'field num', value: fmt(v, 6), style: { width: '110px' } });
      grid.append(h('span', { 'data-tip': p.help || p.label }, p.label), fields[p.name], h('span', { class: 'muted mono', style: { fontSize: '11px' } }, p.unit));
    }
  };
  base.addEventListener('change', fill);
  fill();
  const ok = await dialog('New material', [name, h('label', { class: 'row' }, 'starting from', base), grid],
    [{ label: 'Cancel' }, { label: 'Add', primary: true, value: () => (name.value.trim() ? true : false) }]);
  if (!ok) return;
  const vals = {};
  for (const p of props) vals[p.name] = Number(fields[p.name].value);
  if (Object.values(vals).some((v) => !Number.isFinite(v))) { toast('error', 'Every property must be a number.'); return; }
  const key = name.value.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '-');
  await send('edit.apply', { ops: [{ op: 'set', path: ['materials', key], value: vals }], label: `new material ${key}` });
}

function renderInspector() {
  const box = clear(els.inspector);
  if (checkMode) { renderCheckPanel(box); return; }
  const g = geomOf();
  const sel = [...editor.sel].map((id) => g.features.find((f) => f.id === id)).filter(Boolean);
  if (sel.length >= 2) {
    box.append(h('div', { style: { fontSize: '17px', fontWeight: 700 } }, `${sel.length} selected`),
      h('div', { class: 'muted' }, 'Shift+click to add to the selection'));
    const bar = h('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' } });
    for (const [op, label, tip] of [['union', 'Union', 'Union · Ctrl+U: merge into one shape'], ['subtract', 'Subtract', 'Subtract · Ctrl+−: cut the others out of the first'],
      ['intersect', 'Intersect', 'Intersect · Ctrl+I: keep only the overlap'], ['split', 'Split', 'Split: arrives in P2'], ['pattern', 'Pattern', 'Pattern: arrives in P2'], ['mirror', 'Mirror', 'Mirror: arrives in P2']]) {
      const b = h('button', { class: 'btn', 'data-tip': tip, disabled: ['split', 'pattern', 'mirror'].includes(op) || null }, label);
      b.addEventListener('click', () => combine(op));
      bar.append(b);
    }
    box.append(bar, h('div', { class: 'muted', 'data-tip': 'The cutters stay in Features and can be moved or resized later' }, 'Every step stays editable.'));
    box.append(h('span', { style: { flex: 1 } }));
    const del = h('button', { class: 'btn danger', 'data-tip': 'Delete · Del' }, 'Delete');
    del.addEventListener('click', () => removeFeatures(sel.map((f) => f.id)));
    box.append(del);
    return;
  }
  if (sel.length === 1) {
    const f = sel[0];
    const name = h('input', { class: 'field', value: f.name || f.id, style: { fontSize: '16px', fontWeight: 700, flex: 1, minWidth: 0 } });
    name.addEventListener('change', () => setField(f.id, 'name', name.value.trim(), `renamed ${f.id}`));
    name.addEventListener('keydown', (e) => { if (e.key === 'Enter') name.blur(); });
    box.append(h('div', { style: { display: 'flex', alignItems: 'center', gap: '6px' } }, name,
      h('span', { class: `tag ${f.role === 'guide' ? 'cut' : f.role}` }, f.role)));
    box.append(rolePicker(f));
    const live = editor.feature(f.id) || f;
    if (f.kind === 'rect') {
      box.append(numField('x', live.x, (v) => setField(f.id, 'x', v, 'moved')), numField('y', live.y, (v) => setField(f.id, 'y', v, 'moved')),
        numField('width', live.w, (v) => setField(f.id, 'w', v, 'resized')), numField('height', live.h, (v) => setField(f.id, 'h', v, 'resized')),
        numField('corner radius', live.r, (v) => setField(f.id, 'r', v, 'rounded'), 'Rounds all four corners'));
    } else if (f.kind === 'ellipse') {
      box.append(numField('x centre', live.cx, (v) => setField(f.id, 'cx', v, 'moved')), numField('y centre', live.cy, (v) => setField(f.id, 'cy', v, 'moved')),
        numField(Math.abs(live.rx - live.ry) < 1e-12 ? 'radius' : 'radius x', live.rx, (v) => (Math.abs(live.rx - live.ry) < 1e-12
          ? commit([{ op: 'set', path: ['geometry', 'features', geomOf().features.findIndex((x) => x.id === f.id), 'rx'], value: v },
            { op: 'set', path: ['geometry', 'features', geomOf().features.findIndex((x) => x.id === f.id), 'ry'], value: v }], 'resized')
          : setField(f.id, 'rx', v, 'resized'))));
      if (Math.abs(live.rx - live.ry) >= 1e-12) box.append(numField('radius y', live.ry, (v) => setField(f.id, 'ry', v, 'resized')));
    } else if (f.kind === 'path') {
      const kinds = {};
      for (const k of f.edges) kinds[k] = (kinds[k] || 0) + 1;
      box.append(h('div', { class: 'row' }, h('span', { class: 'muted' }, 'path'),
        h('span', {}, `${f.closed ? 'closed' : 'open'} · ${f.points.length} points`)));
      box.append(h('div', { class: 'muted', style: { fontSize: '12px' } }, Object.entries(kinds).map(([k, n]) => `${n} ${k === 'cubic' ? 'curve' : k}${n > 1 ? 's' : ''}`).join(', ')));
    } else if (f.kind === 'combine') {
      box.append(h('div', { class: 'row' }, h('span', { class: 'muted' }, { union: 'union', subtract: 'subtract', intersect: 'intersect' }[f.op]), h('span', {}, `${f.inputs.length} inputs`)));
      const ug = h('button', { class: 'btn small', 'data-tip': 'Undo the combine; the shapes stay' }, 'Ungroup');
      ug.addEventListener('click', () => ungroup(f.id));
      box.append(ug);
    }
    if (f.role === 'add' || f.role === 'region') box.append(palette(f));
    box.append(h('span', { style: { flex: 1 } }));
    const dup = h('button', { class: 'btn', style: { flex: 1 }, 'data-tip': 'Duplicate · Ctrl+D' }, 'Duplicate');
    dup.addEventListener('click', () => duplicate([f.id]));
    const del = h('button', { class: 'btn danger', style: { flex: 1 }, 'data-tip': 'Delete · Del' }, 'Delete');
    del.addEventListener('click', () => removeFeatures([f.id]));
    box.append(h('div', { style: { display: 'flex', gap: '6px' } }, dup, del));
    return;
  }
  // nothing selected: the drawing, a tool in progress, or the palette
  if (editor.draft && (editor.draft.kind === 'rect' || editor.draft.kind === 'ellipse')) {
    const d = editor.draftFeature();
    box.append(h('div', { style: { fontSize: '17px', fontWeight: 700 } }, d.kind === 'rect' ? 'Rectangle' : 'Ellipse'),
      h('span', { class: `tag ${d.role}` }, d.role));
    const kv = h('div', { class: 'kv' });
    const pairs = d.kind === 'rect' ? [['x centre', d.x + d.w / 2], ['y centre', d.y + d.h / 2], ['width', d.w], ['height', d.h]]
      : [['x centre', d.cx], ['y centre', d.cy], ['width', 2 * d.rx], ['height', 2 * d.ry]];
    for (const [k, v] of pairs) kv.append(h('span', { class: 'k' }, k), h('span', {}, fmt(v, 4)));
    box.append(kv, h('div', { class: 'muted', style: { fontSize: '12px' } }, h('kbd', {}, 'Tab'), ' type the size · ', h('kbd', {}, 'Shift'), ' square · ', h('kbd', {}, 'Alt'), ' from centre'));
    if (editor.snapped.length) {
      box.append(h('div', { class: 'label' }, 'Snapped to'), ...editor.snapped.map((s) => h('div', { style: { fontSize: '12px' } }, s.label)));
    }
    return;
  }
  if (editor.pen) {
    const pts = editor.pen.points;
    box.append(h('div', { style: { fontSize: '17px', fontWeight: 700 } }, 'Pen'),
      h('div', { class: 'muted' }, `${pts.length} points`),
      h('div', { class: 'label' }, 'Segment to cursor'),
      h('div', { class: 'muted', style: { fontSize: '12px' } }, 'Type a number, Enter places the point.'),
      h('div', { class: 'muted', style: { fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '3px' } },
        h('span', {}, h('kbd', {}, 'Click'), ' corner · ', h('kbd', {}, 'Drag'), ' curve'),
        h('span', {}, h('kbd', {}, 'Alt'), ' break a handle · ', h('kbd', {}, 'Shift'), ' 15°'),
        h('span', {}, h('kbd', {}, 'Backspace'), ' last point'),
        h('span', {}, 'click the first point to close · ', h('kbd', {}, 'Enter'), ' finish open'),
        h('span', {}, h('kbd', {}, 'Esc'), ' cancel')));
    return;
  }
  const st = store.state;
  const d = st?.drawing;
  box.append(h('div', { style: { fontSize: '17px', fontWeight: 700 } }, 'Drawing'));
  if (d) {
    box.append(h('div', { class: 'kv' },
      h('span', { class: 'k', 'data-tip': 'Cells along x and y' }, 'grid'), h('span', {}, `${d.nx} × ${d.ny}`),
      h('span', { class: 'k', 'data-tip': 'The side of one cell' }, 'cell'), h('span', {}, `${fmt(d.dx, 4)} ${unit()}`),
      h('span', { class: 'k' }, 'size'), h('span', {}, `${fmt(d.width)} × ${fmt(d.height)}`),
      h('span', { class: 'k' }, 'shapes'), h('span', {}, String(geomOf().features.length))));
  }
  const ns = h('button', { class: 'btn', 'data-tip': 'Start from a plate, a bend, a channel… or a file' }, 'New geometry…');
  ns.addEventListener('click', () => { showStart = true; renderStart(); });
  box.append(ns);
  box.append(palette(null));
}

// -- the start screen -----------------------------------------------------------------------
const THUMBS = {
  plate: '<rect x="10" y="12" width="100" height="40" rx="2" class="t-fill"></rect>',
  'plate-hole': '<rect x="10" y="12" width="100" height="40" rx="2" class="t-fill"></rect><circle cx="78" cy="32" r="10" class="t-hole"></circle>',
  'l-bracket': '<path d="M20 6H44V36Q44 40 48 40H100V58H20Z" class="t-fill"></path>',
  bend: '<path d="M30 58V34A28 28 0 0 1 58 6H90V24H58A10 10 0 0 0 48 34V58Z" class="t-fill"></path>',
  's-channel': '<path d="M8 40C40 40 40 10 80 10H112V26H80C56 26 56 56 8 56Z" class="t-fill"></path>',
  annulus: '<circle cx="60" cy="32" r="26" class="t-fill"></circle><circle cx="60" cy="32" r="12" class="t-hole"></circle>',
  'high-contrast': '<rect x="10" y="12" width="100" height="40" rx="2" class="t-fill"></rect><path d="M10 28C40 24 70 34 110 30" stroke="#00D7D2" stroke-width="3" fill="none"></path><g fill="#00D7D2"><circle cx="28" cy="42" r="3"></circle><circle cx="52" cy="18" r="2.5"></circle><circle cx="74" cy="44" r="3.5"></circle><circle cx="94" cy="20" r="2.5"></circle></g>',
  blank: '<path d="M60 22v20M50 32h20" stroke="currentColor" stroke-width="1.6"></path>',
};

function renderStart() {
  const ov = els.start;
  const show = (showStart || (!hasShape() && !dismissedStart)) && !checkMode;
  ov.hidden = !show;
  if (!show) return;
  clear(ov);
  const d = store.state?.drawing;
  const wIn = h('input', { class: 'field num', value: fmt(d ? Math.min(d.width, d.height * 2) : 0.6, 4), style: { width: '64px' }, 'aria-label': 'width' });
  const hIn = h('input', { class: 'field num', value: fmt(d ? Math.min(d.height, d.width / 2) : 0.3, 4), style: { width: '64px' }, 'aria-label': 'height' });
  const units = h('select', { class: 'field', 'aria-label': 'Units' }, h('option', { value: unit() }, unit()), ...(unit() === 'm' ? [h('option', { value: 'mm' }, 'mm')] : []));
  const cards = h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '10px' } });
  for (const s of store.catalog?.starts || []) {
    const svg = `<svg width="100%" height="58" viewBox="0 0 120 64" aria-hidden="true">${THUMBS[s.key] || ''}</svg>`;
    const b = h('button', { class: 'startcard', 'data-tip': s.tip, 'data-start': s.key,
      style: s.key === 'high-contrast' ? { borderColor: 'var(--accent)', background: 'var(--accent-wash)' } : (s.key === 'blank' ? { borderStyle: 'dashed', background: 'transparent' } : {}) });
    b.innerHTML = svg;
    b.append(h('span', { style: { fontWeight: 600 } }, s.label));
    b.addEventListener('click', async () => {
      const W = Number(wIn.value), H = Number(hIn.value);
      if (!(W > 0 && H > 0)) { toast('error', 'Width and height must be positive numbers.'); return; }
      busy++; renderBusy();
      const r = await send('geometry.start', { shape: s.key, params: { W, H, unit: units.value } });
      busy--; renderBusy();
      if (r.ok) { showStart = s.key === 'blank'; editor.sel.clear(); if (s.key !== 'blank') renderStart(); stage.fitted = false; }
    });
    cards.append(b);
  }
  const drop = h('div', { class: 'dropzone' },
    svgIcon('M12 3v12M7 10l5 5 5-5M4 19h16', 34), h('b', {}, 'Drop a file'),
    h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '5px', justifyContent: 'center' } },
      ...[['STEP', 'CAD solid or surface; a 2-D section is taken'], ['DXF', '2-D CAD drawing: lines, arcs, splines'],
        ['.msh', 'Gmsh mesh; physical groups name the domain, regions and boundaries'], ['SVG', 'A vector drawing; its paths become shapes']]
        .map(([t, tip]) => h('span', { class: 'chipkind', 'data-tip': tip }, t))));
  drop.addEventListener('click', importFile);
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('over'); const f = e.dataTransfer.files[0]; if (f) readFile(f); });
  const bench = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '4px' } });
  for (const b of store.catalog?.benchmarks || []) {
    const btn = h('button', { class: 'btn', 'data-tip': b.tip, style: { display: 'flex', justifyContent: 'space-between' } },
      h('span', {}, b.label), h('span', { class: 'muted', style: { fontSize: '12px' } }, b.meta));
    btn.addEventListener('click', async () => { const r = await send('case.example', { key: b.key }); if (r.ok) { showStart = false; renderStart(); } });
    bench.append(btn);
  }
  const recent = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '2px', fontSize: '13px' } });
  for (const f of lastFiles.slice(0, 3)) {
    const btn = h('button', { class: 'item', 'data-tip': `Open ${f.name}` }, h('span', { class: 'name' }, f.name.replace(/\.json$/, '')), h('span', { class: 'meta' }, f.modified.slice(5, 10)));
    btn.addEventListener('click', async () => { const r = await send('case.open', { name: f.name }); if (r.ok) { showStart = false; renderStart(); } });
    recent.append(btn);
  }
  const closeBtn = hasShape() ? h('button', { class: 'btn small', 'aria-label': 'Close', 'data-tip': 'Back to the drawing' }, '×') : null;
  if (closeBtn) closeBtn.addEventListener('click', () => { showStart = false; renderStart(); });
  ov.append(h('section', { style: { flex: 1, display: 'flex', flexDirection: 'column', gap: '14px', minWidth: 0 } },
    h('div', { style: { display: 'flex', alignItems: 'baseline', gap: '14px' } }, h('h1', { style: { margin: 0, fontSize: '24px', letterSpacing: '-0.4px' } }, 'New geometry'),
      h('span', { class: 'muted' }, 'or press ', h('kbd', {}, 'P'), ' and draw'), h('span', { style: { flex: 1 } }), closeBtn),
    h('div', { class: 'sizebar' }, h('label', { class: 'muted' }, 'width ', wIn), h('span', { class: 'muted' }, '×'), h('label', { class: 'muted' }, 'height ', hIn), units),
    h('div', { class: 'label' }, 'Start from'), cards),
    h('aside', { style: { width: '250px', flex: 'none', display: 'flex', flexDirection: 'column', gap: '12px' } },
      drop, h('div', { class: 'label' }, 'Benchmarks'), bench, lastFiles.length ? h('div', { class: 'label' }, 'Recent') : null, recent));
}

// -- imports -----------------------------------------------------------------------------------
function importFile() {
  const input = h('input', { type: 'file', accept: '.step,.stp,.dxf,.svg,.msh' });
  input.addEventListener('change', () => { const f = input.files && input.files[0]; if (f) readFile(f); });
  input.click();
}

async function readFile(file) {
  const ext = file.name.toLowerCase().split('.').pop();
  const empty = !hasShape();
  try {
    if (ext === 'msh' || ext === 'step' || ext === 'stp') {
      const buf = new Uint8Array(await file.arrayBuffer());
      let bin = '';
      for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
      busy++; renderBusy();
      const r = await send('geometry.import', { format: ext, data: btoa(bin), name: file.name });
      busy--; renderBusy();
      if (r.ok) { showStart = false; renderStart(); stage.fitted = false; }
      return;
    }
    let pieces;
    if (ext === 'dxf') pieces = readDxf(await file.text());
    else if (ext === 'svg') {
      const unitIn = h('input', { class: 'field num', value: '0.001', style: { width: '90px' } });
      const ok = await dialog('SVG units', [h('label', { class: 'row' }, `One SVG unit is`, h('span', {}, unitIn, ` ${unit()}`))],
        [{ label: 'Cancel' }, { label: 'Import', primary: true, value: () => Number(unitIn.value) > 0 }]);
      if (!ok) return;
      pieces = readSvg(await file.text(), Number(unitIn.value));
    } else { toast('error', `${file.name}: STEP, DXF, SVG or a Gmsh .msh`); return; }
    if (!pieces.loops.length && !pieces.open.length) { toast('warning', `${file.name} holds no shapes this page can read.`); return; }
    const g = JSON.parse(JSON.stringify(geomOf()));
    const mk = (kind, role, extra) => {
      const f = blank(kind, nextId(g), role, extra);
      if (role === 'add' || role === 'region') f.material = defaultMaterial(role);
      g.features.push(f);
      return f;
    };
    const feats = toFeatures(pieces, empty ? 'add' : editor.role, file.name.replace(/\.[^.]+$/, ''), mk, empty);
    const r = await commit(feats.map((f) => ({ op: 'insert', path: ['geometry', 'features'], value: f })), `imported ${file.name}`);
    if (r.ok) { showStart = false; renderStart(); stage.fitted = false; toast('success', `${feats.length} shapes from ${file.name}.`); }
  } catch (err) {
    toast('error', `Could not read ${file.name}: ${err.message || err}`);
  }
}

// -- the Check view ----------------------------------------------------------------------------
async function runCheck(n = checkN) {
  const g = geomOf();
  const byId = byIdOf(g);
  const areas = {};
  try {
    const dom = domainItem(g, byId);
    areas.domain = areaOf(dom);
    for (const f of topFeatures(g)) if ((f.role === 'region' || f.role === 'add') && dom) areas[f.id] = areaOf(itemOf(f, byId).intersect(dom, { insert: false }));
  } catch (e) { /* the server's areas stand alone */ }
  const r = await send('geometry.check', n ? { n, areas } : { areas });
  if (!r.ok) return;
  check = r.check;
  checkN = check.n;
  renderFoot();
  if (checkMode) { drawCells(); renderInspector(); renderBar(); }
}

function drawCells() {
  editor.pins = checkMode && check ? (check.notes || []).filter((n) => n.at) : [];
  if (!checkMode || !check || !check.cells) { editor.checkRaster = null; editor.redraw(); return; }
  const c = check.cells;
  const bin = atob(c.data);
  const cv = document.createElement('canvas');
  cv.width = c.nx; cv.height = c.ny;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(c.nx, c.ny);
  const base = new paper.Color(cssVar('--shape-fill'));
  const legend = {};
  for (const l of c.legend) legend[l.value] = new paper.Color(editor.materialColour(l.material));
  for (let j = 0; j < c.ny; j++) {
    for (let i = 0; i < c.nx; i++) {
      const v = bin.charCodeAt(j * c.nx + i);
      const o = ((c.ny - 1 - j) * c.nx + i) * 4;
      if (!v) continue;
      const col = v === 1 ? base : (legend[v] || base);
      img.data[o] = col.red * 255; img.data[o + 1] = col.green * 255; img.data[o + 2] = col.blue * 255; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  stage.activate();
  // a canvas is an image to Raster's constructor; its `source` wants a URL
  const r = new paper.Raster(cv);
  if (r.parent) r.remove();
  r.smoothing = false;
  const W = c.nx * check.dx, H = c.ny * check.dx;
  r.bounds = new paper.Rectangle(0, -H, W, H);
  editor.checkRaster = r;
  editor.redraw();
}

function renderCheckPanel(box) {
  box.append(h('div', { style: { display: 'flex', alignItems: 'center' } }, h('span', { style: { fontSize: '17px', fontWeight: 700, flex: 1 } }, 'Check'),
    h('span', { class: 'muted', style: { fontSize: '11px' } }, check?.kernel ? `areas: ${check.kernel}` : '')));
  if (!check) { box.append(h('div', { class: 'empty' }, 'Checking…')); return; }
  const fixes = [];
  for (const n of check.notes || []) {
    const acts = h('div', { style: { display: 'flex', gap: '4px', flexWrap: 'wrap' } });
    for (const fx of n.fixes || []) {
      const b = h('button', { class: `btn small ${fx.key === 'keep' ? '' : 'primary'}`, 'data-tip': fx.tip || fx.label }, fx.label);
      b.addEventListener('click', async () => {
        if (fx.key === 'keep') { check.notes = check.notes.filter((x) => x !== n); renderInspector(); drawCells(); return; }
        busy++; renderBusy();
        await send('geometry.fix', { key: fx.key });
        busy--; renderBusy();
        if (fx.key.startsWith('grid:')) checkN = null;
        runCheck();
      });
      acts.append(b);
      if (fx.key !== 'keep' && !fixes.length) fixes.push(fx.key);
    }
    box.append(h('div', { class: `note ${n.severity === 'error' ? 'err' : 'warn'}`, style: { display: 'flex', flexDirection: 'column', gap: '4px' } },
      h('div', {}, h('b', {}, `${n.pin ? n.pin + ' ' : ''}`), n.message[0].toUpperCase() + n.message.slice(1)),
      n.detail ? h('div', { style: { opacity: 0.85 } }, n.detail) : null, acts));
  }
  const c = check.checks || {};
  const row = (label, val, tip) => h('div', { class: 'row', 'data-tip': tip }, h('span', {}, label),
    typeof val === 'number' ? h('b', { class: val ? 'warn-text' : 'ok-text' }, val ? String(val) : '✓') : h('b', { class: val ? 'ok-text' : 'err-text' }, val ? '✓' : '✗'));
  box.append(h('div', { class: 'label' }, 'Check'),
    row('outline closed', c.closed !== false), row('no self-crossings', c.no_crossings !== false), row('holes inside', c.holes_inside !== false),
    row('regions inside', c.regions_inside !== false), row('thin features', c.thin || 0, 'Any part narrower than one grid cell'),
    row('slivers', c.slivers || 0, 'A gap or strip between two shapes narrower than one cell; it can vanish or create a disconnected bit'),
    row('near-touching', c.near || 0, 'Two edges closer than one cell without meeting'));
  if ((check.areas || []).length) {
    box.append(h('div', { class: 'label' }, 'Areas', h('span', { class: 'q', 'data-tip': `Drawn: the exact area (${check.kernel}). In cells: the cells whose centres fall inside, as the solver sees it.` }, '?')));
    const grid = h('div', { style: { display: 'grid', gridTemplateColumns: 'minmax(0,1fr) auto auto auto', gap: '3px 8px', fontSize: '12px' } },
      h('span', { class: 'muted' }, ''), h('span', { class: 'muted' }, 'drawn'), h('span', { class: 'muted' }, 'in cells'), h('span', { class: 'muted' }, 'diff'));
    for (const a of check.areas) {
      grid.append(h('span', { style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, a.name),
        h('span', { class: 'mono', 'data-tip': a.page !== undefined ? `The page's own (Paper.js): ${fmt(a.page, 5)}` : null }, fmt(a.exact, 4)),
        h('span', { class: 'mono' }, fmt(a.cells, 4)),
        h('span', { class: 'mono' }, a.difference === null ? '–' : `${(100 * a.difference).toFixed(1)}%`));
    }
    box.append(grid, h('div', { class: 'muted', style: { fontSize: '11px' } }, `${unit()}²`));
  }
  box.append(h('span', { style: { flex: 1 } }));
  if (fixes.length) {
    const all = h('button', { class: 'btn primary', 'data-tip': 'Apply the first fix of every note' }, 'Fix all with defaults');
    all.addEventListener('click', async () => {
      const grids = fixes.filter((k) => k.startsWith('grid:')).map((k) => Number(k.slice(5)));
      const keys = [...new Set(fixes.filter((k) => !k.startsWith('grid:')))];
      if (grids.length) keys.push(`grid:${Math.max(...grids)}`);
      busy++; renderBusy();
      for (const k of keys) await send('geometry.fix', { key: k });
      busy--; renderBusy();
      checkN = null; runCheck();
    });
    box.append(all);
  }
}

function toggleCheck(on = !checkMode) {
  checkMode = on;
  els.checkBtn.setAttribute('aria-pressed', String(checkMode));
  if (checkMode) { showStart = false; renderStart(); if (check) drawCells(); runCheck(); }
  else { editor.checkRaster = null; editor.pins = []; editor.redraw(); }
  renderBar(); renderInspector();
}

// -- the bar, the foot, busy ------------------------------------------------------------------
function renderBar() {
  const bar = clear(els.bar);
  if (checkMode) {
    const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Grid n', 'data-tip': 'Cells per side of the full-domain grid. The check reads the geometry at this resolution.' });
    for (const n of check?.grids || [32, 64, 128, 256]) {
      const b = h('button', { 'aria-pressed': String((check?.n ?? checkN) === n) }, n === check?.own_n ? `${n} (case)` : String(n));
      b.addEventListener('click', () => { checkN = n; runCheck(n); });
      seg.append(b);
    }
    bar.append(h('span', { class: 'muted' }, 'grid n'), seg,
      h('span', { class: 'toggle', 'aria-pressed': 'true', 'data-tip': 'Draw the cells whose centres fall in each region: what a grid solver actually sees' }, 'as the solver sees it'),
      h('span', { class: 'spacer' }), h('span', { class: 'muted mono' }, check?.cells ? `${check.cells.count.toLocaleString()} cells` : ''));
    return;
  }
  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'What new shapes do' });
  for (const [r, label, tip] of [['add', 'Add', 'New shapes add to the domain'], ['cut', 'Cut', 'New shapes cut holes · hold Alt to switch while drawing'], ['region', 'Region', 'New shapes become material regions inside the domain']]) {
    const b = h('button', { 'aria-pressed': String(editor.role === r), 'data-tip': tip }, label);
    b.addEventListener('click', () => { editor.role = r; renderBar(); });
    seg.append(b);
  }
  const tog = (key, label, tip) => {
    const b = h('button', { class: 'toggle', 'aria-pressed': String(!!editor.opts[key]), 'data-tip': tip }, label);
    b.addEventListener('click', () => { editor.opts[key] = !editor.opts[key]; renderBar(); });
    return b;
  };
  bar.append(seg, h('span', { class: 'vr' }), tog('grid', 'Grid', 'Snap to the grid'), tog('guides', 'Guides', 'Smart guides: align to edges, centres and midpoints'),
    tog('tangent', 'Tangent', 'Snap new curves tangent to existing edges'),
    h('span', { class: 'spacer' }), els.busy, h('span', { class: 'muted', 'data-tip': 'Units for every number on screen' }, unit()),
    h('span', { class: 'muted mono', 'data-tip': 'Zoom · Ctrl+scroll; Fit · Shift+1', id: 'geo-zoom' }, ''));
  renderZoom();
}

function colourBy() {
  const colour = h('div', { class: 'seg', role: 'group', 'aria-label': 'Colour by', style: { flex: 'none' } });
  for (const [v, label, tip] of [['material', 'material', 'Colour regions by material'], ['kappa', 'κ, log', 'Colour by κ on a log scale']]) {
    const b = h('button', { 'aria-pressed': String(editor.colourBy === v), 'data-tip': tip }, label);
    b.addEventListener('click', () => { editor.colourBy = v; editor.redraw(); renderTree(); renderInspector(); });
    colour.append(b);
  }
  return colour;
}

function renderZoom() {
  const z = root.querySelector('#geo-zoom');
  if (!z || !stage) return;
  const d = store.state?.drawing;
  if (!d) return;
  const fitZoom = Math.min(stage.view.viewSize.width / d.width, stage.view.viewSize.height / d.height) * 0.88;
  z.textContent = `${Math.round(100 * stage.view.zoom / fitZoom)}%`;
}

function renderBusy() {
  els.busy.hidden = busy <= 0;
}

function renderFoot() {
  const foot = clear(els.foot);
  const c = check?.checks || {};
  const notes = (check?.notes || []).length;
  const geoErr = (store.state?.issues || []).filter((i) => i.where === 'geometry' && i.severity === 'error');
  foot.append(h('span', { class: c.closed === false ? 'err-text' : 'ok-text' }, c.closed === false ? '✗ open outline' : '✓ closed'),
    h('span', { class: c.no_crossings === false ? 'err-text' : 'ok-text' }, c.no_crossings === false ? '✗ crossing' : '✓ no crossings'));
  if (geoErr.length) foot.append(h('span', { class: 'err-text', 'data-tip': geoErr[0].message }, `${geoErr.length} ${geoErr.length === 1 ? 'problem' : 'problems'}`));
  if (notes) {
    const b = h('button', { class: 'linkish warn-text', 'data-tip': `${(check.notes || []).map((n) => n.message).join('; ')}. Open Check · K` }, `${notes} ${notes === 1 ? 'note' : 'notes'}`);
    b.addEventListener('click', () => toggleCheck(true));
    foot.append(b);
  }
  foot.append(h('span', { class: 'spacer' }), els.checkBtn, h('span', {}, h('kbd', {}, 'Space'), ' pan'), h('span', {}, h('kbd', {}, 'K'), ' check'));
}

async function typeSize() {
  const d = editor.draftFeature();
  if (!d) return;
  const w = h('input', { class: 'field num', value: fmt(d.kind === 'rect' ? d.w : 2 * d.rx, 6), style: { width: '90px' } });
  const ht = h('input', { class: 'field num', value: fmt(d.kind === 'rect' ? d.h : 2 * d.ry, 6), style: { width: '90px' } });
  editor.drag = null;
  const ok = await dialog(`${d.kind === 'rect' ? 'Rectangle' : 'Ellipse'} size`, [
    h('label', { class: 'row' }, 'width', h('span', {}, w, ` ${unit()}`)), h('label', { class: 'row' }, 'height', h('span', {}, ht, ` ${unit()}`))],
  [{ label: 'Cancel' }, { label: 'Place', primary: true, value: () => Number(w.value) > 0 && Number(ht.value) > 0 }]);
  if (ok) editor.typedSize(Number(w.value), Number(ht.value));
  else { editor.draft = null; editor.redraw(); }
}

// -- mount ---------------------------------------------------------------------------------------
export function mount(el) {
  root = el;
  const canvas = h('canvas', { 'aria-label': 'The geometry canvas' });
  const rail = h('nav', { class: 'rail', 'aria-label': 'Geometry tools' },
    ...RAIL.map(railButton), h('span', { class: 'gap' }), ...UTIL.map(railButton), h('span', { class: 'fill' }));
  const undo = h('button', { class: 'tool small', 'aria-label': 'Undo', 'data-tip': 'Undo · Ctrl+Z' }, svgIcon('M9 14L4 9l5-5M4 9h11a5 5 0 0 1 0 10h-3', 18));
  const redo = h('button', { class: 'tool small', 'aria-label': 'Redo', 'data-tip': 'Redo · Ctrl+Shift+Z' }, svgIcon('M15 14l5-5-5-5M20 9H9a5 5 0 0 0 0 10h3', 18));
  undo.addEventListener('click', () => send('edit.undo'));
  redo.addEventListener('click', () => send('edit.redo'));
  rail.append(undo, redo);
  els.rail = rail;
  els.tree = h('div', { class: 'items', style: { flex: 1, minHeight: 0 } });
  const features = h('aside', { class: 'panel', style: { width: '196px', flex: 'none' } },
    h('div', { class: 'label' }, 'Features', h('span', { class: 'q', 'data-tip': 'Applied top to bottom. Drag to reorder; click the dot to hide.' }, '?')),
    els.tree,
    h('div', { class: 'legend' },
      h('span', {}, h('i', { style: { background: 'var(--accent)' } }), 'add'),
      h('span', {}, h('i', { style: { border: '1.5px solid var(--err)' } }), 'cut'),
      h('span', {}, h('i', { style: { background: 'var(--ready)' } }), 'region')));
  els.bar = h('div', { class: 'bar' });
  els.foot = h('div', { class: 'foot' });
  els.busy = h('span', { class: 'busy', hidden: true, 'data-tip': 'The server is rebuilding the case from the drawing' }, 'Rebuilding…');
  els.checkBtn = h('button', { class: 'toggle', 'aria-pressed': 'false', 'data-tip': 'Check · K: the geometry as the solver sees it' }, 'Check');
  els.checkBtn.addEventListener('click', () => toggleCheck());
  els.start = h('div', { class: 'startover', hidden: true });
  const wrap = h('div', { class: 'canvas-wrap' }, canvas);
  const stageEl = h('main', { class: 'stage' }, els.bar, wrap, els.foot);
  els.inspector = h('aside', { class: 'panel scroll', style: { width: '240px', flex: 'none' }, id: 'geo-inspector' });
  // the start screen takes the whole work area beside the rail, as the mockup draws it
  el.append(rail, h('div', { class: 'work', style: { position: 'relative' } }, features, stageEl, els.inspector, els.start));
  stage = new Stage(canvas);
  editor = new Editor(stage, {
    commit, notify: toast, fmt: (x) => fmt(x, 4), unit,
    material: (role) => defaultMaterial(role),
    setTool, remove: removeFeatures, duplicate, importFile, toggleCheck, typeSize,
    changed: () => { if (editor.draft || editor.pen || editor.drag) renderInspector(); renderZoom(); },
    selected: () => { renderTree(); renderInspector(); },
  });
  editor.setTool('select');
  renderRail(); renderBar(); renderFoot();
  // for the page checks: what a person does through a file picker, by a File object
  window.atlas = Object.assign(window.atlas || {}, { geometry: { readFile, editor, runCheck, toggleCheck } });
  window.addEventListener('keydown', (e) => {
    if (e.shiftKey && e.code === 'Digit1' && stage.isShown()) stage.fit();
  });
  window.addEventListener('themechange', () => { stage.drawGrid(); if (checkMode) drawCells(); else editor.redraw(); });
  let lastSerial = null;
  subscribe(async (what) => {
    if (what === 'state' || what === 'catalog') {
      const st = store.state;
      if (!st) return;
      if (lastSerial !== st.meta.serial) {
        lastSerial = st.meta.serial; editor.sel.clear(); stage.fitted = false; check = null; checkN = null;
        dismissedStart = false;
      }
      editor.setGeom(st.geometry, st.drawing, { ...(family()?.materials || {}), ...st.case.materials });
      renderTree(); renderInspector(); renderBar(); renderStart();
      if (what === 'state') {
        const r = await send('case.files');
        if (r.ok) { lastFiles = r.files; if (!els.start.hidden) renderStart(); }
        runCheck(checkMode ? checkN : null);
      }
    }
    if (what === 'tab' && store.tab === 'geometry') editor.redraw();
  });
}
