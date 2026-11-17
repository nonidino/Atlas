// Coefficient fields and scatter inclusions (plan §3.1, §3.4).
//
// A region's coefficient is constant (its material), a formula of x, y and the
// parameters, or an image whose grey levels map onto a range.  The page previews the
// formula's range at the cells the solver will use, with the server's parser in
// formula.js; the server bins the field into at most 16 materials and reports the
// range it found (`state.fields`), shown beside the page's own: they must agree.
//
// A scatter places inclusions from a seed: count, size range, gap, shape (circles,
// channels, both) and a box.  The same seed places the same inclusions, byte for byte
// in the case file: the server draws them (resolve.py); the page only asks.

/* global paper */
import { store, family } from '../store.js';
import { h, fmt, toast } from '../ui.js';
import { parse, evaluate } from './formula.js';
import { values, formulaTrouble } from './params.js';
import { itemOf, byIdOf, domainItem, nextId, blank, P } from './model.js';

/** the field's property: what the family's materials carry (k for heat) */
function props() {
  const fam = family();
  return (fam?.material_props || []).map((p) => p.name);
}

/** the page's own range of a formula field: at the centres of the cells inside it */
export function pageRange(geom, f, drawing) {
  if (!f.field || f.field.kind !== 'formula' || !drawing) return null;
  try {
    const byId = byIdOf(geom);
    let item = itemOf(f, byId);
    const dom = domainItem(geom, byId);
    if (dom) item = item.intersect(dom, { insert: false });
    const b = item.bounds;
    const dx = drawing.dx;
    const tree = parse(f.field.expr);
    const env = values(geom);
    let lo = Infinity, hi = -Infinity, n = 0;
    const i0 = Math.max(0, Math.floor(b.left / dx)), i1 = Math.min(drawing.nx - 1, Math.ceil(b.right / dx));
    const j0 = Math.max(0, Math.floor(-b.bottom / dx)), j1 = Math.min(drawing.ny - 1, Math.ceil(-b.top / dx));
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        const x = (i + 0.5) * dx, y = (j + 0.5) * dx;
        if (!item.contains(P(x, y))) continue;
        const v = evaluate(tree, { ...env, x, y });
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        n++;
      }
    }
    return n ? { min: lo, max: hi, cells: n } : null;
  } catch (err) {
    return { error: err.message || String(err) };
  }
}

const agree = (a, b) => a !== null && b !== null && Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));

/** a grey image as a strip, for the preview */
function greyCanvas(grid) {
  const bin = atob(grid.data);
  const c = h('canvas', { width: grid.nx, height: grid.ny });
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(grid.nx, grid.ny);
  for (let k = 0; k < grid.nx * grid.ny; k++) {
    const g = bin.charCodeAt(k);
    img.data[4 * k] = g; img.data[4 * k + 1] = g; img.data[4 * k + 2] = g; img.data[4 * k + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  c.style.width = '100%'; c.style.height = '60px'; c.style.objectFit = 'fill'; c.style.borderRadius = '6px'; c.style.imageRendering = 'pixelated';
  return c;
}

/** an image file as grey levels, at most 128 a side, row 0 at the top */
export async function readGrey(file, most = 128) {
  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise((ok, bad) => { const im = new Image(); im.onload = () => ok(im); im.onerror = () => bad(new Error('not an image this browser can read')); im.src = url; });
    const s = Math.min(1, most / Math.max(img.naturalWidth, img.naturalHeight));
    const nx = Math.max(1, Math.round(img.naturalWidth * s)), ny = Math.max(1, Math.round(img.naturalHeight * s));
    const c = document.createElement('canvas');
    c.width = nx; c.height = ny;
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0, nx, ny);
    const d = ctx.getImageData(0, 0, nx, ny).data;
    let bin = '';
    for (let k = 0; k < nx * ny; k++) bin += String.fromCharCode(Math.round(0.299 * d[4 * k] + 0.587 * d[4 * k + 1] + 0.114 * d[4 * k + 2]));
    return { nx, ny, data: btoa(bin) };
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** a vertical gradient of the formula's values across the feature, for the preview */
function formulaStrip(geom, f) {
  const c = h('canvas', { width: 48, height: 24 });
  try {
    const byId = byIdOf(geom);
    const b = itemOf(f, byId).bounds;
    const tree = parse(f.field.expr);
    const env = values(geom);
    const vals = [];
    for (let j = 0; j < 24; j++) for (let i = 0; i < 48; i++) {
      const x = b.left + (i + 0.5) / 48 * b.width, y = -(b.top + (j + 0.5) / 24 * b.height);
      vals.push(evaluate(tree, { ...env, x, y }));
    }
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(48, 24);
    const ramp = [[233, 227, 253], [142, 114, 238], [25, 25, 39]];
    vals.forEach((v, k) => {
      const t = hi > lo ? (lo > 0 ? Math.log(v / lo) / Math.log(hi / lo) : (v - lo) / (hi - lo)) : 0.5;
      const x = Math.max(0, Math.min(1, t)) * 2;
      const q = Math.min(1, Math.floor(x));
      const u = x - q;
      for (let ch = 0; ch < 3; ch++) img.data[4 * k + ch] = ramp[q][ch] + (ramp[q + 1][ch] - ramp[q][ch]) * u;
      img.data[4 * k + 3] = 255;
    });
    ctx.putImageData(img, 0, 0);
  } catch (err) { /* the input shows the error */ }
  c.style.width = '100%'; c.style.height = '60px'; c.style.borderRadius = '6px';
  return c;
}

/**
 * The coefficient section of a region: Constant (its material), Formula or Image.
 * ctx: {geom, commit, drawing, fields (the server's ranges by feature)}.
 */
export function coefficientSection(ctx, f) {
  const { geom } = ctx;
  const i = geom.features.findIndex((x) => x.id === f.id);
  const kind = f.field ? f.field.kind : 'constant';
  const ps = props();
  const prop = (f.field && f.field.prop) || (ps.includes('k') ? 'k' : ps[0]) || 'k';
  const sec = h('section', { class: 'subpanel' });
  const sym = { k: 'κ', sigma: 'σ', rho: 'ρ', cp: 'c_p' }[prop] || prop;
  sec.append(h('div', { style: { display: 'flex', alignItems: 'center', gap: '6px' } },
    h('span', { style: { fontWeight: 700, flex: 1 } }, `${f.name || f.id} · ${sym}${kind === 'constant' ? '' : ` · ${kind}`}`)));
  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Coefficient' });
  for (const [k, label, tip] of [['constant', 'Constant', 'The material\'s own value'], ['formula', 'Formula', 'A formula of x, y and the parameters; the solver sees it binned into at most 16 levels'],
    ['image', 'Image', 'A greyscale image mapped onto a range: black the minimum, white the maximum']]) {
    const b = h('button', { 'aria-pressed': String(kind === k), 'data-tip': tip }, label);
    b.addEventListener('click', async () => {
      if (k === kind) return;
      if (k === 'constant') { await ctx.commit([{ op: 'set', path: ['geometry', 'features', i, 'field'], value: null }], `${f.name || f.id}: constant ${sym}`); return; }
      if (k === 'formula') {
        const mats = { ...(family()?.materials || {}), ...(store.state?.case?.materials || {}) };
        const base = (mats[f.material] || {})[prop] ?? 1;
        await ctx.commit([{ op: 'set', path: ['geometry', 'features', i, 'field'], value: { kind: 'formula', prop, expr: `${fmt(base, 6)} * (1 + 0.5 * sin(pi * x / 0.1))` } }], `${f.name || f.id}: ${sym} formula`);
        return;
      }
      pickImage(ctx, f, i, prop);
    });
    seg.append(b);
  }
  sec.append(seg);
  if (kind === 'formula') {
    const input = h('input', { class: 'field mono', value: f.field.expr, spellcheck: 'false', 'aria-label': `Formula for ${sym}`, style: { width: '100%' } });
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') input.blur(); });
    input.addEventListener('change', async () => {
      const t = input.value.trim();
      const bad = formulaTrouble(t, geom, true);
      if (bad) { toast('error', `${sym} = ${t}: ${bad}.`); input.value = f.field.expr; return; }
      await ctx.commit([{ op: 'set', path: ['geometry', 'features', i, 'field', 'expr'], value: t }], `${f.name || f.id}: ${sym} = ${t}`);
    });
    sec.append(input, formulaStrip(geom, f));
    const mine = pageRange(geom, f, ctx.drawing);
    const theirs = (ctx.fields || {})[f.id];
    if (mine && mine.error) sec.append(h('div', { class: 'note err' }, mine.error));
    else if (mine) {
      sec.append(h('div', { style: { display: 'flex', justifyContent: 'space-between', fontSize: '12px' }, class: 'mono' },
        h('span', {}, `${sym} ${fmt(mine.min, 4)}`), h('span', {}, `${sym} ${fmt(mine.max, 4)}`)));
    }
    if (theirs && theirs.min !== null && mine && !mine.error) {
      const ok = agree(mine.min, theirs.min) && agree(mine.max, theirs.max) && mine.cells === theirs.cells;
      sec.append(h('div', { class: ok ? 'ok-text' : 'warn-text', style: { fontSize: '11.5px' },
        'data-tip': `The server evaluated the same formula at the same ${theirs.cells} cell centres: ${fmt(theirs.min, 6)} to ${fmt(theirs.max, 6)}, run as ${theirs.levels} materials.` },
      ok ? `✓ server: same range, ${theirs.cells} cells, ${theirs.levels} levels` : `server: ${fmt(theirs.min, 4)} to ${fmt(theirs.max, 4)} over ${theirs.cells} cells`));
    }
  } else if (kind === 'image') {
    const fl = f.field;
    sec.append(greyCanvas(fl.grid));
    const lo = h('input', { class: 'field num', value: fmt(fl.min, 6), style: { width: '70px' }, 'aria-label': 'black is' });
    const hi = h('input', { class: 'field num', value: fmt(fl.max, 6), style: { width: '70px' }, 'aria-label': 'white is' });
    const scale = h('select', { class: 'field', 'aria-label': 'Scale' }, h('option', { value: 'log', selected: fl.scale === 'log' || null }, 'log'), h('option', { value: 'linear', selected: fl.scale !== 'log' || null }, 'linear'));
    const apply = async () => {
      const a = Number(lo.value), b = Number(hi.value);
      if (!(Number.isFinite(a) && Number.isFinite(b) && b > a && (scale.value !== 'log' || a > 0))) { toast('error', 'The range needs min < max (and min above 0 on a log scale).'); return; }
      await ctx.commit([{ op: 'set', path: ['geometry', 'features', i, 'field'], value: { ...fl, min: a, max: b, scale: scale.value } }], `${f.name || f.id}: image range`);
    };
    for (const el of [lo, hi, scale]) el.addEventListener('change', apply);
    sec.append(h('div', { style: { display: 'flex', gap: '6px', alignItems: 'center', fontSize: '12px', flexWrap: 'wrap' } },
      h('span', { 'data-tip': 'The value black maps to' }, 'black'), lo, h('span', { 'data-tip': 'The value white maps to' }, 'white'), hi, scale));
    const theirs = (ctx.fields || {})[f.id];
    if (theirs && theirs.min !== null) sec.append(h('div', { class: 'muted', style: { fontSize: '11.5px' } }, `server: ${sym} ${fmt(theirs.min, 4)} to ${fmt(theirs.max, 4)}, ${theirs.levels} levels`));
    const again = h('button', { class: 'btn small', 'data-tip': 'Another image for this region' }, 'Image…');
    again.addEventListener('click', () => pickImage(ctx, f, i, prop));
    sec.append(again);
  }
  return sec;
}

function pickImage(ctx, f, i, prop) {
  const input = h('input', { type: 'file', accept: 'image/*' });
  input.addEventListener('change', async () => {
    const file = input.files && input.files[0];
    if (!file) return;
    try {
      const grid = await readGrey(file);
      const byId = byIdOf(ctx.geom);
      const b = itemOf(f, byId).bounds;
      const mats = { ...(family()?.materials || {}), ...(store.state?.case?.materials || {}) };
      const base = Number((mats[f.material] || {})[prop] ?? 1);
      await ctx.commit([{ op: 'set', path: ['geometry', 'features', i, 'field'], value: { kind: 'image', prop, min: base, max: 100 * base, scale: 'log', grid, box: [b.left, -b.bottom, b.right, -b.top] } }],
        `${f.name || f.id}: ${prop} from ${file.name}`);
    } catch (err) {
      toast('error', `${file.name}: ${err.message || err}`);
    }
  });
  input.click();
}

// -- scatter -----------------------------------------------------------------------------

const DEFAULTS = { count: 12, rmin: 0.01, rmax: 0.03, gap: 0.005, seed: 7, shape: 'circles', aspect: 8 };

/** the scatter form: the mockup's count, size, material, gap, shape, seed; Area and Scatter */
export function scatterSection(ctx, f = null) {
  const { geom } = ctx;
  const p = { ...DEFAULTS, ...(f ? f.params : ctx.state.scatterDraft || {}) };
  const mats = Object.keys({ ...(family()?.materials || {}), ...(store.state?.case?.materials || {}) });
  const sec = h('section', { class: 'subpanel' });
  sec.append(h('div', { style: { display: 'flex', alignItems: 'center' } }, h('span', { style: { fontWeight: 700, flex: 1 } }, 'Scatter inclusions'),
    h('span', { class: 'chipkind', 'data-tip': 'Random high-contrast inclusions, the same every time for a given seed: the Stage 0b test family' }, '0b')));
  const u = ctx.unit;
  const field = (label, val, tip) => h('input', { class: 'field mono', value: val, 'aria-label': label, 'data-tip': tip, style: { width: '100%' } });
  const count = field('count', String(p.count), 'How many to place; fewer are placed if the box is too full for the gap');
  const size = field('size', `${fmt(2 * p.rmin, 4)} – ${fmt(2 * p.rmax, 4)}`, `Diameter (circles) or length (channels), from – to, in ${u}`);
  const gap = field('min gap', fmt(p.gap, 4), `The least distance between two inclusions, in ${u}`);
  const seed = field('seed', String(p.seed), 'The same seed gives the same inclusions, byte for byte');
  const mat = h('select', { class: 'field', 'aria-label': 'material' }, ...mats.map((m) => h('option', { value: m, selected: (f ? f.material : ctx.state.scatterMaterial) === m || null }, m)));
  const shape = h('select', { class: 'field', 'aria-label': 'shape', 'data-tip': 'Circles; channels: straight strips, as long as the size and an eighth as wide; or both' },
    ...['circles', 'channels', 'both'].map((s) => h('option', { value: s, selected: p.shape === s || null }, s)));
  const lab = (t, el) => h('label', { class: 'stack' }, t, el);
  sec.append(h('div', { class: 'grid2' }, lab('count', count), lab(`size, ${u}`, size), lab('material', mat), lab(`min gap, ${u}`, gap), lab('shape', shape), lab('seed', seed)));
  const read = () => {
    const sz = size.value.split(/[–-]/).map((t) => Number(t.trim())).filter((t) => Number.isFinite(t));
    const out = { count: Math.round(Number(count.value)), rmin: sz[0] / 2, rmax: (sz[1] ?? sz[0]) / 2, gap: Number(gap.value), seed: Math.round(Number(seed.value)), shape: shape.value, aspect: Number(p.aspect) || 8 };
    if (!(out.count >= 1 && out.count <= 500 && out.rmin > 0 && out.rmax >= out.rmin && out.gap >= 0 && Number.isFinite(out.seed))) {
      toast('error', `Scatter: 1 to 500 inclusions, a size range from – to above 0, a gap of 0 or more, a whole-number seed.`);
      return null;
    }
    return out;
  };
  if (f) {
    const i = geom.features.indexOf(f);
    const apply = async () => {
      const q = read();
      if (!q) return;
      await ctx.commit([{ op: 'set', path: ['geometry', 'features', i, 'params'], value: q }, { op: 'set', path: ['geometry', 'features', i, 'material'], value: mat.value }], `${f.name || f.id}: scatter`);
    };
    for (const el of [count, size, gap, seed, mat, shape]) el.addEventListener('change', apply);
    const reseed = h('button', { class: 'btn', style: { flex: 1 }, 'data-tip': 'The next seed: another layout, as reproducible' }, 'Reseed');
    reseed.addEventListener('click', () => { seed.value = String(Math.round(Number(seed.value)) + 1); apply(); });
    sec.append(h('div', { style: { display: 'flex', gap: '6px' } }, reseed),
      h('div', { class: 'muted', style: { fontSize: '11.5px' } }, `${(f.items || []).length} placed in ${fmt(f.w, 3)} × ${fmt(f.h, 3)} ${u}`));
    return sec;
  }
  const area = h('button', { class: 'btn', style: { flex: 1 }, 'data-tip': 'Draw the box the inclusions go in' }, 'Area');
  area.addEventListener('click', () => { ctx.state.scatterDraft = read() || p; ctx.state.scatterMaterial = mat.value; ctx.armArea(); });
  const go = h('button', { class: 'btn primary', style: { flex: 2 } }, 'Scatter');
  go.addEventListener('click', () => {
    const q = read();
    if (!q) return;
    const box = ctx.state.scatterBox || defaultBox(geom);
    if (!box) { toast('info', 'Draw a domain first, or an Area for the inclusions.'); return; }
    makeScatter(ctx, q, mat.value, box);
  });
  sec.append(h('div', { style: { display: 'flex', gap: '6px' } }, area, go));
  if (ctx.state.scatterBox) sec.append(h('div', { class: 'muted', style: { fontSize: '11.5px' } }, `area ${fmt(ctx.state.scatterBox[2], 3)} × ${fmt(ctx.state.scatterBox[3], 3)} ${u}`));
  return sec;
}

function defaultBox(geom) {
  const dom = domainItem(geom, byIdOf(geom));
  if (!dom) return null;
  const b = dom.bounds;
  const m = 0.06 * Math.min(b.width, b.height);
  return [b.left + m, -b.bottom + m, b.width - 2 * m, b.height - 2 * m];
}

export async function makeScatter(ctx, params, material, box) {
  const g = ctx.geom;
  const id = nextId(g);
  const n = g.features.filter((x) => x.kind === 'scatter').length + 1;
  const f = blank('scatter', id, 'region', { name: `Inclusions ${n}`, material, x: box[0], y: box[1], w: box[2], h: box[3], params });
  ctx.state.scatterBox = null;
  await ctx.commit([{ op: 'insert', path: ['geometry', 'features'], value: f }], `scattered ${params.count} inclusions, seed ${params.seed}`);
}
