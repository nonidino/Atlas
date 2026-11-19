// Pieces (plan §5): the domain cut into four-sided pieces, each charted, certified and
// given an expert; the seams in their verdict's colour.  The server does the work
// (atlas/studio/pieces.py, piecejob.py): this page asks and draws.  A heat case gets
// the chart operator's pieces; another kind shows the windows its arms run on.

/* global paper */
import { store, subscribe, send, family } from '../store.js';
import { h, clear, cssVar, fmt, toast } from '../ui.js';
import { Stage, SWATCHES, P } from '../canvas.js';

let root = null;
let stage = null;
let els = {};
let selected = 0;
let tool = 'select';           // select, cut, merge, corner
let drag = null;               // a corner being dragged
let cutMode = 'auto';
const counts = { cols: 2, rows: 1, n_r: 3, n_th: 5, along: 0, split: 3 };
let resolution = 64;
let fittedFor = null;          // the case the view was last fitted to

const EXPERT_TOKEN = { classical: '--ex-c', H2r: '--ex-r', TL0: '--ex-t0', 'TL-s0': '--ex-tl', 'TL-s1': '--ex-tl', 'Sp-s0': '--ex-tl', 'Sp-s1': '--ex-tl' };
const EXPERT_SHORT = { classical: 'C', H2r: 'R', TL0: 'T0', 'TL-s0': 'TL', 'TL-s1': 'TL', 'Sp-s0': 'S′', 'Sp-s1': 'S′' };
const VERDICT = { admit: ['#00A9A5', 'admitted', 'Admitted: conforming, the two pieces\' nodes coincide (Theorem 2)'],
  'admit-uncertified': ['#F5A524', 'uncertified', 'Admitted without a certificate: a mortar between pieces whose nodes differ'],
  refused: ['#E5484D', 'refused', 'Refused: the seam\'s ports do not match'] };
const CUTTERS = {
  grid: ['Grid', 'A rectangle on a grid, cut again at every material interface'],
  polar: ['Polar', 'An annular sector at equal log r and angle: exact conformal charts'],
  strip: ['Strip', 'A four-sided channel at equal harmonic coordinate; conformal charts'],
  insert: ['Insert', 'A plate with one round insert: the O-H layout'],
};

// heat conduction (CO-D) and the coolant's conjugate heat (CO-T: atlas/studio/cot.py)
const isHeat = () => ['conduction-2d', 'conjugate-heat-2d'].includes(store.state?.meta?.family);
const isCOT = () => summary()?.physics === 'CO-T';
const summary = () => store.pieces?.summary || null;
const busy = () => ['starting', 'working'].includes(store.pieces?.job?.status);

async function act(type, args = {}) {
  const r = await send(type, args);
  if (!r.ok) return r;
  return r;
}

// -- drawing --------------------------------------------------------------------------
function draw() {
  if (!stage || !store.state) return;
  const d = store.state.drawing;
  // a new case, or a canvas that had no size when it was last fitted: fit again
  const key = `${store.state.meta.serial}:${d.width}x${d.height}`;
  if (key !== fittedFor && stage.view.viewSize.width > 0 && stage.isShown()) { fittedFor = key; stage.fitted = false; }
  stage.clear();
  stage.activate();
  if (!isHeat()) { drawWindows(d); return; }
  const s = summary();
  const dom = d.outline ? [d.outline] : [[[0, 0], [d.width, 0], [d.width, d.height], [0, d.height]]];
  stage.main.addChild(stage.rings(dom, { fill: cssVar('--shape-fill'), stroke: cssVar('--shape-stroke'), width: 1.2 }));
  (d.holes || []).forEach((ring) => stage.main.addChild(stage.rings([ring], { fill: cssVar('--canvas'), stroke: cssVar('--shape-stroke') })));
  if (s) {
    const z = stage.view.zoom;
    s.pieces.forEach((p, i) => {
      const fill = cssVar(EXPERT_TOKEN[p.expert] || '--ex-c');
      const bad = !p.certificate.ok;
      stage.main.addChild(stage.rings([p.outline], { fill: bad ? '#FBDADB' : fill, fillAlpha: 0.9, stroke: null }));
      for (const l of p.lines) stage.main.addChild(stage.line(l, { stroke: cssVar('--muted'), width: 0.5, alpha: 0.45 }));
      stage.main.addChild(stage.rings([p.outline], { fill: null, stroke: i === selected ? cssVar('--accent') : cssVar('--shape-stroke'), width: i === selected ? 2.4 : 1.1 }));
      const t = new paper.PointText({ point: P(p.centre[0], p.centre[1]).add(new paper.Point(0, 4 / z)), content: EXPERT_SHORT[p.expert] || '?',
        justification: 'center', fontFamily: 'JetBrains Mono, monospace', fontSize: 11 / z, fontWeight: 'bold', fillColor: bad ? '#9B1C20' : '#191927', insert: false });
      stage.over.addChild(t);
    });
    for (const sm of s.seams || []) {
      const v = VERDICT[sm.verdict];
      if (!v) continue;
      stage.over.addChild(stage.line(sm.points, { stroke: v[0], width: 2.2 }));
    }
  }
  stage.setBounds(0, 0, d.width, d.height);
  stage.update();
}

function drawWindows(d) {
  const dom = d.outline ? [d.outline] : [[[0, 0], [d.width, 0], [d.width, d.height], [0, d.height]]];
  stage.main.addChild(stage.rings(dom, { fill: cssVar('--shape-fill'), stroke: cssVar('--shape-stroke'), width: 1.4 }));
  d.holes.forEach((ring) => stage.main.addChild(stage.rings([ring], { fill: cssVar('--canvas'), stroke: cssVar('--shape-stroke') })));
  d.windows.forEach((w, i) => {
    const c = SWATCHES[i % SWATCHES.length];
    stage.main.addChild(stage.rings(w.rings, { fill: c, fillAlpha: 0.18, stroke: c, width: 1.6, dash: [5, 3] }));
  });
  stage.setBounds(0, 0, d.width, d.height);
  stage.update();
}

// -- the canvas's tools ------------------------------------------------------------------
function pieceAt(w) {
  const s = summary();
  if (!s) return -1;
  stage.activate();
  for (const [i, p] of s.pieces.entries()) {
    const path = new paper.Path({ segments: p.outline.map(([x, y]) => P(x, y)), closed: true, insert: false });
    if (path.contains(P(w.x, w.y))) return i;
  }
  return -1;
}

function nearestSide(p, w) {
  let best = null;
  p.sides_xy.forEach((pts, k) => {
    for (const q of pts) {
      const d = Math.hypot(q[0] - w.x, q[1] - w.y);
      if (!best || d < best.d) best = { side: k, d };
    }
  });
  return best;
}

function worldOf(e) {
  const r = stage.canvas.getBoundingClientRect();
  return stage.viewToWorld(new paper.Point(e.clientX - r.left, e.clientY - r.top));
}

function cornerNear(w) {
  const s = summary();
  if (!s) return null;
  const tol = 9 / stage.view.zoom;
  let best = null;
  for (const p of s.pieces) {
    for (const side of p.sides_xy) {
      const q = side[0];
      const d = Math.hypot(q[0] - w.x, q[1] - w.y);
      if (d < tol && (!best || d < best.d)) best = { x: q[0], y: q[1], d };
    }
  }
  return best;
}

function onDown(e) {
  if (tool !== 'corner' || !isHeat() || busy()) return;
  const c = cornerNear(worldOf(e));
  if (!c) return;
  drag = { from: c, to: c };
  stage.canvas.setPointerCapture(e.pointerId);
}

function onMove(e) {
  if (!drag) return;
  drag.to = worldOf(e);
  draw();
  stage.activate();
  stage.over.addChild(stage.line([[drag.from.x, drag.from.y], [drag.to.x, drag.to.y]], { stroke: cssVar('--accent'), width: 2, dash: [4, 3] }));
  const dot = new paper.Path.Circle({ center: P(drag.to.x, drag.to.y), radius: 5 / stage.view.zoom, insert: false });
  dot.fillColor = cssVar('--accent');
  stage.over.addChild(dot);
  stage.update();
}

async function onUp() {
  if (!drag) return;
  const d = drag;
  drag = null;
  if (Math.hypot(d.to.x - d.from.x, d.to.y - d.from.y) * stage.view.zoom < 3) { draw(); return; }
  await act('pieces.corner', { old: [d.from.x, d.from.y], new: [d.to.x, d.to.y] });
}

async function onCanvas(e) {
  if (!isHeat() || busy() || tool === 'corner') return;
  const w = worldOf(e);
  const i = pieceAt(w);
  if (i < 0) return;
  const s = summary();
  if (tool === 'select') { selected = i; draw(); renderRight(); return; }
  const ns = nearestSide(s.pieces[i], w);
  if (tool === 'cut') {
    await act('pieces.split', { piece: i, side: ns.side, at: [w.x, w.y] });
    return;
  }
  if (tool === 'merge') {
    const cid = (s.pieces[i].chords || {})[ns.side];
    if (!cid) { toast('info', 'Click on a cut you made by hand: only those can be merged.'); return; }
    await act('pieces.merge', { chord: cid });
  }
}

// -- the panels ---------------------------------------------------------------------------
function seg(label, options, current, onPick, tipOf = () => null) {
  const g = h('div', { class: 'seg', role: 'group', 'aria-label': label });
  for (const [v, text] of options) {
    const b = h('button', { 'aria-pressed': String(v === current), 'data-tip': tipOf(v) }, text);
    b.addEventListener('click', () => onPick(v));
    g.append(b);
  }
  return g;
}

function kv(k, v, tip = null, cls = '') {
  return h('div', { class: 'kvrow' }, tip ? h('abbr', { 'data-tip': tip }, k) : h('span', {}, k), h('b', { class: `mono ${cls}` }, v));
}

function renderLeft() {
  const box = clear(els.left);
  const s = summary();
  box.append(h('div', { class: 'label' }, 'Chart method'));
  const cur = s && s.pieces.every((p) => p.method === s.pieces[0].method) ? s.pieces[0].method : null;
  box.append(seg('Chart method', [['conformal', 'Conformal'], ['laplacian', 'Laplacian'], ['tfi', 'TFI']], cur,
    (m) => act('pieces.method', { method: m }),
    (m) => ({ conformal: 'Conformal map: preserves angles; smooth, but cells crowd near sharp corners. A numerical conformal map (new), or the closed form where there is one',
      laplacian: 'Laplacian (harmonic) coordinates: Winslow\'s chart, solved with the sides as data; robust on curved pieces',
      tfi: 'Transfinite interpolation (Coons patch): blends the four sides; cheapest, can fold on strongly curved pieces' }[m])));
  if (s && cur !== 'own') {
    const own = h('button', { class: 'btn small', 'data-tip': 'Each piece back to the chart its cut made (the benchmarks\' own)' }, 'The cut\'s own charts');
    own.addEventListener('click', () => act('pieces.method', { method: 'own' }));
    box.append(own);
  }
  if (s) {
    box.append(kv('min det J', s.min_detJ_rel === null ? '–' : fmt(s.min_detJ_rel, 3), 'Smallest Jacobian determinant over all pieces\' cell corners, relative to its mean. Above 0 means no chart folds over itself.', s.min_detJ_rel !== null && s.min_detJ_rel <= 0 ? 'err-text' : ''),
      kv('max skew', s.max_skew === null ? '–' : `${Math.round(s.max_skew)}°`, 'Largest departure of a cell corner from 90°'),
      kv('certified', `${s.certified[0]} / ${s.certified[1]}`, 'Proposition 3: det J above 0 at every cell corner and a simple, positively oriented boundary', s.certified[0] < s.certified[1] ? 'err-text' : 'ok-text'));
  }
  box.append(h('div', { class: 'label' }, 'Cutting'));
  box.append(seg('How the domain is cut', [['auto', 'Automatic'], ['hand', 'By hand']], cutMode, (m) => { cutMode = m; tool = m === 'hand' ? 'cut' : 'select'; renderLeft(); renderBar(); }));
  if (cutMode === 'auto') {
    const res = h('select', { class: 'field', 'aria-label': 'Resolution n' }, ...[32, 64, 128].map((n) => h('option', { value: n, selected: n === resolution || null }, `n = ${n}`)));
    res.addEventListener('change', () => { resolution = Number(res.value); });
    const num = (key, label, tip) => {
      const i = h('input', { class: 'field num', value: String(counts[key] || ''), style: { width: '48px' }, 'aria-label': label, placeholder: key === 'along' ? 'auto' : '' });
      i.addEventListener('change', () => { counts[key] = Math.max(0, Math.round(Number(i.value) || 0)); });
      return h('label', { class: 'row', 'data-tip': tip }, label, i);
    };
    box.append(h('label', { class: 'row', 'data-tip': 'Nodes per piece side: the experts were trained at 32, 64 and 128' }, 'resolution', res),
      num('cols', 'grid columns', 'A rectangle: pieces across'), num('rows', 'grid rows', 'A rectangle: pieces up'),
      num('n_r', 'polar, radial', 'An annular sector: pieces across the radius'), num('n_th', 'polar, around', 'An annular sector: pieces around'),
      num('along', 'strip pieces', 'A channel: pieces along it; empty for its module (a piece near a square)'),
      num('split', 'insert split', 'A plate with an insert: each ring cut into this many'));
    const go = h('button', { class: 'btn primary', disabled: busy() || null, 'data-tip': 'Cut the domain automatically: a rectangle on a grid and at material interfaces; a bend in polar strips; a channel at equal harmonic coordinate; a plate with a round insert in the O-H layout. Says so when it cannot.' }, 'Cut automatically');
    go.addEventListener('click', () => {
      const params = {};
      for (const k of ['cols', 'rows', 'n_r', 'n_th', 'split']) if (counts[k]) params[k] = counts[k];
      if (counts.along) params.along = counts.along;
      act('pieces.cut', { params, n: resolution });
    });
    box.append(go);
  } else {
    box.append(h('div', { class: 'muted', style: { fontSize: '12px' } }, 'Cut: click a side of a piece; the cut runs across it and on through its neighbours. Merge: click a cut you made. Corner: drag an inside corner.'));
    for (const c of s?.chords || []) {
      const m = h('button', { class: 'btn small', 'data-tip': `Remove the cut ${c}: each pair across it becomes one piece` }, `Merge ${c}`);
      m.addEventListener('click', () => act('pieces.merge', { chord: c }));
      box.append(m);
    }
  }
  box.append(h('div', { class: 'label' }, 'Experts'));
  const rows = store.pieces?.experts || [];
  const n = {};
  for (const p of s?.pieces || []) n[p.expert] = (n[p.expert] || 0) + 1;
  const prices = store.pieces?.prices || {};
  const name = (k) => ({ classical: 'classical', 'Sp-s0': 'S′-s0', 'Sp-s1': 'S′-s1' }[k] || k);
  const priceText = (k) => {
    const p = prices[k];
    if (!p) return k === 'H2r' ? 'never dearer than the classical solve (one solve per class of squares)' : 'no registered record prices it';
    return p.record ? `${fmt(p.value, 2)}× the classical solve (${p.record}, its mean over ${p.set})` : '1× by definition';
  };
  for (const r of rows) {
    box.append(h('div', { class: 'kvrow', style: { opacity: r.offered ? 1 : 0.5 } },
      h('span', { style: { display: 'flex', gap: '6px', alignItems: 'center' } }, h('span', { class: 'sw', style: { background: cssVar(EXPERT_TOKEN[r.key] || '--ex-c'), width: '12px', height: '12px', borderRadius: '3px' } }),
        h('abbr', { 'data-tip': r.offered ? `${r.tip}. Price: ${priceText(r.key)}` : `${r.label}: ${r.why}` }, r.key === 'Sp-s0' ? 'S′-s0' : (r.key === 'Sp-s1' ? 'S′-s1' : r.label.split(',')[0]))),
      h('span', { class: 'mono' }, String(n[r.key] || 0))));
  }
  const order = rows.filter((r) => r.offered && prices[r.key]).sort((a, b) => prices[a.key].value - prices[b.key].value)
    .map((r) => `${name(r.key)} ${fmt(prices[r.key].value, 2)}×`).join(', ');
  const best = h('button', { class: 'btn', disabled: !s || busy() || null,
    'data-tip': `Give every piece the cheapest expert admitted on it (by its card, or H2r's premise), by the registered records' prices over the classical solve: ${order || 'classical 1×'}. An expert no record prices is not chosen.` }, 'Best admitted for all');
  best.addEventListener('click', () => act('pieces.expert', { expert: 'best' }));
  const chk = h('button', { class: 'btn', disabled: !s || busy() || null, 'data-tip': 'Couple the pieces with the classical expert and compare with the whole domain solved at once (host.monolithic): round-off when every seam conforms and each side keeps its whole trace' }, 'Check the coupling');
  chk.addEventListener('click', () => act('pieces.check'));
  box.append(best, chk);
  const c = s?.check;
  if (c) {
    const f = (x) => (x === null || x === undefined ? '–' : x.toExponential(1));
    box.append(h('div', { class: 'subpanel' },
      kv('whole traces', f(c.full.max_diff_rel), 'Every side\'s whole trace (n port modes): the coupled answer against the monolithic solve, the largest difference over the range', c.full.max_diff_rel < 1e-9 ? 'ok-text' : 'warn-text'),
      kv('16 modes', f(c.ports16.max_diff_rel), 'The experts\' 16 port modes per side: the truncation floor every expert stands on'),
      ...(isCOT() ? [kv('outlet, 16 modes', f(c.ports16.outlet_bulk_err_rel), 'The outlet\'s mixing-cup temperature with 16 port modes against the monolithic solve, over the field\'s range'),
        kv('balance, 16 modes', f(c.ports16.balance_rel), 'The heat the 16-mode coupled field leaves unaccounted for, over the heat throughput: zero to round-off with whole traces')] : [])));
  }
}

function renderBar() {
  const bar = clear(els.bar);
  const s = summary();
  const job = store.pieces?.job;
  const seams = (s?.seams || []).filter((x) => x.pieces.length === 2).length;
  bar.append(h('span', {}, s ? `${s.pieces.length} pieces · ${seams} seams` : 'Not cut yet'));
  if (job && ['starting', 'working'].includes(job.status)) bar.append(h('span', { class: 'busy' }, `${job.label[0].toUpperCase() + job.label.slice(1)}…`));
  if (store.pieces?.stale) bar.append(h('span', { class: 'warn-text', 'data-tip': 'The domain, its materials or boundaries changed since the cut' }, 'stale: cut again'));
  bar.append(h('span', { class: 'spacer' }));
  for (const [t, label, tip] of [['cut', 'Cut', 'Click a side of a piece: the cut runs across it and on through its neighbours'], ['merge', 'Merge', 'Click a cut you made by hand: each pair across it becomes one piece'],
    ['corner', 'Corner', 'Drag an inside corner where straight cuts meet; the pieces around it follow']]) {
    const b = h('button', { class: 'toggle', 'aria-pressed': String(tool === t), 'data-tip': tip, disabled: !s || null }, label);
    b.addEventListener('click', () => { tool = tool === t ? 'select' : t; cutMode = tool === 'select' ? cutMode : 'hand'; renderBar(); renderLeft(); stage.canvas.style.cursor = tool === 'select' ? 'default' : 'crosshair'; });
    bar.append(b);
  }
}

function renderFoot() {
  const foot = clear(els.foot);
  for (const [k, [c, label, tip]] of Object.entries(VERDICT)) {
    foot.append(h('span', { style: { display: 'flex', alignItems: 'center', gap: '5px' }, 'data-tip': tip },
      h('i', { style: { display: 'inline-block', width: '16px', height: '3px', background: c, borderRadius: '2px' } }), label));
  }
  const s = summary();
  if (s?.refused) foot.append(h('span', { class: 'warn-text', 'data-tip': s.refused, style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0 } }, 'seams unknown: ' + s.refused));
  foot.append(h('span', { class: 'spacer' }));
  const job = store.pieces?.job;
  if (job?.status === 'failed') foot.append(h('span', { class: 'err-text', 'data-tip': job.error }, 'the last action failed'));
}

function renderRight() {
  const box = clear(els.right);
  const s = summary();
  if (!s) {
    box.append(h('div', { style: { fontSize: '17px', fontWeight: 700 } }, 'Pieces'),
      h('div', { class: 'muted' }, 'Cut the domain into four-sided pieces: each gets a chart onto the unit square, a certificate, and an expert.'));
    if (s === null && store.pieces?.job?.status === 'failed') box.append(h('div', { class: 'note err' }, store.pieces.job.error));
    return;
  }
  if (selected >= s.pieces.length) selected = 0;
  const p = s.pieces[selected];
  box.append(h('div', { style: { display: 'flex', alignItems: 'baseline', gap: '8px' } },
    h('abbr', { style: { fontSize: '17px', fontWeight: 700 }, 'data-tip': `Piece ${selected + 1}, in the cut's order` }, `P${selected + 1}`),
    h('span', { class: 'muted', style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, `${p.label} · ${p.material}`)));
  const ce = p.certificate;
  box.append(h('div', { class: 'label' }, h('abbr', { 'data-tip': 'Proposition 3: a chart with det J above 0 at every cell corner and a simple, positively oriented boundary is a valid map of the square onto the piece' }, 'Certificate')));
  box.append(kv('min det J', fmt(ce.min_detJ_rel, 3), 'Smallest Jacobian determinant over this piece\'s cell corners, relative to its mean', ce.ok ? '' : 'err-text'),
    kv('boundary simple', ce.simple ? '✓' : '✗', null, ce.simple ? 'ok-text' : 'err-text'),
    kv(isCOT() ? 'in CO-T envelope' : 'in card envelope', p.envelope.refusals.length ? '✗' : '✓', p.envelope.refusals.length ? `Outside the ${isCOT() ? 'CO-T envelope' : 'CO-D card'}: ${p.envelope.refusals.join('; ')}` : (isCOT() ? 'Inside the CO-T envelope: a certified chart, a Péclet number at most the cap of 200 per piece, and a cell Péclet number at most 2 (Galerkin does not oscillate)' : 'Inside the shapes and coefficients the expert\'s card declares'), p.envelope.refusals.length ? 'warn-text' : 'ok-text'));
  if (p.flow) box.append(kv('Pe', fmt(p.flow.pe, 3), 'The piece\'s Péclet number |u|L/α along the flow (n times its largest cell Péclet number); the owner\'s cap is 200 per piece: cut a faster channel into shorter pieces', p.flow.pe > 200 ? 'warn-text' : ''),
    kv('cell Pe', fmt(p.flow.cell_pe, 3), 'The largest cell Péclet number: Galerkin oscillates above 2, so a faster piece needs a finer grid', p.flow.cell_pe > 2 ? 'warn-text' : ''));
  if (!ce.ok) box.append(h('div', { class: 'note err' }, `Refused: ${ce.reason}. min det J ${fmt(ce.min_detJ_rel, 3)}: try the Laplacian or conformal chart, or cut the piece.`));
  if (p.skew !== null) box.append(kv('max skew', `${Math.round(p.skew)}°`, 'Largest departure of a cell corner from 90°'));
  box.append(h('div', { class: 'label' }, 'Chart'));
  const ms = h('select', { class: 'field', 'aria-label': 'This piece\'s chart method' },
    h('option', { value: 'own', selected: p.method === 'own' || null }, `the cut's own (${p.own})`),
    ...[['conformal', 'Conformal'], ['laplacian', 'Laplacian'], ['tfi', 'TFI']].map(([v, l]) => h('option', { value: v, selected: p.method === v || null }, l)));
  ms.addEventListener('change', () => act('pieces.method', { method: ms.value, pieces: [selected] }));
  box.append(ms, h('div', { class: 'muted', style: { fontSize: '11.5px' } }, `made as ${p.chart}`));
  box.append(h('div', { class: 'label' }, 'Expert'));
  const g = h('div', { class: 'seg wrap', role: 'group', 'aria-label': 'Expert for this piece' });
  for (const r of store.pieces?.experts || []) {
    if (!r.offered) continue;
    const why = p.admitted[r.key];
    const b = h('button', { 'aria-pressed': String(p.expert === r.key), disabled: why ? true : null, 'data-tip': why ? `${r.key}: ${why}` : r.tip },
      r.key === 'Sp-s0' ? 'S′-s0' : (r.key === 'Sp-s1' ? 'S′-s1' : r.key === 'classical' ? 'Classical' : r.key));
    b.addEventListener('click', () => act('pieces.expert', { expert: r.key, pieces: [selected] }));
    g.append(b);
  }
  box.append(g);
  box.append(h('div', { class: 'label' }, 'Sides'));
  p.sides.forEach((sd, k) => {
    const to = sd.to === null ? (sd.kind === 'dirichlet' ? 'fixed temperature' : sd.kind) : `P${sd.to + 1}`;
    box.append(h('div', { class: 'kvrow' }, h('span', {}, `${k} · ${to}`),
      sd.to === null ? h('span', { class: 'muted' }, sd.flow && ['inlet', 'outlet'].includes(sd.flow) ? sd.flow : 'boundary') : h('span', {}, h('abbr', { 'data-tip': sd.kind === 'THERM+ADVEC' ? 'Thermal and advective ports: the flow crosses this seam, so temperature, the diffusive heat flux and the heat the flow carries (ADVEC) are exchanged' : 'Thermal port: temperature and heat flux are exchanged across the side' }, sd.kind === 'THERM+ADVEC' ? 'THERM+ADVEC' : 'THERM'),
        sd.conforming ? h('b', { class: 'ok-text' }, ' ✓') : h('b', { class: 'warn-text', 'data-tip': 'A mortar: the nodes differ' }, ' ~'))));
  });
  for (const n of s.notes || []) box.append(h('div', { class: 'muted', style: { fontSize: '11.5px' } }, n));
}

function render() {
  if (!root || !store.state) return;
  const heat = isHeat();
  els.left.hidden = !heat;
  els.right.hidden = !heat;
  if (!heat) {
    clear(els.bar).append(h('span', { class: 'muted' }, `${family()?.label || 'This kind'} runs on windows; chart-operator pieces are built for heat conduction (CO-D) and the coolant's conjugate heat (CO-T).`));
    clear(els.foot);
    draw();
    return;
  }
  renderLeft(); renderBar(); renderFoot(); renderRight(); draw();
}

export function mount(el) {
  root = el;
  const canvas = h('canvas', { 'aria-label': 'The pieces' });
  els.left = h('aside', { class: 'panel scroll', style: { width: '232px', flex: 'none', gap: '6px' } });
  els.right = h('aside', { class: 'panel scroll', style: { width: '244px', flex: 'none', gap: '6px' } });
  els.bar = h('div', { class: 'bar' });
  els.foot = h('div', { class: 'foot' });
  el.append(h('div', { class: 'work' }, els.left, h('main', { class: 'stage' }, els.bar, h('div', { class: 'canvas-wrap' }, canvas), els.foot), els.right));
  stage = new Stage(canvas);
  canvas.addEventListener('click', onCanvas);
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  window.addEventListener('themechange', () => { stage.drawGrid(); draw(); });
  window.atlas = Object.assign(window.atlas || {}, { pieces: { stage, world: (x, y) => { const r = canvas.getBoundingClientRect(); const q = stage.view.projectToView(P(x, y)); return [r.left + q.x, r.top + q.y]; } } });
  let asked = false;
  subscribe((what) => {
    if (what === 'state' || what === 'pieces' || (what === 'tab' && store.tab === 'pieces')) {
      if (what === 'tab' && !asked) { asked = true; send('pieces.state'); }
      render();
    }
  });
}
