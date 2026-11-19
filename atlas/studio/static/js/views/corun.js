// Run on the pieces (plan §6): Single and Compare.  Each window is the full domain (one
// solve, no pieces) or the chart operator (the pieces, an expert each: as assigned on
// Pieces, or one expert on every piece).  The server runs them in turns
// (atlas/studio/corun.py); this page lays out the windows, the step slider, the views and
// the cards.  The price is a ratio; no raw seconds reach the page.

import { store, send } from '../store.js';
import { h, clear, fmt, toast } from '../ui.js';
import { fieldCanvas } from './run.js';

const state = { mode: 'compare', view: 'field', same: true, k: null,
  A: { method: 'full', solver: 'classical' }, B: { method: 'chart', solver: 'TL-s0' } };
let frame = null;            // the fields of the step shown

const sci = (x) => (x === null || x === undefined || !Number.isFinite(x) ? '–' : (Math.abs(x) < 1e-2 || Math.abs(x) >= 1e4 ? x.toExponential(1) : fmt(x, 3)));

function solverOptions(win) {
  const rows = (store.pieces?.experts || []).filter((r) => r.offered);
  if (state[win].method === 'full') return [['classical', 'Classical FE']];
  const name = (k) => ({ classical: 'Classical', 'Sp-s0': 'S′-s0', 'Sp-s1': 'S′-s1' }[k] || k);
  return [['mixed', 'Mixed (as assigned)'], ...rows.map((r) => [r.key, name(r.key)])];
}

function windowHead(win, run) {
  const w = state[win];
  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': `Method, ${win}` });
  for (const [m, label, tip] of [['full', 'Full', 'One solve over the whole domain: the pieces merged into one mesh (host.monolithic)'], ['chart', 'Chart op.', 'Chart operator: the pieces, an expert each, coupled by the host']]) {
    const b = h('button', { 'aria-pressed': String(w.method === m), 'data-tip': tip, disabled: run || null }, label);
    b.addEventListener('click', () => { w.method = m; if (m === 'full') w.solver = 'classical'; else if (w.solver === 'classical') w.solver = 'mixed'; renderAll(); });
    seg.append(b);
  }
  const opts = solverOptions(win);
  if (!opts.some(([v]) => v === w.solver)) w.solver = opts[0][0];       // e.g. CO-T: no TL-s0
  const sel = h('select', { class: 'field', 'aria-label': `Solver, ${win}`, disabled: run || null, style: { minWidth: 0, maxWidth: '150px' } },
    ...opts.map(([v, l]) => h('option', { value: v, selected: v === w.solver || null }, l)));
  sel.addEventListener('change', () => { w.solver = sel.value; });
  const n = store.pieces?.summary?.n;
  const res = h('span', { class: 'muted mono', 'data-tip': 'The resolution of the cut (set on Pieces): every piece and the merged mesh share it' }, n ? `n ${n}` : '');
  return h('div', { class: 'winhead' }, h('b', {}, win), seg, sel, res);
}

function drawField(box, f, lo, hi, diverging = false) {
  clear(box);
  if (!f) { box.append(h('div', { class: 'empty' }, 'Nothing has run yet.')); return; }
  const cv = fieldCanvas(f, diverging, lo, hi);
  cv.style.width = '100%'; cv.style.height = '100%'; cv.style.objectFit = 'contain';
  box.append(cv, h('div', { class: 'scale mono' }, diverging ? `±${sci(Math.max(Math.abs(f.min), Math.abs(f.max)))}` : `${fmt(lo ?? f.min, 4)} … ${fmt(hi ?? f.max, 4)}`));
}

function panel(win, run) {
  const cr = store.corun;
  const sec = h('section', { class: 'cowin' }, windowHead(win, run));
  const box = h('div', { class: 'cofield' });
  sec.append(box);
  const fr = frame;
  if (!fr) { drawField(box, null); return sec; }
  const ref = state.mode === 'compare' ? 'B' : 'full domain';
  if (state.view === 'difference' && win === (state.mode === 'compare' ? 'B' : 'A')) {
    drawField(box, fr.difference, null, null, true);
  } else if (state.view === 'piece' && win === (state.mode === 'compare' ? 'B' : 'A')) {
    clear(box);
    const acc = cr?.result?.accuracy || null;
    const per = acc?.per_piece || [];
    const labels = cr?.result?.pieces || [];
    const top = Math.max(...per, 1e-300);
    const list = h('div', { class: 'perpiece' });
    per.forEach((e, i) => {
      list.append(h('div', { class: 'ppr', 'data-tip': `${labels[i]?.label || `piece ${i + 1}`}: energy error ${sci(e)} against ${state.mode === 'compare' ? 'A' : 'the full domain'} (${labels[i]?.expert || ''})` },
        h('span', { class: 'mono' }, `P${i + 1}`), h('span', { class: 'bar' }, h('i', { style: { width: `${Math.max(2, 100 * e / top)}%` } })), h('span', { class: 'mono' }, sci(e))));
    });
    if (!per.length) list.append(h('div', { class: 'empty' }, 'The errors per piece come when the run ends.'));
    box.append(list);
  } else if (state.view === 'seams' && win === (state.mode === 'compare' ? 'B' : 'A')) {
    clear(box);
    const sm = cr?.result?.seams;
    box.append(h('div', { class: 'placeholder' }, h('h2', {}, sm ? `largest jump in T: ${sci(sm.max_jump)}` : '–'),
      h('div', { class: 'muted' }, sm ? `over ${sm.count} seams: the host's coupled field shares each seam's trace, so a jump is round-off` : 'When the run ends.')));
  } else {
    const f = win === 'A' ? fr.A : (state.mode === 'compare' ? fr.B : fr['full domain']);
    let lo = null, hi = null;
    if (state.same && fr.A && (fr.B || fr['full domain'])) {
      const o = fr.B || fr['full domain'];
      lo = Math.min(fr.A.min, o.min); hi = Math.max(fr.A.max, o.max);
    }
    drawField(box, f, lo, hi);
  }
  if (win === 'B' || (state.mode === 'single' && win === 'A' && false)) void ref;
  return sec;
}

function cards() {
  const cr = store.corun;
  const r = cr?.result;
  const side = h('aside', { class: 'panel scroll', style: { width: '232px', flex: 'none', gap: '10px' } });
  const sec = (title, ...kids) => h('section', { class: 'subpanel' }, h('div', { class: 'label' }, title), ...kids);
  const row = (k, v, tip = null, cls = '') => h('div', { class: 'kvrow' }, tip ? h('abbr', { 'data-tip': tip }, k) : h('span', {}, k), h('b', { class: `mono ${cls}` }, v));
  if (state.mode === 'compare') {
    const p = r?.price;
    side.append(sec(h('abbr', { 'data-tip': 'B\'s cost over A\'s, timed in turns on this machine (the median of each). Below 1, B is cheaper.' }, 'Price B / A'),
      h('div', { class: 'big mono' }, p?.ratio ? `${fmt(p.ratio, 3)}×` : '–'),
      h('div', { class: 'muted', style: { fontSize: '12px' } }, h('abbr', { 'data-tip': 'A/A check: the same arm timed again and again. Its spread is the timing noise; a ratio inside it means no difference.' }, 'A/A'),
        p?.aa_spread ? ` spread ${fmt(100 * (p.aa_spread - 1), 2)}% over ${p.turns} turns` : ' –'),
      r?.machine ? h('div', { class: 'muted', style: { fontSize: '11px' }, 'data-tip': (r.machine.other_python || []).map((x) => x.command || x.name).join('; ') || 'no other Python processes' },
        `${r.machine.power?.ac === false ? 'on battery' : (r.machine.power?.ac ? 'on AC' : 'power not read')}${r.threads?.torch ? ` · torch ${r.threads.torch} threads` : ''}`) : null));
  }
  const acc = r?.accuracy;
  const vs = state.mode === 'compare' ? 'B vs A' : 'vs the full domain';
  const worst = acc?.per_piece?.length ? Math.max(...acc.per_piece) : null;
  side.append(sec(vs,
    row('energy, all', sci(acc?.energy_rel), 'Relative energy-norm error over every piece, against the reference window'),
    row('energy, worst piece', sci(worst), 'Largest relative energy-norm error over the pieces'),
    row('max error', sci(acc?.max_err), 'Largest nodal difference, in the field\'s units'),
    row('seam mismatch', sci(r?.seams?.max_jump), 'Largest jump in T across a seam'),
    row('balance', r?.balance ? sci(Math.max(...Object.values(r.balance).filter((x) => x !== null))) : '–', 'Net heat through the boundary over the heat in: zero for a conserving steady solution'),
    ...(r?.physics === 'CO-T' ? [row('outlet T', r.outlet?.bulk != null ? `${fmt(r.outlet.bulk, 5)} K` : '–', `The outlet's mixing-cup temperature in ${state.mode === 'compare' ? 'B' : 'this window'}; the reference's is ${r.outlet?.reference != null ? fmt(r.outlet.reference, 5) : '–'} K`),
      row('outlet error', sci(r.outlet?.err_over_range), 'The mixing-cup temperature\'s difference over the field\'s range'),
      row('ADVEC seams', String(r.seams?.advective ?? '–'), 'Seams the flow crosses: they exchange the heat it carries as well as T and the diffusive flux')] : [])));
  const chips = h('div', { class: 'chips' });
  const chip = (label, ok, tip) => chips.append(h('span', { class: `chipk ${ok === true ? 'ok' : (ok === false ? 'bad' : '')}`, 'data-tip': tip }, `${label}${ok === true ? ' ✓' : (ok === false ? ' ✕' : '')}`));
  if (r) {
    chip('agree', r.e4 ? r.e4.passed : null, r.e4 ? (r.physics === 'CO-T' ? `${r.e4.tip}. Here: energy ${sci(acc.energy_rel)}, outlet ${sci(acc.outlet_bulk_err_rel)} of the range.` : `${r.e4.tip} Here: energy ${sci(acc.energy_rel)} against ${r.e4.energy_bar}; max ${sci(acc.max_err)} against ${sci(r.e4.max_bar)}.`) : 'E4\'s bar applies to a chart-operator window against the full domain, steady');
    const bal = r.balance ? Object.values(r.balance).filter((x) => x !== null) : [];
    chip('balance', bal.length ? bal.every((x) => x < 1e-8) : null,
      `Heat balance to round-off (1e-8) in every window: ${Object.entries(r.balance || {}).map(([k, v]) => `${k} ${sci(v)}`).join(', ')}. ` +
      'The full domain conserves heat to round-off; a learned expert\'s field conserves it only to its own error.');
    chip('charts', r.certificate.charts[0] === r.certificate.charts[1], `${r.certificate.charts[0]} of ${r.certificate.charts[1]} charts certified (Proposition 3)`);
    if (r.physics !== 'CO-T') {
      chip('CX', null, 'CX, coarse exactness: measured by the registered evaluation, not in this run');
      chip('DUAL', null, 'DUAL, the 2-D dual bracket: held by the registered evaluation\'s records, not computed in this run');
    }
  }
  side.append(h('section', { class: 'subpanel' }, h('div', { class: 'label' }, 'Checks', h('b', { class: 'mono', style: { marginLeft: 'auto' } }, r ? `${chips.querySelectorAll('.ok').length} / ${chips.querySelectorAll('.ok, .bad').length}` : '')), chips));
  side.append(sec(h('span', {}, 'Certificate ', h('abbr', { class: 'q', 'data-tip': r?.certificate?.note || 'G0: linear, symmetric, energy-based' }, r?.certificate?.class || 'G0')),
    row('dual bound', '–', 'A guaranteed upper bound on B\'s energy error from the complementary problem: not computed in this run'),
    row('charts', r ? `${r.certificate.charts[0]} / ${r.certificate.charts[1]}` : '–'),
    row('min eigenvalue', sci(r?.certificate?.min_eig), r?.physics === 'CO-T' ? 'Of the symmetric part of the assembled interface matrix, over its norm: positive, so the coupled system is positive real' : 'Of the assembled interface matrix: positive, as Theorem 1 says it stays')));
  return side;
}

let root = null;

export function renderInto(el) {
  root = el;
  renderAll();
}

async function showStep(k) {
  const cr = store.corun;
  if (!cr) return;
  if (cr.frame && cr.frame_index === k) { frame = cr.frame; renderAll(); return; }
  const r = await send('corun.frame', { k });
  if (r.ok) { frame = r.frame; state.k = k; renderAll(); }
}

function renderAll() {
  if (!root) return;
  clear(root);
  const cr = store.corun;
  const run = cr && ['starting', 'running'].includes(cr.status);
  if (cr?.frame && (state.k === null || state.k === cr.frame_index || run)) { frame = cr.frame; state.k = cr.frame_index; }
  const top = h('div', { class: 'cotop' });
  const mode = h('div', { class: 'seg', role: 'group', 'aria-label': 'Mode' });
  for (const [m, label, tip] of [['single', 'Single', 'One window, one solver, against the full domain'], ['compare', 'Compare', 'Two windows side by side, each with its own solver']]) {
    const b = h('button', { 'aria-pressed': String(state.mode === m), 'data-tip': tip, disabled: run || null }, label);
    b.addEventListener('click', () => { state.mode = m; renderAll(); });
    mode.append(b);
  }
  const tabs = h('div', { class: 'seg', role: 'tablist', 'aria-label': 'View' });
  for (const [v, label, tip] of [['field', 'Field', 'The fields, on one colour scale when asked'], ['difference', 'Difference', state.mode === 'compare' ? 'B minus A, on one colour scale' : 'This window minus the full domain'],
    ['piece', 'Per piece', 'The energy error, one value per piece'], ['seams', 'Seams', 'The jump in T across the seams'], ['coarse', 'Coarse basis', 'The two-level expert\'s coarse functions, one at a time: not built yet']]) {
    const b = h('button', { role: 'tab', 'aria-selected': String(state.view === v), 'aria-pressed': String(state.view === v), 'data-tip': tip, disabled: v === 'coarse' || null }, label);
    b.addEventListener('click', () => { state.view = v; renderAll(); });
    tabs.append(b);
  }
  const go = h('button', { class: `btn ${run ? 'danger' : 'primary'}`, 'data-tip': run ? 'Stop after the step in progress' : 'Run the windows in turns on this machine' }, run ? 'Stop' : 'Run');
  go.addEventListener('click', async () => {
    if (run) { await send('corun.stop'); return; }
    frame = null; state.k = null;
    const r = await send('corun.start', state.mode === 'compare' ? { mode: 'compare', a: { ...state.A }, b: { ...state.B } } : { mode: 'single', a: { ...state.A } });
    if (!r.ok && r.error) toast('error', r.error);
  });
  top.append(mode, h('span', { class: 'spacer' }), tabs, h('span', { class: 'spacer' }), go);
  const wins = h('div', { class: 'cowins' }, panel('A', run));
  if (state.mode === 'compare') wins.append(panel('B', run));
  const steps = cr?.steps || 0;
  const done = cr?.frames || 0;
  const slider = h('input', { type: 'range', min: '0', max: String(Math.max(0, done - 1)), value: String(state.k ?? Math.max(0, done - 1)), 'aria-label': 'Step, both windows', disabled: done < 2 || null, style: { flex: 1 } });
  slider.addEventListener('change', () => showStep(Number(slider.value)));
  const same = h('input', { type: 'checkbox', checked: state.same || null, 'aria-label': 'Same colour scale' });
  same.addEventListener('change', () => { state.same = same.checked; renderAll(); });
  const status = cr ? (cr.status === 'failed' ? h('span', { class: 'err-text', 'data-tip': cr.error }, 'failed') : h('span', { class: 'muted' }, cr.status)) : null;
  const foot = h('div', { class: 'cofoot' }, slider, h('span', { class: 'mono' }, `${(state.k ?? done - 1) + 1 || 0} / ${steps || 0}`),
    h('label', { class: 'row', style: { gap: '4px' } }, same, 'same scale'), status);
  const main = h('main', { class: 'panel', style: { flex: 1, minWidth: 0, gap: '8px' } }, top, wins, foot);
  if (cr?.status === 'failed') main.append(h('div', { class: 'note err' }, cr.error));
  if (!store.pieces?.summary) main.append(h('div', { class: 'note info' }, 'Cut the domain on Pieces first: a run on the chart operator runs on its pieces.'));
  root.append(h('div', { class: 'work' }, main, cards()));
}

export function refresh() { renderAll(); }

/** set the windows before the page shows Run (Records' Open) */
export function preset(p) {
  if (p.mode) state.mode = p.mode;
  if (p.A) state.A = { ...p.A };
  if (p.B) state.B = { ...p.B };
  frame = null; state.k = null;
  renderAll();
}
