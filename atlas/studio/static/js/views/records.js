// Records (plan §7): the chart operator's registered evaluations, read only.  Nothing
// here re-runs anything; the data is the server's reading of out/chartop
// (atlas/studio/records.py), and a kind registered later appears without a change here.

import { store, subscribe, send } from '../store.js';
import { h, clear, fmt } from '../ui.js';
import { showTab } from '../header.js';
import { preset } from './corun.js';
import { usePieces } from './run.js';

let root = null;
let list = null;
let chosen = null;
let asked = false;

const BENCH = ['bend-3', 's-channel', 'insert-round'];

async function load() {
  const r = await send('records.list');
  if (!r.ok) return;
  list = r.records;
  if (!chosen || !list.find((e) => e.file === chosen)) chosen = (list.find((e) => e.status === 'evaluated') || list[0])?.file || null;
  render();
}

function svg(tag, attrs, ...kids) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  for (const c of kids) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

function priceChart(prices) {
  // grouped by problem set: numbers over different sets are never side by side as comparable
  const sets = [...new Set(prices.map((p) => p.label))];
  const box = h('div', { class: 'pricegroups' });
  const top = Math.max(1.1, ...prices.map((p) => p.value));
  for (const set of sets) {
    const rows = prices.filter((p) => p.label === set);
    const g = h('div', { class: 'pgroup' }, h('div', { class: 'label' }, set));
    for (const p of rows) {
      const w = Math.max(2, 100 * p.value / top);
      g.append(h('div', { class: 'prow', 'data-tip': `${p.arm} over the classical solve, the geometric mean over ${set}${p.bar ? `; the bar is ${p.bar}` : ''}. Below 1 is cheaper.` },
        h('span', { class: 'mono' }, p.arm), h('span', { class: 'pbar' }, h('i', { style: { width: `${w}%`, background: p.value < 1 ? 'var(--ready)' : 'var(--accent)' } }),
          h('b', { class: 'one', style: { left: `${100 / top}%` } })), h('span', { class: 'mono' }, fmt(p.value, 3))));
    }
    box.append(g);
  }
  return box;
}

function rhoChart(r) {
  const W = 300, H = 120, pad = 18;
  const bins = r.bins;
  const top = Math.max(...bins, 1);
  const n = bins.length;
  const bw = (W - 2 * pad) / n;
  const s = svg('svg', { viewBox: `0 0 ${W} ${H + 26}`, width: '100%', role: 'img', 'aria-label': `rho over ${r.pieces} test pieces: median ${fmt(r.median, 3)}, 10th percentile ${fmt(r.p10, 3)}` });
  bins.forEach((c, i) => {
    const hgt = (H - pad) * c / top;
    s.append(svg('rect', { x: pad + i * bw + 0.5, y: H - hgt, width: Math.max(1, bw - 1), height: hgt, fill: i === 0 ? '#E5484D' : '#8E72EE', rx: 1 }));
  });
  const xOf = (v) => pad + bw + (W - 2 * pad - bw) * v;
  for (const [v, label, colour] of [[r.median, `median ${fmt(r.median, 3)}`, '#00A9A5'], [r.p10, `p10 ${fmt(r.p10, 3)}`, '#F5A524']]) {
    s.append(svg('line', { x1: xOf(v), x2: xOf(v), y1: 8, y2: H, stroke: colour, 'stroke-width': 2, 'stroke-dasharray': '4 3' }));
    s.append(svg('text', { x: xOf(v) + 3, y: label.startsWith('median') ? 16 : 30, 'font-size': 10, fill: 'currentColor', 'font-family': 'JetBrains Mono, monospace' }, label));
  }
  for (const [v, t] of [[0, '0'], [1, '1']]) s.append(svg('text', { x: xOf(v), y: H + 14, 'font-size': 10, fill: 'currentColor', 'text-anchor': 'middle' }, t));
  s.append(svg('text', { x: pad + bw / 2, y: H + 14, 'font-size': 10, fill: '#E5484D', 'text-anchor': 'middle' }, '< 0'));
  return s;
}

async function openInRun(e, key) {
  // the problem, cut automatically, in Run's compare mode: A the full domain, B the record's main network
  const r = await send('case.example', { key });
  if (!r.ok) return;
  await send('pieces.cut', { n: 64 });
  // the experts' list comes with the cut's first frame
  for (let i = 0; i < 100 && !(store.pieces?.experts || []).length; i++) await new Promise((ok) => setTimeout(ok, 100));
  const offered = (store.pieces?.experts || []).filter((x) => x.offered).map((x) => x.key);
  const main = e.main && offered.includes(e.main) ? e.main : 'classical';
  preset({ mode: 'compare', A: { method: 'full', solver: 'classical' }, B: { method: 'chart', solver: main } });
  usePieces();
  showTab('run');
}

function render() {
  if (!root) return;
  clear(root);
  const side = h('aside', { class: 'panel scroll', style: { width: '220px', flex: 'none', gap: '4px' } }, h('div', { class: 'label' }, 'Evaluations'));
  for (const e of list || []) {
    const b = h('button', { class: 'item recrow', 'aria-selected': String(e.file === chosen), disabled: !e.file || null,
      'data-tip': e.note || (e.status === 'evaluated' ? `${e.file}` : e.status) },
      h('span', { class: 'name' }, h('b', {}, e.name), e.stamp ? ` · ${e.stamp.slice(4, 6)}-${e.stamp.slice(6, 8)}` : ''),
      h('span', { class: 'meta' }, e.status === 'evaluated' ? (e.arms || []).slice(0, 3).join(', ') : e.status));
    b.addEventListener('click', () => { chosen = e.file; render(); });
    side.append(b);
  }
  if (!list) side.append(h('div', { class: 'empty' }, 'Reading out/chartop…'));
  const e = (list || []).find((x) => x.file === chosen);
  const main = h('main', { class: 'panel scroll', style: { flex: 1, minWidth: 0, gap: '10px' } });
  if (!e) {
    main.append(h('div', { class: 'placeholder' }, h('h2', {}, 'Chart-operator records'), h('div', {}, 'No registered evaluation under out/chartop.')));
  } else {
    main.append(h('div', { style: { display: 'flex', alignItems: 'baseline', gap: '10px', flexWrap: 'wrap' } },
      h('span', { style: { fontSize: '19px', fontWeight: 700 } }, e.name, ' ', h('abbr', { 'data-tip': 'CO-D: the chart-operator family for linear diffusion', class: 'muted', style: { fontSize: '13px' } }, 'CO-D')),
      h('span', { class: 'muted mono', style: { fontSize: '12px' } }, `${e.file}${e.code ? ` · ${String(e.code).slice(0, 7)}` : ''}`),
      e.registered_run ? h('abbr', { class: 'cotag', 'data-tip': `The gates and code were fixed by hash before anything was measured. ${e.registration}` }, 'hash-pinned') :
        h('span', { class: 'muted', style: { fontSize: '12px' }, 'data-tip': e.registration || '' }, e.status)));
    const chips = h('div', { class: 'chips' });
    for (const g of e.gates) {
      const state = g.met === true ? 'ok' : (g.met === false ? 'bad' : '');
      chips.append(h('span', { class: `chipk ${state}`, 'data-gate': g.key, 'data-tip': `${g.key}: ${g.title}. ${g.bar}${g.met === null ? ' (reported, not judged here)' : (g.met ? '. Met.' : '. Missed.')}` },
        `${g.key}${g.met === true ? ' ✓' : (g.met === false ? ' ✕' : '')}`));
    }
    if (!e.gates.length) chips.append(h('span', { class: 'muted' }, 'No gates are read for this kind.'));
    main.append(h('div', { class: 'label' }, 'Gates'), chips);
    const row = h('div', { class: 'recrow2' });
    const pr = h('section', { class: 'subpanel', style: { flex: 1, minWidth: 0 } },
      h('div', { class: 'label' }, h('abbr', { 'data-tip': 'Each arm\'s time over the classical solve, as the record\'s registered price protocol measured it. Below 1 is cheaper. Grouped by problem set: a mean over every priced problem is not comparable with one over the held-out problems.' }, 'Price / classical')));
    pr.append(e.prices.length ? priceChart(e.prices) : h('div', { class: 'muted' }, 'This record holds no price.'));
    const rh = h('section', { class: 'subpanel', style: { flex: 1, minWidth: 0 } },
      h('div', { class: 'label' }, h('abbr', { 'data-tip': 'ρ = 1 − ε_TL / ε_TL0: the share of TL0\'s error the network removes, per test piece. 1 is perfect; 0 no help; below 0 worse than TL0.' }, e.rho ? `ρ · ${e.rho.pieces.toLocaleString('en')} test pieces · ${e.rho.arm}` : 'ρ')));
    rh.append(e.rho ? rhoChart(e.rho) : h('div', { class: 'muted' }, 'This record holds no per-piece ρ.'));
    row.append(pr, rh);
    main.append(row);
    if (e.status === 'evaluated') {
      const open = h('div', { style: { display: 'flex', gap: '6px', alignItems: 'center', flexWrap: 'wrap' } }, h('span', { class: 'muted' }, 'Open'));
      for (const k of BENCH) {
        const b = h('button', { class: 'btn small', 'data-tip': `Open ${k} in Run, compare: ${e.main || 'the classical expert'} on every piece against the full-domain solve` }, k);
        b.addEventListener('click', () => openInRun(e, k));
        open.append(b);
      }
      const m = e.machine;
      open.append(h('span', { class: 'spacer', style: { flex: 1 } }),
        h('span', { class: 'muted', style: { fontSize: '12px' } }, m && typeof m === 'object' ? `${m.available_cpus || m.cpu_count || ''} cores · ${m.cpu || m.system || ''}` : (typeof m === 'string' ? m : '')));
      main.append(open);
    } else if (e.note) {
      main.append(h('div', { class: 'note info' }, e.note));
    }
  }
  root.append(h('div', { class: 'work' }, side, main));
}

export function mount(el) {
  root = el;
  subscribe((what) => {
    if (what === 'tab' && store.tab === 'records' && !asked) { asked = true; load(); }
    if (what === 'tab' && store.tab === 'records') render();
  });
  window.atlas = Object.assign(window.atlas || {}, { records: { reload: load, data: () => list } });
}
