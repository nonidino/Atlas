// Run, phase P1's minimal run: the existing runner, shown as cards and fields.
// P5 adds Single and Compare, the chart-operator path and the views of plan
// section 6.  The price is a ratio only; no raw seconds are ever drawn.

import { store, subscribe, send } from '../store.js';
import { h, clear, fmt, cssVar } from '../ui.js';
import { GLOSSARY } from '../glossary.js';
import { renderInto as corunInto } from './corun.js';

let root = null;
let fieldsBox = null;
const ARMS = ['serial', 'parallel', 'full'];
let chosen = null;
let runKind = null;          // 'pieces' (the chart operator, P5) or 'windows' (the classical arms)
let runKindChosen = false;   // until the person picks, it follows whether pieces exist

// a perceptual ramp from the palette's ink to cyan and violet (sequential), and a
// diverging one for differences
const SEQ = [[25, 25, 39], [59, 38, 144], [142, 114, 238], [0, 169, 165], [0, 215, 210], [228, 247, 245]];
const DIV = [[59, 38, 144], [142, 114, 238], [242, 241, 247], [0, 215, 210], [0, 80, 78]];

function ramp(stops, t) {
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function decode(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Uint16Array(buf);
}

export function fieldCanvas(f, diverging = false, lo = null, hi = null) {
  const q = decode(f.data);
  const c = h('canvas', { width: f.nx, height: f.ny });
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(f.nx, f.ny);
  const span = f.max - f.min;
  const a = lo ?? f.min, b = hi ?? f.max;
  const m = diverging ? Math.max(Math.abs(a), Math.abs(b)) || 1 : 1;
  for (let j = 0; j < f.ny; j++) {
    for (let i = 0; i < f.nx; i++) {
      const v = q[j * f.nx + i];
      const o = ((f.ny - 1 - j) * f.nx + i) * 4;
      if (v === 65535) { img.data[o + 3] = 0; continue; }
      const val = f.min + (v / 65534) * span;
      const t = diverging ? 0.5 + val / (2 * m) : (b > a ? (val - a) / (b - a) : 0.5);
      const [r, g, bl] = ramp(diverging ? DIV : SEQ, t);
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = bl; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  c.style.imageRendering = 'pixelated';
  return c;
}

function card(title, value, detail, verdict = null, good = null, tipText = null) {
  return h('div', { class: 'panel', style: { flex: 1, minWidth: '150px', gap: '2px' }, 'data-tip': tipText },
    h('div', { class: 'label' }, title, verdict ? h('span', { style: { marginLeft: 'auto', color: good ? 'var(--ready-ink)' : (good === false ? 'var(--err-ink)' : 'var(--muted)') } }, verdict) : null),
    h('div', { class: 'mono', style: { fontSize: '22px', fontWeight: 600 } }, value),
    h('div', { class: 'muted', style: { fontSize: '12px' } }, detail));
}

function cards(cd) {
  const row = h('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap' } });
  if (cd.speed) {
    const sp = cd.spread[cd.speed.arm];
    row.append(card('Price', `${fmt(cd.speed.ratio, 3)}×`,
      `full domain over ${cd.speed.label.toLowerCase()}; above 1 is faster` + (sp ? ` · A/A ${fmt(sp, 2)}` : ''),
      null, null, 'The whole domain\'s mean step time divided by this arm\'s, timed in turns on this machine. ' + GLOSSARY['A/A']));
  } else {
    row.append(card('Price', '–', 'needs a decomposed arm and the whole domain'));
  }
  for (const [k, title] of [['agreement', 'Agreement'], ['balance', 'Balance']]) {
    const c = cd[k];
    if (c) row.append(card(title, fmt(c.value, 3), `${c.title}; limit ${fmt(c.tolerance, 1)}`, c.verdict, c.passed,
      `${c.title}, against its registered tolerance`));
  }
  row.append(card('Checks', `${cd.passed} of ${cd.checks.length}`,
    cd.failed ? `${cd.failed} failed against the tolerance registered before any run` : 'every check that could be measured passed',
    cd.failed ? 'FAIL' : 'pass', !cd.failed, 'Each check was registered with its tolerance before any run'));
  const out = [row];
  if (cd.fast) {
    out.push(h('div', { class: 'panel' }, h('div', { class: 'label' }, `Fast example · ${cd.fast.mechanism}`),
      h('div', {}, cd.fast.line), h('div', { class: 'muted' }, cd.fast.verdict + ' Measured in this run, on this machine.'),
      h('div', { class: 'muted', style: { fontStyle: 'italic' } }, cd.fast.fixed)));
  }
  if (cd.learned) {
    const L = cd.learned;
    const lines = Object.values(L.speed_vs_learned).map((s) => `${s.label} took ${fmt(s.ratio, 3)}× the learned arm's step`);
    out.push(h('div', { class: 'panel' }, h('div', { class: 'label' }, 'Learned experts'),
      ...lines.map((l) => h('div', {}, l)),
      L.gate ? h('div', {}, `Registered evaluation (${L.gate_at}): ` + L.gate.map((g) => `${g.key} ${g.passed === true ? 'pass' : (g.passed === false ? 'FAIL' : '–')}`).join(' · ')) : null,
      h('div', { class: 'muted', style: { fontStyle: 'italic' } }, L.caveat)));
  }
  const det = h('details', {}, h('summary', { class: 'muted' }, 'Every check'));
  const tbl = h('div', { class: 'issues' });
  for (const c of cd.checks) {
    tbl.append(h('div', { class: `issue ${c.passed === false ? 'error' : (c.passed === null ? 'info' : 'ok')}` },
      h('span', { class: 'sev', style: { background: c.passed === false ? 'var(--err)' : (c.passed === null ? 'var(--dim)' : 'var(--ready)') } }),
      h('span', { class: 'msg' }, `${c.title}: ${fmt(c.value, 3)}` + (c.tolerance !== null ? ` (limit ${fmt(c.tolerance, 1)})` : ' (exact)') + (c.detail ? ` · ${c.detail}` : ''))));
  }
  for (const n of cd.notes) tbl.append(h('div', { class: 'note info' }, n));
  det.append(tbl);
  out.push(det);
  return out;
}

function renderFields() {
  if (!fieldsBox) return;
  clear(fieldsBox);
  const f = store.fields;
  if (!f) return;
  const labels = store.run?.labels || {};
  const arms = Object.keys(f.arms);
  const plain = arms.filter((a) => a !== 'difference');
  const lo = Math.min(...plain.map((a) => f.arms[a].min)), hi = Math.max(...plain.map((a) => f.arms[a].max));
  for (const a of arms) {
    const fa = f.arms[a];
    const cv = fieldCanvas(fa, a === 'difference', a === 'difference' ? null : lo, a === 'difference' ? null : hi);
    cv.style.width = '100%'; cv.style.height = 'auto'; cv.style.maxHeight = '170px'; cv.style.objectFit = 'contain';
    fieldsBox.append(h('figure', { style: { margin: 0, flex: '1 1 200px', minWidth: '160px' } }, cv,
      h('figcaption', { class: 'muted', style: { fontSize: '11.5px' } },
        a === 'difference' ? 'decomposed − full' : (labels[a] || a),
        h('span', { class: 'mono', style: { marginLeft: '6px' } }, `${fmt(fa.min)} … ${fmt(fa.max)}`))));
  }
}

function render() {
  if (!root || !store.state) return;
  clear(root);
  const heat = store.state.meta.family === 'conduction-2d';
  if (heat) {
    if (!runKindChosen) runKind = store.pieces?.summary ? 'pieces' : 'windows';
    const kind = h('div', { class: 'seg', role: 'group', 'aria-label': 'What runs', style: { margin: '8px 8px 0', alignSelf: 'flex-start' } });
    for (const [k, label, tip] of [['pieces', 'Pieces', 'The chart operator on the pieces cut on Pieces: Single or Compare'], ['windows', 'Windows', "The classical arms on the case's windows: serial, parallel and the full domain"]]) {
      const b = h('button', { 'aria-pressed': String(runKind === k), 'data-tip': tip }, label);
      b.addEventListener('click', () => { runKind = k; runKindChosen = true; render(); });
      kind.append(b);
    }
    const holder = h('div', { style: { display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, minWidth: 0 } }, kind);
    root.append(holder);
    if (runKind === 'pieces') {
      const box = h('div', { style: { display: 'flex', flex: 1, minHeight: 0 } });
      holder.append(box);
      corunInto(box);
      return;
    }
    const box = h('div', { style: { display: 'flex', flex: 1, minHeight: 0 } });
    holder.append(box);
    renderWindows(box);
    return;
  }
  renderWindows(root);
}

function renderWindows(into) {
  const r = store.run;
  const active = r && ['starting', 'running', 'stopping'].includes(r.status);
  const left = h('aside', { class: 'panel scroll', style: { width: '220px', flex: 'none' } },
    h('div', { class: 'label' }, 'Arms', h('span', { class: 'q', 'data-tip': 'The arms march in turns, so they share the machine and the moment.' }, '?')));
  chosen = chosen || [...ARMS];
  for (const a of ARMS) {
    const cb = h('input', { type: 'checkbox', checked: chosen.includes(a) || null });
    cb.addEventListener('change', () => { chosen = cb.checked ? [...new Set([...chosen, a])] : chosen.filter((x) => x !== a); });
    left.append(h('label', { class: 'row', style: { justifyContent: 'flex-start' } }, cb,
      { serial: 'Decomposed, serial', parallel: 'Decomposed, parallel', full: 'Full domain' }[a]));
  }
  const go = h('button', { class: `btn ${active ? 'danger' : 'primary'}` }, active ? 'Stop' : 'Run');
  go.addEventListener('click', async () => {
    if (active) await send('run.stop');
    else await send('run.start', { arms: chosen });
  });
  left.append(go);
  const comp = h('button', { class: 'btn', 'data-tip': 'The Atlas compiler\'s verdict on every seam' }, 'Compile');
  comp.addEventListener('click', () => send('compile.start'));
  left.append(comp);
  // a sweep: the case once per value of a parameter marked in Geometry
  const swept = (store.state.geometry?.parameters || []).filter((p) => p.sweep && p.sweep.length);
  const sw = store.sweep;
  const sweeping = sw && ['starting', 'running', 'stopping'].includes(sw.status);
  if (swept.length || sw) {
    left.append(h('div', { class: 'rule' }), h('div', { class: 'label' }, 'Sweep', h('span', { class: 'q', 'data-tip': 'The case runs once per value, each on its own copy, in turns; this case is not changed.' }, '?')));
    const pick = h('select', { class: 'field', 'aria-label': 'Parameter to sweep' }, ...swept.map((p) => h('option', { value: p.name }, `${p.name}: ${p.sweep.length} values`)));
    if (swept.length) left.append(pick);
    const sb = h('button', { class: `btn ${sweeping ? 'danger' : ''}`, disabled: (!swept.length && !sweeping) || active || null,
      'data-tip': sweeping ? 'Stop after the current run' : 'Run the case once per value of the parameter' }, sweeping ? 'Stop sweep' : 'Run sweep');
    sb.addEventListener('click', async () => {
      if (sweeping) await send('run.stop_sweep');
      else await send('run.sweep', { param: pick.value, arms: chosen });
    });
    left.append(sb);
    if (!swept.length) left.append(h('div', { class: 'muted', style: { fontSize: '12px' } }, 'Mark a parameter for a sweep in Geometry, Parameters.'));
  }
  const cj = store.compile;
  if (cj) {
    const s = cj.summary;
    left.append(h('div', { class: `note ${cj.status === 'failed' ? 'err' : (s && s.verdict === 'admit' ? 'ok' : 'warn')}` },
      cj.status === 'compiling' || cj.status === 'starting' ? 'Compiling…' :
        (cj.status === 'failed' ? `Compile failed: ${cj.error}` :
          (s ? `${s.verdict}${s.refused_before ? ' before the compiler' : ''} · ${s.seams.length} seams` : cj.status)),
      cj.this_case === false ? h('div', {}, 'This was another case.') : null));
  }
  const main = h('section', { class: 'panel scroll', style: { flex: 1, minWidth: 0 } });
  if (sw) main.append(sweepTable(sw));
  if (!r) {
    main.append(h('div', { class: 'placeholder' }, h('h2', {}, 'Nothing has run yet'),
      h('div', {}, 'Run marches every chosen arm in turns and compares them.')));
  } else {
    main.append(h('div', { style: { display: 'flex', alignItems: 'center', gap: '10px' } },
      h('b', {}, r.status === 'done' ? 'Finished' : (r.status === 'stopped' ? 'Stopped' : (r.status === 'failed' ? 'Failed' : 'Marching'))),
      h('span', { class: 'mono' }, `${r.step} / ${r.steps} ${r.step_label}s`),
      active ? h('span', { class: 'muted' }, r.arm_label) : null,
      r.this_case === false ? h('span', { class: 'tag cut' }, 'another case') : null,
      h('span', { class: 'muted', style: { marginLeft: 'auto', fontSize: '12px' } }, r.message)));
    const pct = r.steps ? Math.round(100 * r.step / r.steps) : 0;
    main.append(h('div', { style: { height: '6px', background: 'var(--line-soft)', borderRadius: '3px' } },
      h('div', { style: { width: `${pct}%`, height: '6px', background: 'var(--accent)', borderRadius: '3px' } })));
    if (r.error) main.append(h('div', { class: 'note err' }, r.error));
    if (r.cards) main.append(...cards(r.cards));
    fieldsBox = h('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap' } });
    main.append(h('div', { class: 'label' }, 'Fields'), fieldsBox);
    renderFields();
  }
  into.append(h('div', { class: 'work' }, left, main));
}

function sweepTable(sw) {
  const box = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '6px' } });
  box.append(h('div', { style: { display: 'flex', alignItems: 'center', gap: '10px' } },
    h('b', {}, `Sweep of ${sw.param}`),
    h('span', { class: 'mono' }, `${sw.results.filter((x) => x.status !== 'running').length} / ${sw.values.length}`),
    h('span', { class: 'muted', style: { marginLeft: 'auto', fontSize: '12px' } }, sw.status)));
  if (sw.error) box.append(h('div', { class: 'note err' }, sw.error));
  const t = h('div', { class: 'sweeptable' },
    h('span', { class: 'ph' }, sw.param), h('span', { class: 'ph' }, 'run'), h('span', { class: 'ph', 'data-tip': 'The registered agreement measure of the decomposed arm against the full domain' }, 'agreement'),
    h('span', { class: 'ph', 'data-tip': "The whole domain's step time over the decomposed arm's, on this machine; above 1 is faster" }, 'price'), h('span', { class: 'ph' }, 'checks'));
  for (const [k, v] of sw.values.entries()) {
    const res = sw.results[k];
    const cd = res && res.cards;
    t.append(h('span', { class: 'mono' }, fmt(v, 4)),
      h('span', {}, res ? (res.error ? h('span', { class: 'err-text', 'data-tip': res.error }, 'failed') : res.status) : (k === sw.index && sw.status === 'running' ? 'running' : '–')),
      h('span', { class: 'mono' }, cd && cd.agreement ? `${fmt(cd.agreement.value, 3)} ${cd.agreement.passed ? '✓' : '✗'}` : '–'),
      h('span', { class: 'mono' }, cd && cd.speed ? `${fmt(cd.speed.ratio, 3)}×` : '–'),
      h('span', { class: 'mono' }, cd ? `${cd.passed}/${cd.checks.length}` : '–'));
  }
  box.append(t, h('div', { class: 'rule' }));
  return box;
}

/** show the pieces run (Records' Open asks for it) */
export function usePieces() { runKind = 'pieces'; runKindChosen = true; render(); }

export function mount(el) {
  root = el;
  subscribe((what) => {
    if (['run', 'state', 'compile', 'tab', 'sweep', 'corun', 'pieces'].includes(what)) render();
    if (what === 'fields') renderFields();
  });
}
