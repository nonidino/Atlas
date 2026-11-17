// Physics, phase P0-P1: the kind of simulation, its parameters and how it runs.
// P3 replaces the kind list with the equation registry (types, guarantee classes,
// regime numbers, refusals).

import { store, subscribe, send, edit, family } from '../store.js';
import { h, clear, fmt } from '../ui.js';

let root = null;

function numberField(value, onCommit, attrs = {}) {
  const input = h('input', { class: 'field num', value: value ?? '', inputmode: 'decimal',
    style: { width: '96px' }, ...attrs });
  const commit = () => {
    const v = input.value.trim();
    if (v === String(value ?? '')) return;
    const x = Number(v);
    if (v === '' || !Number.isFinite(x)) { input.value = value ?? ''; return; }
    onCommit(x);
  };
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { input.blur(); } if (e.key === 'Escape') { input.value = value ?? ''; input.blur(); } });
  input.addEventListener('change', commit);
  return input;
}

function render() {
  if (!root || !store.state || !store.catalog) return;
  const st = store.state;
  const c = st.case;
  const fam = family();
  clear(root);
  const kinds = h('aside', { class: 'panel scroll', style: { width: '230px', flex: 'none' } },
    h('div', { class: 'label' }, 'Kind', h('span', { class: 'q', 'data-tip': 'Another kind starts a new case of it, ready to run. Undo brings this one back.' }, '?')));
  const list = h('div', { class: 'items' });
  for (const f of store.catalog.families) {
    const b = h('button', { class: 'item', 'aria-selected': String(f.id === st.meta.family), disabled: !f.runs || null,
      'data-tip': f.runs ? f.label : `${f.label}: no runner yet. ${f.note}` },
      h('span', { class: 'name' }, f.short[0].toUpperCase() + f.short.slice(1)),
      h('span', { class: 'meta' }, f.styles.join(' ')));
    b.addEventListener('click', () => { if (f.id !== st.meta.family) send('case.new', { family: f.id }); });
    list.append(b);
  }
  kinds.append(list);

  const main = h('section', { class: 'panel scroll', style: { flex: 1, minWidth: 0 } });
  main.append(h('div', { style: { fontSize: '17px', fontWeight: 700 } }, fam ? fam.label : ''));
  main.append(h('div', { class: 'label' }, 'Parameters'));
  const grid = h('div', { style: { display: 'grid', gridTemplateColumns: 'minmax(0,1fr) auto auto', gap: '6px 10px', alignItems: 'center' } });
  for (const p of fam?.params || []) {
    const v = c.physics.params[p.name] ?? p.default;
    grid.append(h('span', { 'data-tip': p.help || p.label }, p.label),
      numberField(v, (x) => edit([{ op: 'set', path: ['physics', 'params', p.name], value: x }], `${p.label} = ${x}`)),
      h('span', { class: 'muted mono', style: { fontSize: '11px' } }, p.unit));
  }
  main.append(grid);
  main.append(h('div', { class: 'rule' }));
  main.append(h('div', { class: 'label' }, 'Run'));
  const runGrid = h('div', { style: { display: 'grid', gridTemplateColumns: 'minmax(0,1fr) auto', gap: '6px 10px', alignItems: 'center' } });
  if ((fam?.modes || []).length > 1) {
    const sel = h('select', { class: 'field' }, ...fam.modes.map((m) => h('option', { value: m, selected: m === c.run.mode || null }, m)));
    sel.addEventListener('change', () => edit([{ op: 'set', path: ['run', 'mode'], value: sel.value }], `mode ${sel.value}`));
    runGrid.append(h('span', { 'data-tip': 'Steady: one solve, timed in repeats. Transient: marched in time.' }, 'mode'), sel);
  }
  runGrid.append(h('span', { 'data-tip': 'Δt: the macro-step, in seconds (for the wind farm, in D/U)' }, 'Δt'),
    numberField(c.run.macro_dt, (x) => edit([{ op: 'set', path: ['run', 'macro_dt'], value: x }], `Δt = ${x}`)));
  runGrid.append(h('span', { 'data-tip': c.run.mode === 'steady' ? 'Timed repeats of the steady solve' : 'Macro-steps marched' }, 'steps'),
    numberField(c.run.steps, (x) => edit([{ op: 'set', path: ['run', 'steps'], value: Math.round(x) }], `steps = ${x}`)));
  runGrid.append(h('span', { 'data-tip': 'Threads for the parallel arm' }, 'threads'),
    numberField(c.run.threads, (x) => edit([{ op: 'set', path: ['run', 'threads'], value: Math.round(x) }], `threads = ${x}`)));
  if ((fam?.styles || []).length > 1) {
    const sel = h('select', { class: 'field' }, ...fam.styles.map((s) => h('option', { value: s, selected: s === c.coupling.style || null }, `${s}: ${store.catalog.styles[s]}`)));
    sel.addEventListener('change', () => edit([{ op: 'set', path: ['coupling', 'style'], value: sel.value }], `style ${sel.value}`));
    runGrid.append(h('span', { 'data-tip': 'How the pieces are coupled' }, 'coupling'), sel);
  }
  main.append(runGrid);

  const right = h('aside', { class: 'panel scroll', style: { width: '260px', flex: 'none' } },
    h('div', { class: 'label' }, 'Boundaries'));
  const bl = h('div', { class: 'items' });
  for (const b of c.boundaries) {
    bl.append(h('div', { class: 'item', 'data-tip': `${b.kind} on ${b.edge}` },
      h('span', { class: 'name' }, b.edge), h('span', { class: 'meta' }, b.kind + (b.value !== null && b.value !== undefined ? ` ${fmt(b.value)}` : ''))));
  }
  if (!c.boundaries.length) bl.append(h('div', { class: 'empty' }, 'None.'));
  right.append(bl);
  root.append(h('div', { class: 'work' }, kinds, main, right));
}

export function mount(el) {
  root = el;
  subscribe((what) => { if (what === 'state' || what === 'catalog') render(); });
}
