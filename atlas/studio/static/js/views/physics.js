// Physics (plan §4): the equation list from the family catalog, the equation screen,
// its properties, the regime numbers and the guarantee class.  Every equation either
// runs here or says why not on hover; every regime number names its definition.  The
// data is the server's (atlas/studio/registry_eq.py): this page only lays it out.

import { store, subscribe, send, edit, family } from '../store.js';
import { h, clear, fmt } from '../ui.js';

let root = null;
let query = '';

function numberField(value, onCommit, attrs = {}) {
  const input = h('input', { class: 'field num', value: value ?? '', inputmode: 'decimal', style: { width: '86px' }, ...attrs });
  const commit = () => {
    const v = input.value.trim();
    if (v === String(value ?? '')) return;
    const x = Number(v);
    if (v === '' || !Number.isFinite(x)) { input.value = value ?? ''; return; }
    onCommit(x);
  };
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') input.blur(); if (e.key === 'Escape') { input.value = value ?? ''; input.blur(); } });
  input.addEventListener('change', commit);
  return input;
}

const sci = (x) => {
  if (x === null || x === undefined || !Number.isFinite(x)) return '–';
  const a = Math.abs(x);
  if (a !== 0 && (a >= 1e4 || a < 1e-2)) {
    const e = Math.floor(Math.log10(a));
    const m = x / 10 ** e;
    return `${Math.abs(m - 1) < 5e-3 ? '' : `${m.toFixed(1)} × `}10${String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d])}`;
  }
  return fmt(x, 3);
};

function equationList(st) {
  const eqs = store.catalog.equations || [];
  const mine = st.regime?.equation;
  const box = h('aside', { class: 'panel scroll', style: { width: '232px', flex: 'none', gap: '4px' } });
  const search = h('input', { class: 'field', placeholder: 'Equation', 'aria-label': 'Search equations', value: query, style: { width: '100%' } });
  search.addEventListener('input', () => { query = search.value; renderList(list); });
  const list = h('div', { class: 'items', style: { gap: '1px' } });
  box.append(search, list);
  renderList(list);
  return box;

  function renderList(el) {
    clear(el);
    const q = query.trim().toLowerCase();
    const groups = store.catalog.groups || {};
    for (const [g, [label, fam, cls]] of Object.entries(groups)) {
      const rows = eqs.filter((e) => e.group === g && (!q || e.name.toLowerCase().includes(q) || e.kind.includes(q) || label.toLowerCase().includes(q)));
      if (!rows.length) continue;
      const clsKey = cls.split(' ')[0];
      el.append(h('div', { class: 'eqgroup' }, h('span', {}, label.toUpperCase()),
        h('abbr', { 'data-tip': `${fam}: ${store.catalog.classes?.[clsKey] || cls}` }, cls)));
      for (const e of rows) {
        const on = e.id === mine;
        const tip = e.runs ? `${e.name}: ${e.kind}${e.steady_kind ? ` (${e.steady_kind} if steady)` : ''}; runs as ${(family(e.runs_with) || {}).label || e.runs_with}` : e.why_not;
        const b = h('button', { class: `item eqrow${e.runs ? '' : ' off'}`, 'aria-selected': String(on), 'aria-disabled': String(!e.runs), 'data-tip': tip },
          h('span', { class: 'name' }, e.name), h('span', { class: 'meta' }, e.kind),
          e.co ? h('abbr', { class: 'cotag', 'data-tip': 'Chart operator available: an expert per four-sided piece (CO-D, the Stage 0 expert)' }, 'CO') : null);
        b.addEventListener('click', () => {
          if (!e.runs) return;
          if (e.runs_with !== st.meta.family) send('case.new', { family: e.runs_with });
        });
        el.append(b);
      }
    }
    if (!el.children.length) el.append(h('div', { class: 'empty' }, 'No equation matches.'));
  }
}

function boundaryTip(fam, b) {
  const k = (fam?.boundary_kinds || []).find((x) => x.id === b.kind);
  const val = b.value !== null && b.value !== undefined ? ` = ${fmt(b.value, 4)}${k?.unit ? ` ${k.unit}` : ''}` : '';
  return { text: `${b.kind}${val}`, tip: k ? `${k.meaning}${val ? `; here${val}` : ''}` : b.kind };
}

function equationScreen(st, eq) {
  const c = st.case;
  const fam = family();
  const main = h('section', { class: 'panel scroll', style: { flex: 1, minWidth: 0, gap: '10px' } });
  const head = h('div', { style: { display: 'flex', alignItems: 'center', gap: '10px' } },
    h('span', { style: { fontSize: '20px', fontWeight: 700 } }, eq ? eq.name : fam?.label || ''), h('span', { style: { flex: 1 } }));
  if ((fam?.modes || []).length > 1) {
    const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Steady or transient' });
    for (const [m, label, tip] of [['steady', 'Steady', 'Time-independent: one solve'], ['transient', 'Transient', 'Marches in time with steps of Δt']]) {
      if (!fam.modes.includes(m)) continue;
      const b = h('button', { 'aria-pressed': String(c.run.mode === m), 'data-tip': tip }, label);
      b.addEventListener('click', () => { if (c.run.mode !== m) edit([{ op: 'set', path: ['run', 'mode'], value: m }], `mode ${m}`); });
      seg.append(b);
    }
    head.append(seg);
  } else {
    head.append(h('span', { class: 'modechip', 'data-tip': 'This kind runs only this way' }, c.run.mode));
  }
  main.append(head);
  if (eq) {
    const f = h('div', { class: 'equation' });
    f.innerHTML = eq.html;                          // the server's static data, symbols explained on hover
    main.append(f);
  }
  // parameters per material in use
  const props = fam?.material_props || [];
  if (props.length) {
    main.append(h('div', { class: 'label' }, 'Parameters', h('span', { class: 'q', 'data-tip': 'Each material\'s properties; a region takes its material\'s values' }, '?')));
    const used = [...new Set((c.regions || []).map((r) => r.material))].filter((m) => c.materials[m]);
    const mats = used.length ? used : Object.keys(c.materials || {});
    const grid = h('div', { class: 'ptable', style: { gridTemplateColumns: `minmax(0, 1.2fr) repeat(${props.length}, minmax(0, 1fr))` } },
      h('span', { class: 'ph' }, 'material'), ...props.map((p) => h('span', { class: 'ph', 'data-tip': `${p.label}${p.unit ? `, ${p.unit}` : ''}${p.help ? `: ${p.help}` : ''}` }, p.name)));
    const shown = mats.filter((m) => !/^f\d+-\d+$/.test(m)).slice(0, 12);
    for (const m of shown) {
      grid.append(h('span', { class: 'pc', style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', padding: '6px' } }, m));
      for (const p of props) {
        const v = c.materials[m][p.name];
        grid.append(h('span', { class: 'pc' }, numberField(v, (x) => edit([{ op: 'set', path: ['materials', m, p.name], value: x }], `${m}: ${p.name} = ${x}`),
          { class: 'field num pin', 'aria-label': `${m} ${p.name}`, style: { width: '100%' } })));
      }
    }
    main.append(grid);
    const fieldMats = mats.filter((m) => /^f\d+-\d+$/.test(m));
    if (fieldMats.length) main.append(h('div', { class: 'muted', style: { fontSize: '12px' } }, `and ${fieldMats.length} levels of coefficient fields, set in Geometry`));
  }
  // how it runs
  const run = h('div', { style: { display: 'flex', gap: '14px', flexWrap: 'wrap', alignItems: 'center' } });
  if (c.run.mode !== 'steady') {
    run.append(h('label', { class: 'row', style: { gap: '6px' } }, h('abbr', { 'data-tip': 'Time step: the macro-step, in seconds (for the wind farm, in D/U)' }, 'Δt'),
      numberField(c.run.macro_dt, (x) => edit([{ op: 'set', path: ['run', 'macro_dt'], value: x }], `Δt = ${x}`), { 'aria-label': 'Δt' })));
  }
  run.append(h('label', { class: 'row', style: { gap: '6px' } }, h('span', { 'data-tip': c.run.mode === 'steady' ? 'Timed repeats of the steady solve' : 'Macro-steps marched' }, 'steps'),
    numberField(c.run.steps, (x) => edit([{ op: 'set', path: ['run', 'steps'], value: Math.round(x) }], `steps = ${x}`), { 'aria-label': 'steps', style: { width: '60px' } })));
  for (const p of fam?.params || []) {
    const v = c.physics.params[p.name] ?? p.default;
    run.append(h('label', { class: 'row', style: { gap: '6px' } }, h('abbr', { 'data-tip': `${p.label}${p.unit ? `, ${p.unit}` : ''}${p.help ? `: ${p.help}` : ''}` }, p.name),
      numberField(v, (x) => edit([{ op: 'set', path: ['physics', 'params', p.name], value: x }], `${p.label} = ${x}`), { 'aria-label': p.label })));
  }
  main.append(run);
  const more = h('div', { style: { display: 'flex', gap: '14px', flexWrap: 'wrap', alignItems: 'center' } });
  more.append(h('label', { class: 'row', style: { gap: '6px' } }, h('span', { 'data-tip': 'Threads for the parallel arm' }, 'threads'),
    numberField(c.run.threads, (x) => edit([{ op: 'set', path: ['run', 'threads'], value: Math.round(x) }], `threads = ${x}`), { 'aria-label': 'threads', style: { width: '50px' } })));
  if ((fam?.styles || []).length > 1) {
    const sel = h('select', { class: 'field', 'aria-label': 'Coupling', style: { width: '100%', minWidth: 0 },
      'data-tip': store.catalog.styles[c.coupling.style] || '' }, ...fam.styles.map((s) => h('option', { value: s, selected: s === c.coupling.style || null }, `${s}: ${store.catalog.styles[s]}`)));
    sel.addEventListener('change', () => edit([{ op: 'set', path: ['coupling', 'style'], value: sel.value }], `style ${sel.value}`));
    more.append(h('label', { class: 'row', style: { gap: '6px', flex: '1 1 260px', minWidth: 0 } }, h('span', { 'data-tip': 'How the pieces are coupled' }, 'coupling'), sel));
  }
  main.append(more);
  // boundaries
  main.append(h('div', { class: 'label' }, 'Boundaries'));
  const bl = h('div', { class: 'bgrid' });
  const groups = {};
  for (const b of c.boundaries) {
    const { text, tip } = boundaryTip(fam, b);
    (groups[text] = groups[text] || { tip, edges: [] }).edges.push(b.edge);
  }
  for (const [text, g] of Object.entries(groups)) {
    bl.append(h('div', { class: 'bcell' }, h('div', { class: 'muted', style: { fontSize: '11.5px' } }, g.edges.slice(0, 4).join(', ') + (g.edges.length > 4 ? ` +${g.edges.length - 4}` : '')),
      h('b', {}, h('abbr', { 'data-tip': g.tip }, text))));
  }
  if (!c.boundaries.length) bl.append(h('div', { class: 'empty' }, 'None.'));
  main.append(bl);
  return main;
}

function yes(v) { return v === true ? 'yes' : (v === false ? 'no' : '–'); }

function sidePanel(st, eq) {
  const r = st.regime || {};
  const side = h('aside', { class: 'panel scroll', style: { width: '262px', flex: 'none', gap: '10px' } });
  const sec = (title, ...kids) => h('section', { class: 'subpanel' }, h('div', { class: 'label' }, title), ...kids);
  const row = (k, v, tip = null, strong = false) => h('div', { class: 'kvrow' }, tip ? h('abbr', { 'data-tip': tip }, k) : h('span', {}, k), strong ? h('b', { class: 'mono' }, v) : h('span', {}, v));
  if (eq) {
    const name = (id) => (store.catalog.equations.find((e) => e.id === id) || {}).name || id;
    side.append(sec('Properties',
      row('type', eq.kind + (eq.steady_kind ? ` · ${eq.steady_kind} if steady` : ''), { elliptic: 'Elliptic: no time; every point affects every other', parabolic: 'Parabolic: diffusion in time; steady problems are elliptic', hyperbolic: 'Hyperbolic: waves travel at finite speed', mixed: 'Mixed: parts of more than one type' }[eq.kind]),
      eq.order ? row('order', eq.order) : null,
      row('linear', yes(eq.linear), 'Doubling the source doubles the answer; solutions add'),
      row('self-adjoint', yes(eq.self_adjoint), 'The operator equals its adjoint: symmetric matrices and an energy to minimise'),
      eq.energy ? row('energy', eq.energy, 'The quantity the solution minimises (steady case)') : null,
      row('max principle', yes(eq.max_principle), 'Without sources, the extremes are on the boundary or at the start: no overshoot inside'),
      eq.conserves ? row('conserves', eq.conserves) : null,
      eq.signal_speed ? row('signal speed', eq.signal_speed, 'How fast a change is felt elsewhere') : null,
      eq.same_operator.length ? row('same operator', eq.same_operator.map(name).join(', ')) : null,
      eq.uses ? row('uses', eq.uses) : null));
  }
  const reg = [];
  if (r.contrast) reg.push(row('contrast', sci(r.contrast.value), `${r.contrast.definition}: ${fmt(r.contrast.max, 4)} / ${fmt(r.contrast.min, 4)}`, true));
  if (r.fo_cell) reg.push(row('Fo, cell', sci(r.fo_cell.value), `${r.fo_cell.definition}; α = ${fmt(r.alpha.value, 4)} m²/s (${r.alpha.material}), Δt = ${fmt(r.dt, 4)} s, h = ${fmt(r.h, 4)} m`, true));
  if (r.fo_domain) {
    const d = r.fo_domain;
    const v = d.value === null ? 'steady' : sci(d.value);
    const tip = d.value === null ? `${d.definition}; the CO-D card admits steady pieces` : `${d.definition}; the CO-D card's range is ${fmt(d.range?.[0])} to ${fmt(d.range?.[1])}${d.within === false ? ': outside it, a learned expert would be refused here' : ''}`;
    reg.push(h('div', { class: 'kvrow' }, h('abbr', { 'data-tip': tip }, 'Fo, piece'),
      h('b', { class: `mono ${d.within === false ? 'warn-text' : ''}` }, v + (d.within === false ? ' ✗' : (d.within ? ' ✓' : '')))));
  }
  reg.push(r.pe ? row('Pe', sci(r.pe.value), `${r.pe.definition}; cell Pe = ${fmt(r.pe.cell, 3)}`, true) : row('Pe', '—', 'Péclet number: advection over diffusion (none here)'));
  if (r.re) reg.push(row('Re', sci(r.re.value), `${r.re.definition}; cell Re = ${fmt(r.re.cell, 3)}`, true));
  reg.push(r.k ? row('k h', fmt(r.k.kh, 3), `${r.k.definition}; k = ${fmt(r.k.value, 3)} 1/m`, true) : row('k', '—', 'Wavenumber, for wave problems (none here)'));
  side.append(sec('Regime', ...reg));
  if (eq) {
    const cls = eq.guarantee.split(' ')[0];
    const g = h('section', { class: 'subpanel' }, h('div', { class: 'label' }, 'Guarantee', h('abbr', { class: 'q', 'data-tip': eq.class_tip || '' }, eq.guarantee)));
    if (cls === 'G0') {
      g.append(h('div', { style: { fontSize: '12.5px' } }, h('abbr', { 'data-tip': 'The assembled interface matrix stays positive semidefinite whatever the network outputs' }, 'Λ̃ ⪰ 0'), ' · ',
        h('abbr', { 'data-tip': 'Céa constant 1: the coupled answer is the best one the experts\' modes allow' }, 'Céa 1'), ' · ',
        h('abbr', { 'data-tip': 'Proved in the Lean 4 proof assistant (lean/ in this repository)' }, 'Lean-checked')));
    } else {
      g.append(h('div', { style: { fontSize: '12.5px' } }, eq.class_tip || eq.guarantee));
    }
    g.append(h('div', { class: 'muted', style: { fontSize: '12px' } }, eq.co
      ? 'A learned expert is admitted only inside its card: the regime numbers above against its envelope.'
      : 'No chart-operator expert for this family yet: its pieces run classically.'));
    side.append(g);
  }
  return side;
}

function render() {
  if (!root || !store.state || !store.catalog) return;
  const st = store.state;
  const eq = (store.catalog.equations || []).find((e) => e.id === st.regime?.equation) || null;
  clear(root);
  root.append(h('div', { class: 'work' }, equationList(st), equationScreen(st, eq), sidePanel(st, eq)));
}

export function mount(el) {
  root = el;
  subscribe((what) => { if (what === 'state' || what === 'catalog') render(); });
}
