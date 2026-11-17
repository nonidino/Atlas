// Parameters, formulas and sweeps (plan §3.1): the Parameters panel, a field bound to a
// formula in the inspector, and the degrees-of-freedom readout.  A parameter is a
// number or a formula of other parameters; a shape's field bound to a formula follows
// it; the server resolves both (atlas/studio/resolve.py) and this page previews them
// with the same grammar (formula.js).  A parameter marked for a sweep holds the values
// a sweep runs, one case per value, from Run.

import { h, fmt, dialog, toast } from '../ui.js';
import { resolveParameters, parse, names, value as fvalue, FormulaError } from './formula.js';

const NAME = /^[A-Za-z_][A-Za-z0-9_]{0,30}$/;
const RESERVED = new Set(['x', 'y', 'T', 'pi', 'e']);

/** every parameter's value; {} when they cannot be resolved (the server says why) */
export function values(geom) {
  try { return resolveParameters(geom.parameters || []); } catch (err) { return {}; }
}

export function freshName(geom, base) {
  const used = new Set((geom.parameters || []).map((p) => p.name));
  if (!used.has(base) && !RESERVED.has(base)) return base;
  let k = 2;
  while (used.has(`${base}${k}`)) k++;
  return `${base}${k}`;
}

/** a formula's trouble against these parameters, or null */
export function formulaTrouble(text, geom, allowXY = false) {
  try {
    const tree = parse(text);
    const known = new Set((geom.parameters || []).map((p) => p.name));
    for (const n of names(tree)) {
      if (known.has(n) || n === 'pi' || n === 'e' || (allowXY && (n === 'x' || n === 'y'))) continue;
      if (n === 'T') return 'T is not a variable of any kind that runs here: the properties are constant in time';
      return `no parameter called '${n}'`;
    }
    if (!allowXY) fvalue(text, values(geom));
    return null;
  } catch (err) {
    return err instanceof FormulaError ? err.message : String(err.message || err);
  }
}

/** a bare parameter name, if the formula is one */
export function bareName(expr) {
  const t = String(expr || '').trim();
  return NAME.test(t) ? t : null;
}

const refRe = (n) => new RegExp(`(^|[^A-Za-z0-9_])${n}(?![A-Za-z0-9_])`, 'g');

/** ops renaming a parameter, and every formula, binding and dimension that reads it */
function renameOps(geom, from, to) {
  const ops = [];
  const sub = (t) => String(t).replace(refRe(from), (m, a) => a + to);
  geom.parameters.forEach((p, i) => {
    if (p.name === from) ops.push({ op: 'set', path: ['geometry', 'parameters', i, 'name'], value: to });
    if (p.expr && refRe(from).test(p.expr)) ops.push({ op: 'set', path: ['geometry', 'parameters', i, 'expr'], value: sub(p.expr) });
  });
  geom.features.forEach((f, i) => {
    for (const [k, e] of Object.entries(f.bind || {})) {
      if (refRe(from).test(e)) ops.push({ op: 'set', path: ['geometry', 'features', i, 'bind', k], value: sub(e) });
    }
    if (f.field && f.field.expr && refRe(from).test(f.field.expr)) ops.push({ op: 'set', path: ['geometry', 'features', i, 'field', 'expr'], value: sub(f.field.expr) });
  });
  (geom.constraints || []).forEach((c, i) => {
    if (c.param === from) ops.push({ op: 'set', path: ['geometry', 'constraints', i, 'param'], value: to });
  });
  return ops;
}

/** who reads a parameter: feature names and parameter names */
function readers(geom, n) {
  const out = [];
  for (const p of geom.parameters) if (p.expr && refRe(n).test(p.expr)) out.push(p.name);
  for (const f of geom.features) {
    if (Object.values(f.bind || {}).some((e) => refRe(n).test(e)) || (f.field && refRe(n).test(f.field.expr || ''))) out.push(f.name || f.id);
  }
  for (const c of geom.constraints || []) if (c.param === n) out.push(`dimension ${c.id}`);
  return out;
}

/** "w from 2 to 8 mm, 4 runs" */
export function sweepLine(p, unit) {
  const s = p.sweep || [];
  if (!s.length) return '';
  return `${p.name} from ${fmt(Math.min(...s), 4)} to ${fmt(Math.max(...s), 4)} ${unit}, ${s.length} ${s.length === 1 ? 'run' : 'runs'}`;
}

async function askSweep(p, unit) {
  const v = Number(p.value) || 1;
  const a = h('input', { class: 'field num', value: fmt(p.sweep ? Math.min(...p.sweep) : v / 2, 4), style: { width: '76px' } });
  const b = h('input', { class: 'field num', value: fmt(p.sweep ? Math.max(...p.sweep) : 2 * v, 4), style: { width: '76px' } });
  const n = h('input', { class: 'field num', value: String(p.sweep ? p.sweep.length : 4), style: { width: '50px' } });
  const ok = await dialog(`Sweep ${p.name}`, [
    h('div', { class: 'muted' }, 'The case runs once per value, each on its own copy; Run shows them side by side.'),
    h('label', { class: 'row' }, 'from', h('span', {}, a, ` ${unit}`)), h('label', { class: 'row' }, 'to', h('span', {}, b, ` ${unit}`)),
    h('label', { class: 'row' }, 'runs', n)],
  [{ label: 'Cancel' }, { label: 'Set', primary: true, value: () => Number.isFinite(+a.value) && Number.isFinite(+b.value) && +n.value >= 2 && +n.value <= 12 }]);
  if (!ok) return null;
  const k = Math.round(+n.value);
  return Array.from({ length: k }, (_, i) => Number((+a.value + ((+b.value - +a.value) * i) / (k - 1)).toPrecision(12)));
}

/**
 * The Parameters panel: name, value (a number or a formula) and sweep, one row each;
 * "+" adds one.  `ctx`: {geom, commit(ops, label), unit, dof, counts, onSolve(ops,
 * label)}: `onSolve` lets the caller re-solve paths a changed parameter drives.
 */
export function panel(ctx) {
  const { geom, unit } = ctx;
  const vals = values(geom);
  const box = h('div', { class: 'params', style: { display: 'flex', flexDirection: 'column', gap: '6px' } });
  const add = h('button', { class: 'btn small', 'data-tip': 'Add a named value', 'aria-label': 'Add a parameter' }, '+');
  add.addEventListener('click', async () => {
    const name = freshName(geom, 'p');
    await ctx.commit([{ op: 'insert', path: ['geometry', 'parameters'], value: { name, value: 1, expr: null, sweep: null } }], `new parameter ${name}`);
  });
  box.append(h('div', { style: { display: 'flex', alignItems: 'center' } },
    h('span', { style: { fontSize: ctx.big ? '17px' : '13px', fontWeight: 700, flex: 1 } }, 'Parameters'), add));
  const grid = h('div', { class: 'ptable' },
    h('span', { class: 'ph' }, 'name'), h('span', { class: 'ph' }, 'value'), h('span', { class: 'ph', 'data-tip': 'A sweep runs the case once per value, for a study across a range (for example channel width 2 to 8 mm)' }, 'sweep'));
  if (!geom.parameters.length) {
    box.append(h('div', { class: 'muted', style: { fontSize: '12px' } }, 'None yet. Dimension a shape (D), or press + and type a name.'));
  }
  geom.parameters.forEach((p, i) => {
    const swept = !!(p.sweep && p.sweep.length);
    const nameIn = h('input', { class: 'field mono pin', value: p.name, 'aria-label': `Name of ${p.name}`, spellcheck: 'false' });
    nameIn.addEventListener('keydown', (e) => { if (e.key === 'Enter') nameIn.blur(); if (e.key === 'Escape') { nameIn.value = p.name; nameIn.blur(); } });
    nameIn.addEventListener('change', async () => {
      const to = nameIn.value.trim();
      if (to === p.name) return;
      if (!NAME.test(to) || RESERVED.has(to)) { toast('error', `${to || 'An empty name'}: a letter, then letters, digits or _ (not x, y, T, pi or e).`); nameIn.value = p.name; return; }
      if (geom.parameters.some((q) => q.name === to)) { toast('error', `There is already a parameter called ${to}.`); nameIn.value = p.name; return; }
      await ctx.commit(renameOps(geom, p.name, to), `renamed ${p.name} to ${to}`);
    });
    const shown = p.expr ? p.expr : fmt(Number(p.value), 6);
    const valIn = h('input', { class: `field mono pin${p.expr ? ' fx' : ''}`, value: shown, 'aria-label': `Value of ${p.name}`, spellcheck: 'false',
      'data-tip': p.expr ? `${p.expr} = ${fmt(vals[p.name], 6)}. An expression: the value follows other parameters.` : `A number in ${unit}, or a formula of other parameters` });
    valIn.addEventListener('keydown', (e) => { if (e.key === 'Enter') valIn.blur(); if (e.key === 'Escape') { valIn.value = shown; valIn.blur(); } });
    valIn.addEventListener('change', async () => {
      const t = valIn.value.trim();
      const num = Number(t);
      let ops;
      if (t !== '' && Number.isFinite(num)) {
        ops = [{ op: 'set', path: ['geometry', 'parameters', i, 'value'], value: num }, { op: 'set', path: ['geometry', 'parameters', i, 'expr'], value: null }];
      } else {
        const g2 = JSON.parse(JSON.stringify(geom));
        g2.parameters[i].expr = t;
        const bad = formulaTrouble(t, g2);
        if (bad) { toast('error', `${p.name} = ${t}: ${bad}.`); valIn.value = shown; return; }
        ops = [{ op: 'set', path: ['geometry', 'parameters', i, 'expr'], value: t }];
      }
      await ctx.onSolve(ops, `${p.name} = ${t}`);
    });
    const sw = h('input', { type: 'checkbox', checked: swept || null, 'aria-label': `Sweep ${p.name}` });
    sw.addEventListener('change', async () => {
      if (!sw.checked) { await ctx.commit([{ op: 'set', path: ['geometry', 'parameters', i, 'sweep'], value: null }], `no sweep of ${p.name}`); return; }
      const s = await askSweep(p, unit);
      if (!s) { sw.checked = false; return; }
      await ctx.commit([{ op: 'set', path: ['geometry', 'parameters', i, 'sweep'], value: s }], `sweep ${p.name}`);
    });
    const del = h('button', { class: 'btn small ghost', 'aria-label': `Delete ${p.name}`, 'data-tip': 'Delete this parameter (only when nothing reads it)' }, '×');
    del.addEventListener('click', async () => {
      const who = readers(geom, p.name);
      if (who.length) { toast('info', `${p.name} is read by ${who.slice(0, 3).join(', ')}${who.length > 3 ? '…' : ''}: unbind those first.`); return; }
      await ctx.commit([{ op: 'remove', path: ['geometry', 'parameters', i] }], `deleted ${p.name}`);
    });
    const row = swept ? { background: 'var(--accent-wash)' } : {};
    grid.append(h('span', { class: 'pc', style: row }, nameIn), h('span', { class: 'pc', style: row }, valIn),
      h('span', { class: 'pc', style: { ...row, display: 'flex', alignItems: 'center', gap: '2px' } }, sw, del));
  });
  if (geom.parameters.length) box.append(grid);
  const sweeps = geom.parameters.filter((p) => p.sweep && p.sweep.length);
  for (const p of sweeps) {
    box.append(h('div', { class: 'muted', style: { fontSize: '11px' } },
      h('abbr', { 'data-tip': 'A sweep runs the case once per value, for a study across a range (for example channel width 2 to 8 mm). Start it from Run.' }, 'Sweep'), `: ${sweepLine(p, unit)}`));
  }
  if (ctx.counts) {
    const c = ctx.counts;
    const cbox = h('div', { style: { display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px' } });
    for (const [k, label, tip] of [['horizontal', 'horizontal', 'Horizontal: the edge stays level'], ['vertical', 'vertical', 'Vertical: the edge stays upright'],
      ['parallel', 'parallel', 'Parallel: the two edges keep the same direction'], ['perpendicular', 'perpendicular', 'Perpendicular: the edges meet at 90°'],
      ['equal', 'equal', 'Equal: the two edges keep the same length'], ['fix', 'fixed', 'Fixed: the shape stays where it is'],
      ['dimension', 'dimensions', 'Sizes and positions set by a parameter or a formula']]) {
      if (c[k]) cbox.append(h('div', { style: { display: 'flex', justifyContent: 'space-between' }, 'data-tip': tip }, h('span', {}, label), h('span', { class: 'mono' }, String(c[k]))));
    }
    const dof = ctx.dof;
    cbox.append(h('div', { style: { display: 'flex', justifyContent: 'space-between' } },
      h('abbr', { 'data-tip': 'Degrees of freedom: how many ways the sketch could still move. 0 means fully defined.' }, 'free'),
      h('b', { class: 'mono', style: { color: dof === 0 ? 'var(--ready-ink)' : 'var(--ink)' } }, dof === null || dof === undefined ? '…' : String(dof))));
    box.append(h('div', { class: 'hr' }), h('div', { class: 'label' }, 'Constraints'), cbox);
  }
  return box;
}

/** what each kind of constraint counts, for the panel */
export function counts(geom) {
  const c = { dimension: 0 };
  for (const k of geom.constraints || []) {
    if (k.type === 'distance') c.dimension++;
    else c[k.type] = (c[k.type] || 0) + 1;
  }
  for (const f of geom.features) c.dimension += Object.keys(f.bind || {}).length;
  return c;
}

/**
 * A numeric field the inspector shows, bindable to a formula: typing a number sets it
 * (and drops a binding); typing anything else binds the field to that formula.  The
 * fx mark says it is bound; clicking it unbinds, keeping the value.
 */
export function bindable(ctx, f, field, label, value, tip = null) {
  const { geom, unit } = ctx;
  const i = geom.features.findIndex((x) => x.id === f.id);
  const expr = (f.bind || {})[field];
  const shown = expr !== undefined ? expr : fmt(value, 6);
  const input = h('input', { class: `field num${expr !== undefined ? ' fx' : ''}`, value: shown, inputmode: 'decimal', spellcheck: 'false',
    style: { width: '86px' }, 'aria-label': label,
    'data-tip': expr !== undefined ? `${expr} = ${fmt(value, 6)}` : 'A number, or a formula of the parameters (W - 0.15)' });
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') input.blur(); if (e.key === 'Escape') { input.value = shown; input.blur(); } });
  input.addEventListener('change', async () => {
    const t = input.value.trim();
    const num = Number(t);
    if (t !== '' && Number.isFinite(num)) {
      const ops = [{ op: 'set', path: ['geometry', 'features', i, field], value: num }];
      if (expr !== undefined) ops.push({ op: 'remove', path: ['geometry', 'features', i, 'bind', field] });
      await ctx.commit(ops, `${f.name || f.id}: ${label} ${t}`);
      return;
    }
    const bad = formulaTrouble(t, geom);
    if (bad) { toast('error', `${label} = ${t}: ${bad}.`); input.value = shown; return; }
    await ctx.commit([{ op: 'set', path: ['geometry', 'features', i, 'bind', field], value: t }], `${f.name || f.id}: ${label} = ${t}`);
  });
  const fx = h('button', { class: `fxmark${expr !== undefined ? ' on' : ''}`, 'aria-label': `${label}: formula`,
    'data-tip': expr !== undefined ? 'An expression: the value follows other parameters. Click to keep the number and drop the formula.' : 'Type a formula of the parameters to bind this size' }, 'fx');
  fx.addEventListener('click', async () => {
    if (expr === undefined) { input.focus(); input.select(); return; }
    await ctx.commit([{ op: 'set', path: ['geometry', 'features', i, field], value }, { op: 'remove', path: ['geometry', 'features', i, 'bind', field] }],
      `${f.name || f.id}: ${label} unbound`);
  });
  return h('label', { class: 'row', 'data-tip': tip }, label,
    h('span', { style: { display: 'flex', gap: '4px', alignItems: 'center' } }, fx, input, h('span', { class: 'muted', style: { fontSize: '11px' } }, unit)));
}
