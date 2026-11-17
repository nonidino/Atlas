// The Dimension tool (D) and what it draws (plan §3.2, §3.3).  Clicking an edge of a
// rectangle makes its width or height a parameter; clicking a hole makes its diameter
// one; clicking two points makes their horizontal or vertical distance one, and binds
// the second shape's position to the first's (`x_h = W − 0.15` style), so changing W
// moves it.  A straight edge of a path becomes a driving length, solved by planegcs.
// Every binding is drawn as a dimension line with its label: `W = 0.600` for a bare
// parameter, `0.150 · fx` for a formula.

/* global paper */
import { h, fmt, dialog, toast } from '../ui.js';
import { P, pointOf } from './model.js';
import { freshName, values, bareName } from './params.js';
import { dragPoint, solver, sketch } from './gcs.js';

const num = (v) => {
  const s = Number(v.toPrecision(12)).toString();
  return s.includes('e') ? `(${s})` : s;
};

/** a field of a feature as a formula: its binding, or its number */
function term(f, field) {
  const b = (f.bind || {})[field];
  return b !== undefined ? (bareName(b) || `(${b})`) : num(f[field]);
}

/** a point's coordinates as formulas of the feature's fields */
function pointTerms(f, key) {
  if (f.kind === 'ellipse') return [term(f, 'cx'), term(f, 'cy')];
  if (f.kind !== 'rect') { const p = pointOf(f, key); return p ? [num(p[0]), num(p[1])] : null; }
  const X = term(f, 'x'), Y = term(f, 'y'), Wd = term(f, 'w'), Ht = term(f, 'h');
  const plus = (a, b) => (a === '0' ? b : `${a} + ${b}`);
  const xs = { c0: X, c3: X, e3: X, c1: plus(X, Wd), c2: plus(X, Wd), e1: plus(X, Wd), e0: plus(X, `${Wd}/2`), e2: plus(X, `${Wd}/2`) };
  const ys = { c0: Y, c1: Y, e0: Y, c2: plus(Y, Ht), c3: plus(Y, Ht), e2: plus(Y, Ht), e1: plus(Y, `${Ht}/2`), e3: plus(Y, `${Ht}/2`) };
  return xs[key] !== undefined ? [xs[key], ys[key]] : null;
}

/** how far a point sits from the field that places the feature: rect x/y, ellipse cx/cy */
function offsetTerms(f, key) {
  if (f.kind === 'ellipse') return ['', ''];
  const Wd = term(f, 'w'), Ht = term(f, 'h');
  const ox = { c0: '', c3: '', e3: '', c1: Wd, c2: Wd, e1: Wd, e0: `${Wd}/2`, e2: `${Wd}/2` }[key];
  const oy = { c0: '', c1: '', e0: '', c2: Ht, c3: Ht, e2: Ht, e1: `${Ht}/2`, e3: `${Ht}/2` }[key];
  return [ox, oy];
}

function nextConstraintId(geom) {
  const ids = new Set((geom.constraints || []).map((c) => c.id));
  let k = ids.size + 1;
  while (ids.has(`k${k}`)) k++;
  return `k${k}`;
}

async function askValue(title, name, v, unit) {
  const input = h('input', { class: 'field num', value: fmt(v, 6), style: { width: '100px' } });
  const ok = await dialog(title, [h('label', { class: 'row' }, h('span', { class: 'mono' }, `${name} =`), h('span', {}, input, ` ${unit}`))],
    [{ label: 'Cancel' }, { label: 'Set', primary: true, value: () => Number(input.value) > 0 }]);
  return ok ? Number(input.value) : null;
}

/**
 * One click of the Dimension tool on a target: {fid, key} (a handle), {fid, edge}
 * (a path's edge), {fid} (a shape's body).  Returns ops and a label, or null.  An
 * already-dimensioned size asks for its new value instead.
 */
export async function sizeDimension(ctx, target) {
  const { geom, unit } = ctx;
  const f = geom.features.find((x) => x.id === target.fid);
  if (!f) return null;
  const i = geom.features.indexOf(f);
  const params = geom.parameters || [];
  const edit = async (field, scale = 1) => {
    const bn = bareName((f.bind || {})[field]) || (/^([A-Za-z_][A-Za-z0-9_]*)\s*\/\s*2$/.exec((f.bind || {})[field] || '') || [])[1];
    const pi = params.findIndex((p) => p.name === bn);
    if (pi < 0) { toast('info', `${f.name || f.id}'s ${field} follows a formula: change it in the inspector.`); return null; }
    if (params[pi].expr) { toast('info', `${bn} is a formula (${params[pi].expr}): change it in Parameters.`); return null; }
    const v = await askValue(`Dimension ${bn}`, bn, Number(params[pi].value), unit);
    if (v === null) return null;
    return { ops: [{ op: 'set', path: ['geometry', 'parameters', pi, 'value'], value: v }], label: `${bn} = ${fmt(v, 6)}`, solve: true };
  };
  if (f.kind === 'rect' && target.key && /^e[0-3]$/.test(target.key)) {
    const field = target.key === 'e0' || target.key === 'e2' ? 'w' : 'h';
    if ((f.bind || {})[field] !== undefined) return edit(field);
    const first = !params.some((p) => p.name === (field === 'w' ? 'W' : 'H'));
    const name = freshName(geom, field === 'w' ? (first ? 'W' : 'w') : (first ? 'H' : 'h'));
    return { ops: [{ op: 'insert', path: ['geometry', 'parameters'], value: { name, value: Number(f[field].toPrecision(12)), expr: null, sweep: null } },
      { op: 'set', path: ['geometry', 'features', i, 'bind', field], value: name }], label: `dimension ${name} on ${f.name || f.id}` };
  }
  if (f.kind === 'ellipse' && !target.key) {
    const circle = Math.abs(f.rx - f.ry) < 1e-12;
    if ((f.bind || {}).rx !== undefined) return edit('rx');
    if (circle) {
      const name = freshName(geom, 'd');
      return { ops: [{ op: 'insert', path: ['geometry', 'parameters'], value: { name, value: Number((2 * f.rx).toPrecision(12)), expr: null, sweep: null } },
        { op: 'set', path: ['geometry', 'features', i, 'bind', 'rx'], value: `${name}/2` },
        { op: 'set', path: ['geometry', 'features', i, 'bind', 'ry'], value: `${name}/2` }], label: `dimension ${name} on ${f.name || f.id}` };
    }
    const a = freshName(geom, 'a');
    const g2 = { parameters: [...params, { name: a }] };
    const b = freshName(g2, 'b');
    return { ops: [{ op: 'insert', path: ['geometry', 'parameters'], value: { name: a, value: Number((2 * f.rx).toPrecision(12)), expr: null, sweep: null } },
      { op: 'insert', path: ['geometry', 'parameters'], value: { name: b, value: Number((2 * f.ry).toPrecision(12)), expr: null, sweep: null } },
      { op: 'set', path: ['geometry', 'features', i, 'bind', 'rx'], value: `${a}/2` },
      { op: 'set', path: ['geometry', 'features', i, 'bind', 'ry'], value: `${b}/2` }], label: `dimension ${a} × ${b} on ${f.name || f.id}` };
  }
  if ((f.kind === 'path') && target.edge !== undefined) {
    const k = target.edge;
    const n = f.points.length;
    if ((f.edges[k] || 'line') !== 'line') { toast('info', 'Dimension a straight edge; a curve has no single length here.'); return null; }
    const refs = [`${f.id}:p${k}`, `${f.id}:p${(k + 1) % n}`];
    const have = (geom.constraints || []).findIndex((c) => c.type === 'distance' && c.value != null && c.refs[0] === refs[0] && c.refs[1] === refs[1]);
    if (have >= 0 && geom.constraints[have].param) {
      const pi = params.findIndex((p) => p.name === geom.constraints[have].param);
      if (pi < 0) return null;
      const v = await askValue(`Dimension ${params[pi].name}`, params[pi].name, Number(params[pi].value), unit);
      if (v === null) return null;
      return { ops: [{ op: 'set', path: ['geometry', 'parameters', pi, 'value'], value: v }], label: `${params[pi].name} = ${fmt(v, 6)}`, solve: true };
    }
    const a = f.points[k], b = f.points[(k + 1) % n];
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const name = freshName(geom, 'L');
    return { ops: [{ op: 'insert', path: ['geometry', 'parameters'], value: { name, value: Number(L.toPrecision(12)), expr: null, sweep: null } },
      { op: 'insert', path: ['geometry', 'constraints'], value: { id: nextConstraintId(geom), type: 'distance', refs, value: Number(L.toPrecision(12)), param: name } }],
    label: `dimension ${name} on ${f.name || f.id}` };
  }
  return null;
}

/**
 * Two points: A {fid, key} then B {fid, key}.  B's position is bound to A's plus a new
 * parameter along the axis the two points differ most in (both with Shift).
 */
export function pointDimension(ctx, A, B, both = false) {
  const { geom } = ctx;
  const fa = geom.features.find((x) => x.id === A.fid), fb = geom.features.find((x) => x.id === B.fid);
  if (!fa || !fb || fa.id === fb.id) { toast('info', 'Pick a point on another shape.'); return null; }
  if (!['rect', 'ellipse'].includes(fb.kind)) { toast('info', 'The second point must be on a rectangle or a hole: its position is what the dimension sets.'); return null; }
  const pa = pointOf(fa, A.key), pb = pointOf(fb, B.key);
  const ta = pointTerms(fa, A.key), off = offsetTerms(fb, B.key);
  if (!pa || !pb || !ta) return null;
  const i = geom.features.indexOf(fb);
  const dx = pb[0] - pa[0], dy = pb[1] - pa[1];
  const axes = both ? ['x', 'y'] : [Math.abs(dx) >= Math.abs(dy) ? 'x' : 'y'];
  const ops = [];
  let g2 = { parameters: [...(geom.parameters || [])], constraints: [...(geom.constraints || [])] };
  const tag = fb.role === 'cut' ? 'h' : fb.id;
  const made = [];
  for (const ax of axes) {
    const d = ax === 'x' ? dx : dy;
    const field = fb.kind === 'ellipse' ? (ax === 'x' ? 'cx' : 'cy') : ax;
    const name = freshName(g2, `${ax}_${tag}`);
    g2.parameters.push({ name });
    const base = ax === 'x' ? ta[0] : ta[1];
    const o = ax === 'x' ? off[0] : off[1];
    let expr = `${base} ${d >= 0 ? '+' : '-'} ${name}`;
    if (o) expr += ` - ${o.includes('/') || o.includes(' ') ? `(${o})` : o}`;
    if (expr.startsWith('0 + ')) expr = expr.slice(4);
    ops.push({ op: 'insert', path: ['geometry', 'parameters'], value: { name, value: Number(Math.abs(d).toPrecision(12)), expr: null, sweep: null } });
    ops.push({ op: 'set', path: ['geometry', 'features', i, 'bind', field], value: expr });
    const id = nextConstraintId(g2);
    const c = { id, type: 'distance', refs: [`${fa.id}:${A.key}`, `${fb.id}:${B.key}`, ax], value: null, param: name };
    g2.constraints.push(c);
    ops.push({ op: 'insert', path: ['geometry', 'constraints'], value: c });
    made.push(name);
  }
  return { ops, label: `dimension ${made.join(', ')}: ${fb.name || fb.id} from ${fa.name || fa.id}` };
}

/**
 * Paths whose driving lengths changed: their points solved again by planegcs with
 * the new values, as ops.  `geom` already holds the new parameters (resolved here).
 */
export async function drivenOps(geom) {
  const vals = values(geom);
  const ops = [];
  let mod = null;
  for (const [i, f] of geom.features.entries()) {
    if (f.kind !== 'path') continue;
    const own = (geom.constraints || []).map((c, ci) => [c, ci]).filter(([c]) => c.refs.length && c.refs.every((r) => r.startsWith(`${f.id}:`)));
    const driven = own.filter(([c]) => c.type === 'distance' && c.param && c.value != null && vals[c.param] !== undefined && Math.abs(vals[c.param] - c.value) > 1e-12 * Math.max(1, Math.abs(c.value)));
    if (!driven.length) continue;
    mod = mod || await solver();
    const cons = JSON.parse(JSON.stringify(geom.constraints));
    for (const [c, ci] of driven) { cons[ci].value = vals[c.param]; ops.push({ op: 'set', path: ['geometry', 'constraints', ci, 'value'], value: vals[c.param] }); }
    // solve with the path's first point held, so it grows from there
    const pts = await dragPoint({ features: [f], constraints: cons }, f, 0, f.points[0][0], f.points[0][1], mod);
    if (pts) ops.push({ op: 'set', path: ['geometry', 'features', i, 'points'], value: pts });
    else toast('warning', `${f.name || f.id}: its constraints cannot all hold at the new size.`);
  }
  return ops;
}

/** a path with its own constraints solved after an edit to its constraints */
export async function settledPoints(geom, f) {
  const own = (geom.constraints || []).filter((c) => c.refs.length && c.refs.every((r) => r.startsWith(`${f.id}:`)));
  if (!own.length) return null;
  return dragPoint({ features: [f], constraints: own }, f, 0, f.points[0][0], f.points[0][1]);
}

// -- drawing --------------------------------------------------------------------------

function dimLine(st, a, b, off, label, z, inked = false) {
  // a and b in world units; off: the offset of the dimension line (world), along the normal
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const L = Math.hypot(dx, dy);
  if (L <= 0) return;
  const nx = -dy / L, ny = dx / L;
  const a2 = [a[0] + nx * off, a[1] + ny * off], b2 = [b[0] + nx * off, b[1] + ny * off];
  const ink = new paper.Color('#5B3FC4');
  const gap = Math.sign(off) * 3 / z;
  for (const [p, q] of [[a, a2], [b, b2]]) {
    const ext = new paper.Path.Line({ from: P(p[0] + nx * gap, p[1] + ny * gap), to: P(q[0] + nx * gap, q[1] + ny * gap), insert: false });
    ext.strokeColor = ink; ext.strokeWidth = 1; ext.strokeScaling = false;
    st.over.addChild(ext);
  }
  const line = new paper.Path.Line({ from: P(...a2), to: P(...b2), insert: false });
  line.strokeColor = ink; line.strokeWidth = 1.2; line.strokeScaling = false;
  st.over.addChild(line);
  const s = 6 / z;
  for (const [p, dir] of [[a2, 1], [b2, -1]]) {
    const ux = (dx / L) * dir, uy = (dy / L) * dir;
    const tri = new paper.Path({ segments: [P(...p), P(p[0] + ux * s + nx * s * 0.45, p[1] + uy * s + ny * s * 0.45), P(p[0] + ux * s - nx * s * 0.45, p[1] + uy * s - ny * s * 0.45)], closed: true, insert: false });
    tri.fillColor = ink;
    st.over.addChild(tri);
  }
  tag(st, [(a2[0] + b2[0]) / 2, (a2[1] + b2[1]) / 2], label, z, inked);
}

function tag(st, at, label, z, inked = false) {
  const t = new paper.PointText({ point: P(at[0], at[1]).add(new paper.Point(0, 4 / z)), content: label, justification: 'center',
    fontFamily: 'JetBrains Mono, monospace', fontSize: 11 / z, fillColor: inked ? '#191927' : '#3B2690', insert: false });
  const bg = new paper.Path.Rectangle({ rectangle: t.bounds.expand(10 / z, 6 / z), radius: 5 / z, insert: false });
  bg.fillColor = '#FFFFFF'; bg.strokeColor = new paper.Color(inked ? '#8E72EE' : '#5B3FC4'); bg.strokeWidth = inked ? 2 : 1; bg.strokeScaling = false;
  st.over.addChild(bg); st.over.addChild(t);
}

function labelOf(expr, v, prefix = '') {
  const bn = bareName(expr);
  const half = /^([A-Za-z_][A-Za-z0-9_]*)\s*\/\s*2$/.exec(expr || '');
  if (bn) return `${prefix}${bn} = ${fmt(v, 3)}`;
  if (half) return `${prefix}${half[1]} = ${fmt(2 * v, 3)}`;
  return `${prefix}${fmt(v, 3)} · fx`;
}

const MARK = { horizontal: '—', vertical: '|', parallel: '∥', perpendicular: '⊥', equal: '=', fix: '⚓' };

/** every dimension and constraint mark of the drawing, on the overlay */
export function drawDims(editor) {
  const st = editor.stage;
  const z = st.view.zoom;
  const geom = editor.geom;
  const byId = Object.fromEntries(geom.features.map((f) => [f.id, f]));
  const off = 18 / z;
  for (const f of geom.features) {
    if (f.hidden) continue;
    const b = f.bind || {};
    if (f.kind === 'rect') {
      if (b.w !== undefined) dimLine(st, [f.x, f.y + f.h], [f.x + f.w, f.y + f.h], off, labelOf(b.w, f.w), z);
      if (b.h !== undefined) dimLine(st, [f.x, f.y], [f.x, f.y + f.h], off, labelOf(b.h, f.h), z);
      if (b.r !== undefined && f.r > 0) tag(st, [f.x + f.w + 0.6 * off, f.y + f.h + 0.6 * off], labelOf(b.r, f.r, 'r '), z);
    } else if (f.kind === 'ellipse' && b.rx !== undefined) {
      const circle = Math.abs(f.rx - f.ry) < 1e-12;
      const at = [f.cx + f.rx * Math.SQRT1_2 + off, f.cy + f.ry * Math.SQRT1_2 + off];
      const leader = new paper.Path.Line({ from: P(f.cx + f.rx * Math.SQRT1_2, f.cy + f.ry * Math.SQRT1_2), to: P(...at), insert: false });
      leader.strokeColor = '#5B3FC4'; leader.strokeWidth = 1.2; leader.strokeScaling = false;
      st.over.addChild(leader);
      tag(st, at, circle ? labelOf(b.rx, f.rx, 'Ø ') : `${fmt(2 * f.rx, 3)} × ${fmt(2 * f.ry, 3)}`, z);
    } else if (f.kind === 'channel' && b.w !== undefined && f.points.length >= 2) {
      const [p, q] = [f.points[0], f.points[1]];
      tag(st, [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2], labelOf(b.w, f.w, 'w '), z, true);
    }
  }
  const vals = values(geom);
  for (const c of geom.constraints || []) {
    const [ra, rb, ax] = c.refs;
    const pt = (r) => { const [fid, key] = String(r || '').split(':'); return pointOf(byId[fid], key); };
    if (c.type === 'distance' && c.value == null && ax) {
      const a = pt(ra), b = pt(rb);
      if (!a || !b) continue;
      const v = vals[c.param];
      const fb = byId[String(rb).split(':')[0]];
      const expr = fb && (fb.bind || {})[fb.kind === 'ellipse' ? `c${ax}` : ax];
      const lab = v !== undefined ? `${c.param} = ${fmt(v, 3)}` : (expr ? `${expr}` : c.param);
      if (ax === 'x') dimLine(st, [a[0], b[1]], [b[0], b[1]], -off * 1.6, lab, z);
      else dimLine(st, [b[0], a[1]], [b[0], b[1]], off * 1.6, lab, z);
    } else if (c.type === 'distance' && c.value != null) {
      const a = pt(ra), b = pt(rb);
      if (a && b) dimLine(st, a, b, off, c.param ? `${c.param} = ${fmt(vals[c.param] ?? c.value, 3)}` : fmt(c.value, 3), z);
    } else if (MARK[c.type]) {
      for (const r of c.refs) {
        const [fid, key] = String(r).split(':');
        const f = byId[fid];
        if (!f) continue;
        let at = null;
        const m = /^e(\d+)$/.exec(key || '');
        if (m && f.kind === 'path') {
          const k = +m[1], n = f.points.length;
          const a = f.points[k], b = f.points[(k + 1) % n];
          if (a && b) at = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        } else if (c.type === 'fix') {
          at = f.kind === 'rect' ? [f.x, f.y] : (f.kind === 'ellipse' ? [f.cx, f.cy] : (f.points || [])[0]);
        }
        if (!at) continue;
        const t = new paper.PointText({ point: P(at[0], at[1]).add(new paper.Point(7 / z, -7 / z)), content: MARK[c.type],
          fontFamily: 'JetBrains Mono, monospace', fontSize: 11 / z, fontWeight: 'bold', fillColor: '#5B3FC4', insert: false });
        st.over.addChild(t);
      }
    }
  }
}

/** the edge of a path nearest a world point, and its distance */
export function nearestEdge(f, w) {
  const n = f.points.length;
  const m = f.closed ? n : n - 1;
  let best = null;
  for (let k = 0; k < m; k++) {
    const a = f.points[k], b = f.points[(k + 1) % n];
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const L2 = dx * dx + dy * dy;
    const t = L2 > 0 ? Math.max(0, Math.min(1, ((w.x - a[0]) * dx + (w.y - a[1]) * dy) / L2)) : 0;
    const d = Math.hypot(a[0] + t * dx - w.x, a[1] + t * dy - w.y);
    if (!best || d < best.d) best = { edge: k, d };
  }
  return best;
}

export { sketch };
