// The feature tree as Paper.js items: what the page draws, hit-tests and previews.
//
// The server evaluates the same tree as the truth the solvers get (geometry_eval.py);
// this module follows its rules exactly where they meet the eye: an arc by its bulge
// (the engine's DXF convention, positive bowing to the right of travel), a spline as
// the engine's centripetal Catmull-Rom through the neighbouring points, the domain as
// the union of what adds minus the union of what cuts, regions on top in order.
// Coordinates are world units (y up); Paper's are (x, -y).

/* global paper */

export const P = (x, y) => new paper.Point(x, -y);

export function arcMidpoint(a, b, bulge) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const c = Math.hypot(dx, dy);
  if (c === 0) return [a[0], a[1]];
  const rx = dy / c, ry = -dx / c;                     // the right of travel
  return [(a[0] + b[0]) / 2 + bulge * c / 2 * rx, (a[1] + b[1]) / 2 + bulge * c / 2 * ry];
}

/** the bulge of the arc on chord a -> b whose midpoint is as far from it as p */
export function bulgeFromPoint(a, b, p) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const c = Math.hypot(dx, dy);
  if (c === 0) return 0;
  const s = (p[0] - (a[0] + b[0]) / 2) * (dy / c) + (p[1] - (a[1] + b[1]) / 2) * (-dx / c);
  return Math.max(-4, Math.min(4, 2 * s / c));
}

/** centripetal Catmull-Rom between p1 and p2 (the engine's shapes._catmull_rom) */
export function catmullRom(p0, p1, p2, p3, n = 16) {
  const knot = (ti, a, b) => ti + Math.max(Math.sqrt(Math.hypot(b[0] - a[0], b[1] - a[1])), 1e-9);
  const t0 = 0, t1 = knot(t0, p0, p1), t2 = knot(t1, p1, p2), t3 = knot(t2, p2, p3);
  const out = [];
  for (let i = 0; i <= n; i++) {
    const tt = t1 + (t2 - t1) * (i / n);
    const lerp = (a, b, ta, tb) => [(tb - tt) / (tb - ta) * a[0] + (tt - ta) / (tb - ta) * b[0],
      (tb - tt) / (tb - ta) * a[1] + (tt - ta) / (tb - ta) * b[1]];
    const a1 = lerp(p0, p1, t0, t1), a2 = lerp(p1, p2, t1, t2), a3 = lerp(p2, p3, t2, t3);
    const b1 = lerp(a1, a2, t0, t2), b2 = lerp(a2, a3, t1, t3);
    out.push(lerp(b1, b2, t1, t2));
  }
  return out;
}

/** a path feature's points and edges as a Paper path (world -> Paper) */
function pathItem(f) {
  const pts = f.points;
  const n = pts.length;
  const m = f.closed ? n : n - 1;
  const path = new paper.Path({ insert: false });
  if (!n) return path;
  path.moveTo(P(...pts[0]));
  for (let k = 0; k < m; k++) {
    const a = pts[k], b = pts[(k + 1) % n];
    const kind = (f.edges || [])[k] || 'line';
    if (kind === 'arc' && Math.abs((f.bulge || [])[k] || 0) > 1e-12) {
      const mid = arcMidpoint(a, b, f.bulge[k]);
      path.arcTo(P(...mid), P(...b));
    } else if (kind === 'cubic' && f.handles && f.handles[k]) {
      const h = f.handles[k];
      path.cubicCurveTo(P(h[0], h[1]), P(h[2], h[3]), P(...b));
    } else if (kind === 'spline' && n >= 3) {
      const q0 = pts[(k - 1 + n) % n], q3 = pts[(k + 2) % n];
      const s = catmullRom(f.closed || k > 0 ? q0 : a, a, b, f.closed || k + 2 < n ? q3 : b, 16);
      for (let i = 1; i < s.length; i++) path.lineTo(P(...s[i]));
    } else {
      path.lineTo(P(...b));
    }
  }
  if (f.closed) path.closePath();
  return path;
}

/** One feature as a Paper item, in Paper coordinates; combines through their inputs. */
export function itemOf(f, byId, depth = 0) {
  if (!f || depth > 32) return new paper.Path({ insert: false });
  if (f.kind === 'rect') {
    const r = Math.min(f.r || 0, f.w / 2, f.h / 2);
    return new paper.Path.Rectangle({ point: [f.x, -(f.y + f.h)], size: [f.w, f.h], radius: r, insert: false });
  }
  if (f.kind === 'ellipse') {
    return new paper.Path.Ellipse({ center: [f.cx, -f.cy], radius: [f.rx, f.ry], insert: false });
  }
  if (f.kind === 'path') return pathItem(f);
  if (f.kind === 'combine') {
    const parts = (f.inputs || []).map((i) => itemOf(byId[i], byId, depth + 1));
    let g = parts[0];
    for (const p of parts.slice(1)) {
      g = f.op === 'union' ? g.unite(p, { insert: false })
        : f.op === 'subtract' ? g.subtract(p, { insert: false }) : g.intersect(p, { insert: false });
    }
    return g || new paper.Path({ insert: false });
  }
  return new paper.Path({ insert: false });
}

export function byIdOf(geom) {
  const out = {};
  for (const f of geom.features) out[f.id] = f;
  return out;
}

export function consumed(geom) {
  const s = new Set();
  for (const f of geom.features) if (f.kind === 'combine') for (const i of f.inputs) s.add(i);
  return s;
}

/** the features that act: not inputs of a combine, not hidden */
export function topFeatures(geom) {
  const c = consumed(geom);
  return geom.features.filter((f) => !c.has(f.id) && !f.hidden);
}

/** the domain: union of what adds, minus the union of what cuts (Paper coords) */
export function domainItem(geom, byId = byIdOf(geom)) {
  const top = topFeatures(geom);
  let d = null;
  for (const f of top.filter((x) => x.role === 'add')) {
    const it = itemOf(f, byId);
    d = d ? d.unite(it, { insert: false }) : it;
  }
  if (!d) return null;
  for (const f of top.filter((x) => x.role === 'cut')) d = d.subtract(itemOf(f, byId), { insert: false });
  return d;
}

export function areaOf(item) { return item ? Math.abs(item.area) : 0; }

/** world bounding box of a feature: [x0, y0, x1, y1] */
export function bboxOf(f, byId) {
  const b = itemOf(f, byId).bounds;
  return [b.left, -b.bottom, b.right, -b.top];
}

/** the few numbers that say what a feature is, for the tree (mono) */
export function describe(f, fmt) {
  if (f.kind === 'rect') return `${fmt(f.w)} × ${fmt(f.h)}`;
  if (f.kind === 'ellipse') return Math.abs(f.rx - f.ry) < 1e-12 ? `Ø ${fmt(2 * f.rx)}` : `${fmt(2 * f.rx)} × ${fmt(2 * f.ry)}`;
  if (f.kind === 'path') return `${f.points.length} pts`;
  if (f.kind === 'combine') return { union: '∪', subtract: '−', intersect: '∩' }[f.op] + ` ${f.inputs.length}`;
  return '';
}

/** a feature moved by (dx, dy), as a copy */
export function translated(f, dx, dy) {
  const g = JSON.parse(JSON.stringify(f));
  g.x += dx; g.y += dy; g.cx += dx; g.cy += dy;
  g.points = (g.points || []).map(([x, y]) => [x + dx, y + dy]);
  g.handles = (g.handles || []).map((h) => (h ? [h[0] + dx, h[1] + dy, h[2] + dx, h[3] + dy] : null));
  return g;
}

/** a fresh feature id not in the tree */
export function nextId(geom) {
  const ids = new Set(geom.features.map((f) => f.id));
  let k = ids.size + 1;
  while (ids.has(`f${k}`)) k++;
  return `f${k}`;
}

/** a feature with every field the server's model has, defaults filled */
export function blank(kind, id, role, extra = {}) {
  return Object.assign({ id, kind, role, name: '', material: null, hidden: false,
    x: 0, y: 0, w: 0, h: 0, r: 0, cx: 0, cy: 0, rx: 0, ry: 0,
    points: [], edges: [], bulge: [], handles: [], closed: true, op: null, inputs: [], params: {} }, extra);
}
