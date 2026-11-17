// Snapping while drawing and dragging (plan §3.2): the grid when Grid is on, and
// smart guides when Guides is on, to the edges, centres and midpoints of the other
// shapes and the grid's own edges.  A snap is drawn as a cyan dashed line with a
// diamond at the snapped point, and named in the inspector (SNAPPED TO).

import { bboxOf, byIdOf, topFeatures } from './model.js';

/** candidate lines: {x|y, value, label} from every shape except `skip` */
export function candidates(geom, skip = new Set(), grid = null) {
  const byId = byIdOf(geom);
  const xs = [], ys = [];
  for (const f of topFeatures(geom)) {
    if (skip.has(f.id)) continue;
    const [x0, y0, x1, y1] = bboxOf(f, byId);
    const name = f.name || f.id;
    xs.push({ v: x0, label: `${name} left edge` }, { v: (x0 + x1) / 2, label: `${name} centre` },
      { v: x1, label: `${name} right edge` });
    ys.push({ v: y0, label: `${name} bottom edge` }, { v: (y0 + y1) / 2, label: `${name} centre line` },
      { v: y1, label: `${name} top edge` });
    if (f.kind === 'path') {
      f.points.forEach((p, k) => {
        xs.push({ v: p[0], label: `${name} point ${k + 1}` });
        ys.push({ v: p[1], label: `${name} point ${k + 1}` });
      });
    }
  }
  if (grid) {
    xs.push({ v: 0, label: 'grid left edge' }, { v: grid.width, label: 'grid right edge' });
    ys.push({ v: 0, label: 'grid bottom edge' }, { v: grid.height, label: 'grid top edge' });
  }
  return { xs, ys };
}

/**
 * Snap a world point.  `opts`: {grid: step|null, guides: {xs, ys}|null, tol: world
 * units}.  Returns {x, y, lines: [{axis, v, label}]}.
 */
export function snap(x, y, opts) {
  let sx = x, sy = y;
  const lines = [];
  if (opts.guides) {
    let best = null;
    for (const c of opts.guides.xs) {
      const d = Math.abs(c.v - x);
      if (d <= opts.tol && (!best || d < best.d)) best = { d, c };
    }
    if (best) { sx = best.c.v; lines.push({ axis: 'x', v: sx, label: best.c.label }); }
    best = null;
    for (const c of opts.guides.ys) {
      const d = Math.abs(c.v - y);
      if (d <= opts.tol && (!best || d < best.d)) best = { d, c };
    }
    if (best) { sy = best.c.v; lines.push({ axis: 'y', v: sy, label: best.c.label }); }
  }
  if (opts.grid) {
    if (!lines.some((l) => l.axis === 'x')) sx = Math.round(x / opts.grid) * opts.grid;
    if (!lines.some((l) => l.axis === 'y')) sy = Math.round(y / opts.grid) * opts.grid;
  }
  return { x: sx, y: sy, lines };
}

/** a segment's end locked to steps of `deg` degrees from its start */
export function lockAngle(ax, ay, x, y, deg = 15) {
  const dx = x - ax, dy = y - ay;
  const L = Math.hypot(dx, dy);
  if (L === 0) return { x, y };
  const step = (deg * Math.PI) / 180;
  const a = Math.round(Math.atan2(dy, dx) / step) * step;
  return { x: ax + L * Math.cos(a), y: ay + L * Math.sin(a) };
}
