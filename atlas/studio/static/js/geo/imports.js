// DXF and SVG read in the page into closed paths (plan §3.6); STEP and Gmsh meshes
// go to the server.  Each closed loop becomes a path feature: straight lines stay
// lines, circular arcs become arcs by their bulge (DXF's own convention, which the
// engine reads), Bezier curves stay cubic.  The largest loop adds, loops inside
// another cut, and open pieces left over become guides.

/* global paper */

const UNITS = { 1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0, 7: 1000.0, 8: 2.54e-8, 9: 2.54e-5 };

function pairs(text) {
  const lines = text.split(/\r?\n/);
  const out = [];
  for (let i = 0; i + 1 < lines.length; i += 2) out.push([parseInt(lines[i].trim(), 10), lines[i + 1].trim()]);
  return out;
}

/** DXF entities as pieces: {pts: [[x,y],...], kinds: [...], bulges: [...], closed} */
export function readDxf(text) {
  const p = pairs(text);
  let scale = 0.001;                                   // millimetres unless the file says
  for (let i = 0; i < p.length; i++) {
    if (p[i][0] === 9 && p[i][1] === '$INSUNITS' && p[i + 1] && p[i + 1][0] === 70) {
      scale = UNITS[parseInt(p[i + 1][1], 10)] || scale;
    }
  }
  const start = p.findIndex(([c, v]) => c === 2 && v === 'ENTITIES');
  if (start < 0) throw new Error('no ENTITIES section');
  const ents = [];
  let cur = null;
  for (let i = start + 1; i < p.length; i++) {
    const [c, v] = p[i];
    if (c === 0) {
      if (cur) ents.push(cur);
      if (v === 'ENDSEC') { cur = null; break; }
      cur = { type: v, codes: [] };
    } else if (cur) cur.codes.push([c, v]);
  }
  if (cur) ents.push(cur);
  const num = (codes, c, d = 0) => { const f = codes.find((x) => x[0] === c); return f ? parseFloat(f[1]) : d; };
  const pieces = [];
  for (let i = 0; i < ents.length; i++) {
    const e = ents[i];
    const k = e.codes;
    if (e.type === 'LINE') {
      pieces.push({ pts: [[num(k, 10), num(k, 20)], [num(k, 11), num(k, 21)]], kinds: ['line'], bulges: [0], closed: false });
    } else if (e.type === 'ARC') {
      const cx = num(k, 10), cy = num(k, 20), r = num(k, 40);
      const a0 = num(k, 50) * Math.PI / 180, a1 = num(k, 51) * Math.PI / 180;
      let sweep = a1 - a0;
      while (sweep <= 0) sweep += 2 * Math.PI;
      pieces.push({ pts: [[cx + r * Math.cos(a0), cy + r * Math.sin(a0)], [cx + r * Math.cos(a1), cy + r * Math.sin(a1)]],
        kinds: ['arc'], bulges: [Math.tan(sweep / 4)], closed: false });
    } else if (e.type === 'CIRCLE') {
      const cx = num(k, 10), cy = num(k, 20), r = num(k, 40);
      pieces.push({ pts: [[cx + r, cy], [cx - r, cy]], kinds: ['arc', 'arc'], bulges: [1, 1], closed: true });
    } else if (e.type === 'LWPOLYLINE') {
      const pts = [], bulges = [];
      for (let j = 0; j < k.length; j++) {
        if (k[j][0] === 10) { pts.push([parseFloat(k[j][1]), 0]); bulges.push(0); }
        else if (k[j][0] === 20 && pts.length) pts[pts.length - 1][1] = parseFloat(k[j][1]);
        else if (k[j][0] === 42 && pts.length) bulges[bulges.length - 1] = parseFloat(k[j][1]);
      }
      const closed = (num(k, 70) & 1) === 1;
      const m = closed ? pts.length : pts.length - 1;
      pieces.push({ pts, kinds: bulges.slice(0, m).map((b) => (Math.abs(b) > 1e-12 ? 'arc' : 'line')), bulges: bulges.slice(0, m), closed });
    } else if (e.type === 'POLYLINE') {
      const closed = (num(k, 70) & 1) === 1;
      const pts = [], bulges = [];
      for (i += 1; i < ents.length && ents[i].type === 'VERTEX'; i++) {
        pts.push([num(ents[i].codes, 10), num(ents[i].codes, 20)]);
        bulges.push(num(ents[i].codes, 42));
      }
      const m = closed ? pts.length : pts.length - 1;
      pieces.push({ pts, kinds: bulges.slice(0, m).map((b) => (Math.abs(b) > 1e-12 ? 'arc' : 'line')), bulges: bulges.slice(0, m), closed });
    } else if (e.type === 'ELLIPSE') {
      const cx = num(k, 10), cy = num(k, 20), mx = num(k, 11), my = num(k, 21), ratio = num(k, 40, 1);
      const t0 = num(k, 41, 0), t1 = num(k, 42, 2 * Math.PI);
      const R = Math.hypot(mx, my), rot = Math.atan2(my, mx);
      const n = 64, pts = [];
      let span = t1 - t0; if (span <= 0) span += 2 * Math.PI;
      const full = Math.abs(span - 2 * Math.PI) < 1e-9;
      for (let j = 0; j <= (full ? n - 1 : n); j++) {
        const t = t0 + span * j / n;
        const ex = R * Math.cos(t), ey = R * ratio * Math.sin(t);
        pts.push([cx + ex * Math.cos(rot) - ey * Math.sin(rot), cy + ex * Math.sin(rot) + ey * Math.cos(rot)]);
      }
      const m = full ? pts.length : pts.length - 1;
      pieces.push({ pts, kinds: Array(m).fill('line'), bulges: Array(m).fill(0), closed: full });
    } else if (e.type === 'SPLINE') {
      const fit = [], ctrl = [];
      for (let j = 0; j < k.length; j++) {
        if (k[j][0] === 11) fit.push([parseFloat(k[j][1]), 0]);
        else if (k[j][0] === 21 && fit.length) fit[fit.length - 1][1] = parseFloat(k[j][1]);
        else if (k[j][0] === 10) ctrl.push([parseFloat(k[j][1]), 0]);
        else if (k[j][0] === 20 && ctrl.length) ctrl[ctrl.length - 1][1] = parseFloat(k[j][1]);
      }
      const pts = fit.length >= 2 ? fit : ctrl;
      const closed = (num(k, 70) & 1) === 1;
      const m = closed ? pts.length : pts.length - 1;
      if (pts.length >= 2) pieces.push({ pts, kinds: Array(m).fill(fit.length >= 2 ? 'spline' : 'line'), bulges: Array(m).fill(0), closed });
    }
  }
  for (const pc of pieces) pc.pts = pc.pts.map(([x, y]) => [x * scale, y * scale]);
  return chain(pieces);
}

function close(a, b, tol) { return Math.hypot(a[0] - b[0], a[1] - b[1]) <= tol; }

function reverse(pc) {
  const n = pc.pts.length;
  return { pts: pc.pts.slice().reverse(), kinds: pc.kinds.slice().reverse(), bulges: pc.bulges.slice().reverse().map((b) => -b), closed: pc.closed, n };
}

/** open pieces joined end to end into loops; returns {loops, open} */
export function chain(pieces) {
  const all = pieces.flatMap((p) => p.pts);
  const ext = all.length ? Math.max(...all.map((p) => Math.max(Math.abs(p[0]), Math.abs(p[1])))) : 1;
  const tol = 1e-7 * Math.max(ext, 1e-3);
  const loops = pieces.filter((p) => p.closed);
  let open = pieces.filter((p) => !p.closed);
  const done = [];
  while (open.length) {
    let cur = open.shift();
    let grew = true;
    while (grew) {
      grew = false;
      const end = cur.pts[cur.pts.length - 1];
      for (let i = 0; i < open.length; i++) {
        let nx = open[i];
        if (close(nx.pts[nx.pts.length - 1], end, tol)) nx = reverse(nx);
        if (close(nx.pts[0], end, tol)) {
          cur = { pts: cur.pts.concat(nx.pts.slice(1)), kinds: cur.kinds.concat(nx.kinds), bulges: cur.bulges.concat(nx.bulges), closed: false };
          open.splice(i, 1);
          grew = true;
          break;
        }
      }
    }
    if (cur.pts.length > 2 && close(cur.pts[0], cur.pts[cur.pts.length - 1], tol)) {
      loops.push({ pts: cur.pts.slice(0, -1), kinds: cur.kinds, bulges: cur.bulges, closed: true });
    } else done.push(cur);
  }
  return { loops, open: done };
}

/** SVG text as pieces in metres (1 user unit = `unit` metres), y flipped */
export function readSvg(text, unit = 0.001) {
  const scope = new paper.PaperScope();
  scope.setup(new paper.Size(10, 10));
  const item = scope.project.importSVG(text, { insert: false, expandShapes: true });
  const out = [];
  const visit = (it) => {
    if (it instanceof scope.CompoundPath) { it.children.forEach(visit); return; }
    if (it instanceof scope.Path) {
      const segs = it.segments;
      const n = segs.length;
      if (n < 2) return;
      const m = it.closed ? n : n - 1;
      const pts = [], kinds = [], bulges = [], handles = [];
      const W = (pt) => [pt.x * unit, -pt.y * unit];
      for (let k = 0; k < n; k++) pts.push(W(segs[k].point));
      for (let k = 0; k < m; k++) {
        const a = segs[k], b = segs[(k + 1) % n];
        if (a.handleOut.isZero() && b.handleIn.isZero()) { kinds.push('line'); handles.push(null); }
        else {
          const h1 = W(a.point.add(a.handleOut)), h2 = W(b.point.add(b.handleIn));
          kinds.push('cubic'); handles.push([h1[0], h1[1], h2[0], h2[1]]);
        }
        bulges.push(0);
      }
      out.push({ pts, kinds, bulges, handles, closed: it.closed });
      return;
    }
    if (it.children) it.children.forEach(visit);
  };
  visit(item);
  scope.remove();
  return { loops: out.filter((p) => p.closed), open: out.filter((p) => !p.closed) };
}

function signedArea(pts) {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i + 1) % pts.length];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return a / 2;
}

function inside(pt, poly) {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}

/**
 * Loops and open pieces as features: the outermost loops take `role`, a loop inside
 * another cuts, open pieces are guides.  Moved so the drawing starts at the origin
 * when `toOrigin`.  `mk(kind, role, extra)` makes a feature with a fresh id.
 */
export function toFeatures({ loops, open }, role, name, mk, toOrigin) {
  const items = loops.map((l) => ({ ...l, area: Math.abs(signedArea(l.pts)) }));
  items.sort((a, b) => b.area - a.area);
  let dx = 0, dy = 0;
  if (toOrigin) {
    const pts = items.flatMap((l) => l.pts).concat(open.flatMap((l) => l.pts));
    if (pts.length) { dx = -Math.min(...pts.map((p) => p[0])); dy = -Math.min(...pts.map((p) => p[1])); }
  }
  const mv = (pts) => pts.map(([x, y]) => [x + dx, y + dy]);
  const mvh = (hs) => (hs || []).map((h) => (h ? [h[0] + dx, h[1] + dy, h[2] + dx, h[3] + dy] : null));
  const feats = [];
  items.forEach((l, k) => {
    const isHole = items.slice(0, k).some((o) => inside(l.pts[0], o.pts));
    const m = l.pts.length;
    feats.push(mk('path', isHole ? 'cut' : role, {
      name: `${name} ${isHole ? 'hole' : 'shape'} ${k + 1}`, points: mv(l.pts), edges: l.kinds.slice(0, m),
      bulge: l.bulges.slice(0, m), handles: l.handles ? mvh(l.handles) : Array(m).fill(null), closed: true }));
  });
  open.forEach((l, k) => {
    const m = l.pts.length - 1;
    if (m < 1) return;
    feats.push(mk('path', 'guide', { name: `${name} line ${k + 1}`, points: mv(l.pts), edges: l.kinds.slice(0, m),
      bulge: l.bulges.slice(0, m), handles: l.handles ? mvh(l.handles) : Array(m).fill(null), closed: false }));
  });
  return feats;
}
