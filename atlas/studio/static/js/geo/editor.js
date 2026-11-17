// The geometry canvas: tools, selection, handles, snapping, previews, and the
// commit on release (plan §3.2).  Everything moves locally at frame rate; nothing
// reaches the server until a gesture ends, and then as one edit ("rebuild on
// commit, not on drag").  A tool stays on until V (or Esc twice).

/* global paper */
import { P, itemOf, byIdOf, topFeatures, domainItem, arcMidpoint, bulgeFromPoint, translated,
  nextId, blank, bboxOf, pointOf } from './model.js';
import { drawDims, nearestEdge } from './dims.js';
import { candidates, snap, lockAngle } from './snap.js';
import { cssVar } from '../ui.js';
import { SWATCHES } from '../canvas.js';

const HANDLE_PX = 5;

export class Editor {
  constructor(stage, host) {
    this.stage = stage;
    this.host = host;                 // {commit(ops, label), changed(), notify(kind, text)}
    this.geom = { parameters: [], features: [] };
    this.grid = { width: 1, height: 1, dx: 1 };
    this.materials = {};
    this.tool = 'select';
    this.role = 'add';
    this.opts = { grid: false, guides: true, tangent: true, dims: true };
    this.dimA = null;                 // the Dimension tool's first point
    this.sel = new Set();
    this.drag = null;                 // an active gesture
    this.draft = null;                // a shape being drawn
    this.pen = null;                  // the pen's points so far
    this.snapped = [];                // the guides of the last snap, for the inspector
    this.measure = null;
    this.colourBy = 'material';
    this.checkRaster = null;          // the Check view's cells
    this.pins = [];
    this.pending = null;              // a server state that arrived mid-gesture
    this.typed = '';                  // digits typed while drawing (Tab, or the pen's length)
    this.install();
  }

  // -- data in ----------------------------------------------------------------------
  setGeom(geom, drawing, materials) {
    if (this.drag || this.draft || this.pen) { this.pending = { geom, drawing, materials }; return; }
    this.geom = JSON.parse(JSON.stringify(geom));
    this.grid = { width: drawing.width, height: drawing.height, dx: drawing.dx, nx: drawing.nx, ny: drawing.ny };
    this.materials = materials || {};
    const ids = new Set(this.geom.features.map((f) => f.id));
    for (const id of [...this.sel]) if (!ids.has(id)) this.sel.delete(id);
    this.stage.setBounds(0, 0, Math.max(drawing.width, 1e-9), Math.max(drawing.height, 1e-9));
    this.redraw();
  }

  flushPending() {
    if (this.pending && !this.drag && !this.draft && !this.pen) {
      const p = this.pending; this.pending = null;
      this.setGeom(p.geom, p.drawing, p.materials);
    }
  }

  get byId() { return byIdOf(this.geom); }
  feature(id) { return this.geom.features.find((f) => f.id === id); }
  index(id) { return this.geom.features.findIndex((f) => f.id === id); }

  /** fields a dimension, a formula or a fix holds: a drag never changes them */
  locked(f) {
    const out = new Set(Object.keys(f.bind || {}));
    for (const c of this.geom.constraints || []) {
      if (c.type === 'fix' && c.refs.includes(f.id)) ['x', 'y', 'cx', 'cy', 'points'].forEach((k) => out.add(k));
    }
    return out;
  }

  /** a path's own constraints (all their references on it) */
  ownConstraints(f) {
    return (this.geom.constraints || []).filter((c) => c.refs.length && c.refs.every((r) => r.startsWith(`${f.id}:`)) && c.type !== 'fix');
  }

  /** a moved copy with what is locked put back */
  hold(f, o) {
    const lk = this.locked(f);
    for (const k of lk) if (k in o) f[k] = JSON.parse(JSON.stringify(o[k]));
    if (lk.has('points')) {
      if (o.handles) f.handles = JSON.parse(JSON.stringify(o.handles));
      if (o.items) f.items = JSON.parse(JSON.stringify(o.items));
    }
    return f;
  }

  materialColour(name) {
    const names = Object.keys(this.materials);
    if (this.colourBy === 'kappa') {
      const k = (m) => (this.materials[m] || {}).k ?? (this.materials[m] || {}).sigma;
      const vals = names.map(k).filter((v) => v > 0);
      const v = k(name);
      if (v > 0 && vals.length) {
        const lo = Math.log10(Math.min(...vals)), hi = Math.log10(Math.max(...vals));
        const t = hi > lo ? (Math.log10(v) - lo) / (hi - lo) : 0.5;
        return new paper.Color({ hue: 260 - 90 * t, saturation: 0.75, lightness: 0.7 - 0.25 * t }).toCSS(true);
      }
    }
    const i = Math.max(0, names.indexOf(name));
    return SWATCHES[(i + 1) % SWATCHES.length];
  }

  // -- drawing ------------------------------------------------------------------------
  redraw() {
    const st = this.stage;
    st.activate();
    st.main.removeChildren();
    st.over.removeChildren();
    const byId = this.byId;
    const top = topFeatures(this.geom);
    const shapeFill = new paper.Color(cssVar('--shape-fill'));
    const ink = new paper.Color(cssVar('--shape-stroke'));
    const accent = new paper.Color(cssVar('--accent'));
    const err = new paper.Color(cssVar('--err'));
    const ready = new paper.Color(cssVar('--ready-line') || '#00A9A5');
    const canvasBg = new paper.Color(cssVar('--canvas'));
    const z = st.view.zoom;
    if (this.checkRaster) {
      st.main.addChild(this.checkRaster);
    } else {
      const dom = domainItem(this.geom, byId);
      if (dom) {
        dom.fillColor = shapeFill; dom.strokeColor = null;
        st.main.addChild(dom);
        // what each adding shape and each region is made of, clipped to the domain
        for (const f of top.filter((x) => (x.role === 'region' || (x.role === 'add' && this.colourBy === 'kappa')) && x.material)) {
          const it = itemOf(f, byId).intersect(dom, { insert: false });
          it.fillColor = new paper.Color(this.materialColour(f.material));
          it.fillColor.alpha = f.role === 'region' ? 0.62 : 0.4;
          st.main.addChild(it);
        }
      }
    }
    // every shape's own outline: what is hit, selected and dragged
    for (const f of this.geom.features) {
      if (f.hidden && !this.sel.has(f.id)) continue;
      const isTop = top.includes(f);
      if (!isTop && !this.sel.has(f.id)) continue;
      const it = itemOf(f, byId);
      it.data = { fid: f.id };
      it.fillColor = new paper.Color(0, 0, 0, 0.001);
      it.strokeScaling = false;
      const selected = this.sel.has(f.id);
      if (f.role === 'cut') { it.strokeColor = err; it.dashArray = [5, 3]; it.strokeWidth = selected ? 2.2 : 1.3; }
      else if (f.role === 'region') { it.strokeColor = ready; it.strokeWidth = selected ? 2.2 : 1.1; }
      else if (f.role === 'guide') { it.strokeColor = new paper.Color(cssVar('--muted')); it.dashArray = [3, 4]; it.fillColor = null; it.strokeWidth = 1; }
      else { it.strokeColor = ink; it.strokeWidth = selected ? 2.2 : 1.4; }
      if (selected) it.strokeColor = f.role === 'cut' ? err : accent;
      if (!isTop) { it.dashArray = [2, 3]; it.strokeColor = accent; }
      st.main.addChild(it);
    }
    this.drawHandles();
    this.drawOverlay();
    st.update();
    this.host.changed && this.host.changed();
  }

  handlePoints(f) {
    if (f.kind === 'rect') {
      const { x, y, w, h } = f;
      return [['c0', x, y], ['c1', x + w, y], ['c2', x + w, y + h], ['c3', x, y + h],
        ['e0', x + w / 2, y], ['e1', x + w, y + h / 2], ['e2', x + w / 2, y + h], ['e3', x, y + h / 2]];
    }
    if (f.kind === 'ellipse') {
      return [['rx+', f.cx + f.rx, f.cy], ['rx-', f.cx - f.rx, f.cy], ['ry+', f.cx, f.cy + f.ry], ['ry-', f.cx, f.cy - f.ry]];
    }
    if (f.kind === 'path' || f.kind === 'channel') {
      const out = f.points.map((p, k) => [`p${k}`, p[0], p[1]]);
      const n = f.points.length;
      const m = f.closed ? n : n - 1;
      for (let k = 0; k < m; k++) {
        if (f.edges[k] === 'cubic' && f.handles[k]) {
          out.push([`h${k}a`, f.handles[k][0], f.handles[k][1]], [`h${k}b`, f.handles[k][2], f.handles[k][3]]);
        } else if (f.edges[k] === 'arc') {
          const mid = arcMidpoint(f.points[k], f.points[(k + 1) % n], f.bulge[k]);
          out.push([`b${k}`, mid[0], mid[1]]);
        }
      }
      return out;
    }
    return [];
  }

  drawHandles() {
    const st = this.stage;
    const z = st.view.zoom;
    const accent = new paper.Color(cssVar('--accent'));
    const white = new paper.Color(cssVar('--surface'));
    if (this.tool === 'dimension') {
      // every point the tool can pick: corners, edge middles, centres, path points
      for (const f of topFeatures(this.geom)) {
        const pts = f.kind === 'ellipse' ? [['c', f.cx, f.cy]] : this.handlePoints(f).filter(([k]) => /^(c\d|e\d|p\d+)$/.test(k));
        for (const [key, x, y] of pts) {
          const s = 3.5 / z;
          const it = key.startsWith('e') ? new paper.Path.Rectangle({ point: [x - s, -y - s], size: [2 * s, 2 * s], insert: false })
            : new paper.Path.Circle({ center: P(x, y), radius: s, insert: false });
          it.fillColor = white; it.strokeColor = accent; it.strokeWidth = 1.5; it.strokeScaling = false;
          it.data = { handle: key, fid: f.id };
          st.over.addChild(it);
        }
      }
      return;
    }
    if (this.sel.size !== 1) return;
    const f = this.feature([...this.sel][0]);
    if (!f) return;
    if (f.kind === 'path') {
      const n = f.points.length;
      for (let k = 0; k < (f.closed ? n : n - 1); k++) {
        if (f.edges[k] === 'cubic' && f.handles[k]) {
          const h = f.handles[k];
          for (const [a, b] of [[f.points[k], [h[0], h[1]]], [f.points[(k + 1) % n], [h[2], h[3]]]]) {
            const l = new paper.Path.Line({ from: P(...a), to: P(...b), insert: false });
            l.strokeColor = accent; l.strokeWidth = 1; l.strokeScaling = false;
            st.over.addChild(l);
          }
        }
      }
    }
    if (f.kind === 'path' || f.kind === 'channel') {
      // edge numbers, for the inspector's constraints
      const n = f.points.length;
      for (let k = 0; k < (f.closed && f.kind === 'path' ? n : n - 1); k++) {
        const a = f.points[k], b = f.points[(k + 1) % n];
        const t = new paper.PointText({ point: P((a[0] + b[0]) / 2, (a[1] + b[1]) / 2).add(new paper.Point(0, -6 / z)), content: `e${k}`, justification: 'center',
          fontFamily: 'JetBrains Mono, monospace', fontSize: 9.5 / z, fillColor: new paper.Color(cssVar('--muted')), insert: false });
        st.over.addChild(t);
      }
    }
    for (const [key, x, y] of this.handlePoints(f)) {
      const round = key.startsWith('h') || key.startsWith('b') || key.startsWith('r');
      const s = HANDLE_PX / z;
      const it = round ? new paper.Path.Circle({ center: P(x, y), radius: s, insert: false })
        : new paper.Path.Rectangle({ point: [x - s, -y - s], size: [2 * s, 2 * s], insert: false });
      it.fillColor = white; it.strokeColor = accent; it.strokeWidth = 2; it.strokeScaling = false;
      it.data = { handle: key, fid: f.id };
      st.over.addChild(it);
    }
  }

  label(x, y, text) {
    const st = this.stage;
    const z = st.view.zoom;
    const t = new paper.PointText({ point: P(x, y).add(new paper.Point(10 / z, -10 / z)), content: text,
      fontFamily: 'JetBrains Mono, monospace', fontSize: 11 / z, fillColor: '#F2F1F7', insert: false });
    const bg = new paper.Path.Rectangle({ rectangle: t.bounds.expand(8 / z, 6 / z), radius: 5 / z, insert: false });
    bg.fillColor = '#191927';
    st.over.addChild(bg); st.over.addChild(t);
  }

  drawOverlay() {
    const st = this.stage;
    const z = st.view.zoom;
    const cyan = new paper.Color('#00D7D2');
    if (this.opts.dims && !this.checkRaster) drawDims(this);
    if (this.dimA) {
      const c = new paper.Path.Circle({ center: P(this.dimA.x, this.dimA.y), radius: 6 / z, insert: false });
      c.strokeColor = cyan; c.strokeWidth = 2; c.strokeScaling = false;
      st.over.addChild(c);
    }
    const b = st.view.bounds;
    for (const g of this.snapped) {
      const l = g.axis === 'x' ? new paper.Path.Line({ from: [g.v, b.top], to: [g.v, b.bottom], insert: false })
        : new paper.Path.Line({ from: [b.left, -g.v], to: [b.right, -g.v], insert: false });
      l.strokeColor = cyan; l.dashArray = [4, 3]; l.strokeWidth = 1; l.strokeScaling = false;
      st.over.addChild(l);
    }
    if (this.snapPoint && this.snapped.length) {
      const s = 4 / z;
      const d = new paper.Path({ segments: [P(this.snapPoint.x, this.snapPoint.y + s), P(this.snapPoint.x + s, this.snapPoint.y),
        P(this.snapPoint.x, this.snapPoint.y - s), P(this.snapPoint.x - s, this.snapPoint.y)], closed: true, insert: false });
      d.fillColor = cyan;
      st.over.addChild(d);
    }
    if (this.draft) this.drawDraft();
    if (this.pen) this.drawPen();
    if (this.marquee) {
      const m = this.marquee;
      const r = new paper.Path.Rectangle({ from: P(m.x0, m.y0), to: P(m.x1, m.y1), insert: false });
      r.strokeColor = new paper.Color(cssVar('--accent')); r.dashArray = [4, 3]; r.strokeScaling = false;
      r.fillColor = new paper.Color(cssVar('--accent')); r.fillColor.alpha = 0.06;
      st.over.addChild(r);
    }
    if (this.measure && this.measure.a) {
      const { a, b } = this.measure;
      const e = b || this.cursor || a;
      const l = new paper.Path.Line({ from: P(a.x, a.y), to: P(e.x, e.y), insert: false });
      l.strokeColor = cyan; l.strokeWidth = 1.5; l.strokeScaling = false;
      st.over.addChild(l);
      const d = Math.hypot(e.x - a.x, e.y - a.y);
      const ang = Math.atan2(e.y - a.y, e.x - a.x) * 180 / Math.PI;
      this.label(e.x, e.y, `${this.host.fmt(d)} ${this.host.unit()}  ${ang.toFixed(1)}°`);
    }
    for (const pin of this.pins) {
      const c = new paper.Path.Circle({ center: P(pin.at[0], pin.at[1]), radius: 9 / z, insert: false });
      c.fillColor = new paper.Color(pin.severity === 'error' ? '#E5484D' : '#F5A524');
      c.strokeColor = '#191927'; c.strokeWidth = 1; c.strokeScaling = false;
      const t = new paper.PointText({ point: P(pin.at[0], pin.at[1]).add(new paper.Point(0, 4 / z)), content: String(pin.pin),
        justification: 'center', fontSize: 11 / z, fontWeight: 'bold', fillColor: '#191927', insert: false });
      st.over.addChild(c); st.over.addChild(t);
    }
  }

  draftFeature() {
    const d = this.draft;
    if (!d) return null;
    if (d.kind === 'rect' || d.kind === 'ellipse') {
      let { x0, y0, x1, y1 } = d;
      if (d.square) {
        const s = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0));
        x1 = x0 + Math.sign(x1 - x0 || 1) * s; y1 = y0 + Math.sign(y1 - y0 || 1) * s;
      }
      if (d.centre) { x0 = 2 * x0 - x1; y0 = 2 * y0 - y1; }
      const x = Math.min(x0, x1), y = Math.min(y0, y1), w = Math.abs(x1 - x0), h = Math.abs(y1 - y0);
      if (d.kind === 'rect') return blank('rect', 'draft', this.effectiveRole(), { x, y, w, h });
      return blank('ellipse', 'draft', this.effectiveRole(), { cx: x + w / 2, cy: y + h / 2, rx: w / 2, ry: h / 2 });
    }
    if (d.kind === 'arc') {
      const b = d.bulge ?? 0.0001;
      return blank('path', 'draft', this.effectiveRole(), { points: [[d.x0, d.y0], [d.x1, d.y1]], edges: ['arc', 'line'], bulge: [b, 0], handles: [null, null] });
    }
    return null;
  }

  effectiveRole() { return this.altDown && this.role !== 'region' ? (this.role === 'add' ? 'cut' : 'add') : this.role; }

  drawDraft() {
    const f = this.draftFeature();
    if (!f) return;
    const st = this.stage;
    const it = itemOf(f, {});
    it.strokeColor = new paper.Color(f.role === 'cut' ? cssVar('--err') : (f.role === 'region' ? '#00A9A5' : cssVar('--accent')));
    it.strokeWidth = 1.6; it.strokeScaling = false; it.dashArray = f.role === 'cut' ? [5, 3] : null;
    it.fillColor = new paper.Color(f.role === 'cut' ? cssVar('--err') : cssVar('--accent')); it.fillColor.alpha = 0.12;
    st.over.addChild(it);
    if (f.kind === 'rect') this.label(f.x + f.w, f.y + f.h, `${this.host.fmt(f.w)} × ${this.host.fmt(f.h)}`);
    if (f.kind === 'ellipse') this.label(f.cx + f.rx, f.cy + f.ry, Math.abs(f.rx - f.ry) < 1e-12 ? `Ø ${this.host.fmt(2 * f.rx)}` : `${this.host.fmt(2 * f.rx)} × ${this.host.fmt(2 * f.ry)}`);
  }

  penFeature(close = false) {
    const pts = this.pen.points;
    const n = pts.length;
    const m = close ? n : n - 1;
    const edges = [], handles = [], bulge = [];
    for (let k = 0; k < m; k++) {
      const a = pts[k], b = pts[(k + 1) % n];
      const out = a.out, inn = b.in;
      if (out || inn) {
        const h1 = out || [a.x, a.y], h2 = inn || [b.x, b.y];
        edges.push('cubic'); handles.push([h1[0], h1[1], h2[0], h2[1]]);
      } else { edges.push('line'); handles.push(null); }
      bulge.push(0);
    }
    return blank('path', 'draft', close ? this.effectiveRole() : 'guide', {
      points: pts.map((p) => [p.x, p.y]), edges, handles, bulge, closed: close });
  }

  drawPen() {
    const st = this.stage;
    const z = st.view.zoom;
    const accent = new paper.Color(cssVar('--accent'));
    const pts = this.pen.points;
    if (pts.length >= 2) {
      const it = itemOf(this.penFeature(false), {});
      it.strokeColor = accent; it.strokeWidth = 1.8; it.strokeScaling = false; it.fillColor = null;
      st.over.addChild(it);
    }
    const last = pts[pts.length - 1];
    if (last && this.cursor && !this.drag) {
      const c = this.cursor;
      const l = new paper.Path.Line({ from: P(last.x, last.y), to: P(c.x, c.y), insert: false });
      l.strokeColor = accent; l.dashArray = [4, 3]; l.strokeWidth = 1.2; l.strokeScaling = false;
      st.over.addChild(l);
      const L = Math.hypot(c.x - last.x, c.y - last.y);
      const ang = Math.atan2(c.y - last.y, c.x - last.x) * 180 / Math.PI;
      this.label(c.x, c.y, (this.typed ? `${this.typed}▌` : this.host.fmt(L)) + `  ${ang.toFixed(0)}°`);
    }
    pts.forEach((p, k) => {
      const s = (k === 0 ? 6 : HANDLE_PX) / z;
      const it = new paper.Path.Rectangle({ point: [p.x - s, -p.y - s], size: [2 * s, 2 * s], insert: false });
      it.fillColor = k === 0 ? accent : new paper.Color(cssVar('--surface'));
      it.strokeColor = accent; it.strokeWidth = 2; it.strokeScaling = false;
      st.over.addChild(it);
      for (const h of [p.in, p.out]) {
        if (!h) continue;
        const l = new paper.Path.Line({ from: P(p.x, p.y), to: P(h[0], h[1]), insert: false });
        l.strokeColor = accent; l.strokeWidth = 1; l.strokeScaling = false;
        const c = new paper.Path.Circle({ center: P(h[0], h[1]), radius: 3.5 / z, insert: false });
        c.fillColor = accent;
        st.over.addChild(l); st.over.addChild(c);
      }
    });
  }

  // -- pointer input ------------------------------------------------------------------
  world(e) {
    const r = this.stage.canvas.getBoundingClientRect();
    const pt = this.stage.view.viewToProject(new paper.Point(e.clientX - r.left, e.clientY - r.top));
    return { x: pt.x, y: -pt.y, view: new paper.Point(e.clientX - r.left, e.clientY - r.top) };
  }

  snapped_(w, skip = new Set(), noGuides = false) {
    const tol = 7 / this.stage.view.zoom;
    const guides = this.opts.guides && !noGuides ? candidates(this.geom, skip, this.grid) : null;
    const s = snap(w.x, w.y, { grid: this.opts.grid ? this.stage.gridStep : null, guides, tol });
    this.snapped = s.lines;
    this.snapPoint = s.lines.length ? { x: s.x, y: s.y } : null;
    return s;
  }

  hit(view) {
    const st = this.stage;
    const tol = 6 / st.view.zoom;
    const pt = st.view.viewToProject(view);
    const h = st.project.hitTest(pt, { fill: true, stroke: true, segments: false, tolerance: tol,
      match: (r) => r.item && r.item.data && (r.item.data.handle || r.item.data.fid) });
    if (!h) return null;
    return h.item.data;
  }

  install() {
    const c = this.stage.canvas;
    c.addEventListener('pointerdown', (e) => this.down(e));
    c.addEventListener('pointermove', (e) => this.move(e));
    c.addEventListener('pointerup', (e) => this.up(e));
    c.addEventListener('pointercancel', () => this.cancel());
    c.addEventListener('dblclick', (e) => this.dbl(e));
    c.addEventListener('pointerleave', () => { this.cursor = null; if (!this.drag) { this.snapped = []; this.redraw(); } });
    window.addEventListener('keydown', (e) => { this.altDown = e.altKey; this.key(e); });
    window.addEventListener('keyup', (e) => { this.altDown = e.altKey; });
  }

  down(e) {
    if (e.button !== 0 || this.stage.space || this.tool === 'hand') return;
    this.stage.canvas.setPointerCapture(e.pointerId);
    this.altDown = e.altKey;
    const w = this.world(e);
    if (this.fillMaterial) { this.fill(w); return; }
    if (this.tool === 'select') return this.downSelect(e, w);
    if (this.tool === 'dimension') return this.downDim(e, w);
    if (this.tool === 'rect' || this.tool === 'ellipse' || this.tool === 'area') {
      const s = this.snapped_(w);
      this.draft = { kind: this.tool === 'area' ? 'rect' : this.tool, area: this.tool === 'area', x0: s.x, y0: s.y, x1: s.x, y1: s.y, square: e.shiftKey, centre: e.altKey };
      this.drag = { kind: 'draft' };
      this.redraw();
      return;
    }
    if (this.tool === 'arc') {
      const s = this.snapped_(w);
      if (this.draft && this.draft.kind === 'arc' && this.draft.stage === 'bulge') {
        this.finishArc();
        return;
      }
      this.draft = { kind: 'arc', x0: s.x, y0: s.y, x1: s.x, y1: s.y, stage: 'end' };
      this.drag = { kind: 'draft' };
      this.redraw();
      return;
    }
    if (this.tool === 'pen' || this.tool === 'channel') return this.downPen(e, w);
    if (this.tool === 'measure') {
      const s = this.snapped_(w);
      if (!this.measure || this.measure.b) this.measure = { a: { x: s.x, y: s.y }, b: null };
      else this.measure.b = { x: s.x, y: s.y };
      this.redraw();
    }
  }

  /** the Dimension tool: an edge or a hole sets a size; two points set a position */
  downDim(e, w) {
    const h = this.hit(w.view);
    if (!h) { this.dimA = null; this.redraw(); return; }
    const f = this.feature(h.fid);
    if (!f) return;
    if (h.handle) {
      if (f.kind === 'rect' && /^e[0-3]$/.test(h.handle) && !this.dimA) { this.host.dimension({ fid: f.id, key: h.handle }); return; }
      const p = pointOf(f, h.handle);
      if (!p) return;
      if (!this.dimA) { this.dimA = { fid: f.id, key: h.handle, x: p[0], y: p[1] }; this.redraw(); if (this.host.changed) this.host.changed(); return; }
      const A = this.dimA;
      this.dimA = null;
      this.redraw();
      this.host.dimPoints(A, { fid: f.id, key: h.handle }, e.shiftKey);
      return;
    }
    this.dimA = null;
    if (f.kind === 'ellipse') { this.host.dimension({ fid: f.id }); return; }
    if (f.kind === 'path') {
      const ne = nearestEdge(f, w);
      if (ne) this.host.dimension({ fid: f.id, edge: ne.edge });
      return;
    }
    if (f.kind === 'rect') {
      const d = [Math.abs(w.y - f.y), Math.abs(w.x - (f.x + f.w)), Math.abs(w.y - (f.y + f.h)), Math.abs(w.x - f.x)];
      this.host.dimension({ fid: f.id, key: `e${d.indexOf(Math.min(...d))}` });
      return;
    }
    this.host.notify('info', 'Dimension an edge of a rectangle, a hole, a straight edge of a path, or two points.');
  }

  downSelect(e, w) {
    const h = this.hit(w.view);
    if (h && h.handle) {
      const f = this.feature(h.fid);
      this.drag = { kind: 'handle', fid: f.id, handle: h.handle, orig: JSON.parse(JSON.stringify(f)), start: w, moved: false };
      return;
    }
    if (h && h.fid) {
      const fid = this.ownerOf(h.fid);
      if (e.shiftKey) { if (this.sel.has(fid)) this.sel.delete(fid); else this.sel.add(fid); }
      else if (!this.sel.has(fid)) { this.sel.clear(); this.sel.add(fid); }
      const ids = [...this.sel].flatMap((id) => this.withInputs(id));
      this.drag = { kind: 'move', ids, orig: ids.map((id) => JSON.parse(JSON.stringify(this.feature(id)))), start: w, moved: false };
      this.redraw();
      this.host.selected && this.host.selected();
      return;
    }
    if (!e.shiftKey) this.sel.clear();
    this.marquee = { x0: w.x, y0: w.y, x1: w.x, y1: w.y };
    this.drag = { kind: 'marquee' };
    this.redraw();
    this.host.selected && this.host.selected();
  }

  /** the top-level feature a hit belongs to (a combine's input selects the combine) */
  ownerOf(fid) {
    for (const f of this.geom.features) if (['combine', 'pattern', 'mirror'].includes(f.kind) && f.inputs.includes(fid)) return this.ownerOf(f.id);
    return fid;
  }

  withInputs(fid) {
    const f = this.feature(fid);
    if (!f || !['combine', 'pattern', 'mirror'].includes(f.kind)) return [fid];
    return [fid, ...f.inputs.flatMap((i) => this.withInputs(i))];
  }

  downPen(e, w) {
    const s = this.snapped_(w);
    let { x, y } = s;
    if (!this.pen) this.pen = { points: [] };
    const pts = this.pen.points;
    const last = pts[pts.length - 1];
    if (last && e.shiftKey) ({ x, y } = lockAngle(last.x, last.y, x, y, 15));
    // a click on the first point closes the shape
    if (pts.length >= 2 && this.tool !== 'channel') {
      const f0 = pts[0];
      const d = Math.hypot(f0.x - x, f0.y - y) * this.stage.view.zoom;
      if (d <= 8) {
        this.drag = { kind: 'pen-close' };
        return;
      }
    }
    pts.push({ x, y, in: null, out: null });
    this.drag = { kind: 'pen', start: { x, y }, index: pts.length - 1, alt: e.altKey };
    this.typed = '';
    this.redraw();
    this.host.changed && this.host.changed();
  }

  move(e) {
    const w = this.world(e);
    this.cursor = { x: w.x, y: w.y };
    this.altDown = e.altKey;
    const d = this.drag;
    if (!d) {
      if ((this.tool === 'pen' || this.tool === 'channel') && this.pen) {
        const s = this.snapped_(w);
        let { x, y } = s;
        const last = this.pen.points[this.pen.points.length - 1];
        if (last && e.shiftKey) ({ x, y } = lockAngle(last.x, last.y, x, y, 15));
        this.cursor = { x, y };
        this.redraw();
      } else if (this.draft && this.draft.kind === 'arc' && this.draft.stage === 'bulge') {
        this.draft.bulge = bulgeFromPoint([this.draft.x0, this.draft.y0], [this.draft.x1, this.draft.y1], [w.x, w.y]);
        this.redraw();
      } else if (this.tool === 'measure' && this.measure && !this.measure.b) {
        const s = this.snapped_(w); this.cursor = { x: s.x, y: s.y }; this.redraw();
      } else if (['rect', 'ellipse', 'arc', 'pen', 'channel', 'area', 'dimension'].includes(this.tool) && this.opts.guides) {
        this.snapped_(w); this.redraw();
      }
      return;
    }
    if (d.kind === 'draft') {
      const s = this.snapped_(w);
      this.draft.x1 = s.x; this.draft.y1 = s.y;
      this.draft.square = e.shiftKey; this.draft.centre = e.altKey && this.draft.kind !== 'arc';
      this.redraw();
      this.host.changed && this.host.changed();
      return;
    }
    if (d.kind === 'marquee') { this.marquee.x1 = w.x; this.marquee.y1 = w.y; this.redraw(); return; }
    if (d.kind === 'move') {
      let dx = w.x - d.start.x, dy = w.y - d.start.y;
      if (Math.hypot(dx, dy) * this.stage.view.zoom < 3 && !d.moved) return;
      d.moved = true;
      // snap the moving shape's own box to the others
      const f0 = d.orig[0];
      const bb = bboxOf(translated(f0, dx, dy), byIdOf({ features: [...d.orig.map((o) => translated(o, dx, dy))] }));
      const skip = new Set(d.ids);
      const corners = [[bb[0], bb[1]], [(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2], [bb[2], bb[3]]];
      let best = null;
      for (const [cx, cy] of corners) {
        const s = this.snapped_({ x: cx, y: cy }, skip);
        if (s.lines.length && (!best || s.lines.length > best.n)) best = { n: s.lines.length, ddx: s.x - cx, ddy: s.y - cy, lines: s.lines };
      }
      if (best) { dx += best.ddx; dy += best.ddy; this.snapped = best.lines; } else this.snapped = [];
      for (const o of d.orig) {
        const i = this.index(o.id);
        this.geom.features[i] = this.hold(translated(o, dx, dy), o);
      }
      this.redraw();
      return;
    }
    if (d.kind === 'handle') {
      d.moved = true;
      const s = this.snapped_(w, new Set([d.fid]));
      const f = this.feature(d.fid);
      if (f && this.locked(f).has('points') && f.kind !== 'rect' && f.kind !== 'ellipse') return;
      if (f && (f.kind === 'path' || f.kind === 'channel') && /^p\d+$/.test(d.handle) && this.ownConstraints(f).length) {
        // the path's constraints hold while its point moves: planegcs solves each move
        const k = parseInt(d.handle.slice(1), 10);
        d.want = [s.x, s.y];
        if (!d.solving) {
          const go = () => {
            const [x, y] = d.want;
            d.want = null;
            d.solving = this.host.solveDrag(this.feature(d.fid), k, x, y).then((pts) => {
              if (pts && this.feature(d.fid)) { this.feature(d.fid).points = pts; this.redraw(); }
              d.solving = null;
              if (d.want) go();
            });
          };
          go();
        }
        return;
      }
      this.applyHandle(d, s.x, s.y, e.shiftKey);
      if (f) this.hold(f, d.orig);
      this.redraw();
      return;
    }
    if (d.kind === 'pen') {
      // dragging out from a new point makes it smooth: handles mirror (Alt breaks)
      const p = this.pen.points[d.index];
      const hx = w.x, hy = w.y;
      if (Math.hypot(hx - p.x, hy - p.y) * this.stage.view.zoom < 4) return;
      p.out = [hx, hy];
      if (!d.alt) p.in = [2 * p.x - hx, 2 * p.y - hy];
      this.redraw();
    }
  }

  applyHandle(d, x, y, shift) {
    const f = this.feature(d.fid);
    const o = d.orig;
    const k = d.handle;
    if (f.kind === 'rect') {
      let x0 = o.x, y0 = o.y, x1 = o.x + o.w, y1 = o.y + o.h;
      if (k === 'c0') { x0 = x; y0 = y; } else if (k === 'c1') { x1 = x; y0 = y; }
      else if (k === 'c2') { x1 = x; y1 = y; } else if (k === 'c3') { x0 = x; y1 = y; }
      else if (k === 'e0') y0 = y; else if (k === 'e1') x1 = x; else if (k === 'e2') y1 = y; else if (k === 'e3') x0 = x;
      f.x = Math.min(x0, x1); f.y = Math.min(y0, y1); f.w = Math.max(Math.abs(x1 - x0), 1e-9); f.h = Math.max(Math.abs(y1 - y0), 1e-9);
      f.r = Math.min(o.r || 0, f.w / 2, f.h / 2);
    } else if (f.kind === 'ellipse') {
      if (k.startsWith('rx')) f.rx = Math.max(Math.abs(x - o.cx), 1e-9);
      else f.ry = Math.max(Math.abs(y - o.cy), 1e-9);
      if (shift) { const r = k.startsWith('rx') ? f.rx : f.ry; f.rx = r; f.ry = r; }
    } else if (f.kind === 'path' || f.kind === 'channel') {
      const n = f.points.length;
      if (k.startsWith('p')) {
        const i = parseInt(k.slice(1), 10);
        const ddx = x - o.points[i][0], ddy = y - o.points[i][1];
        f.points[i] = [x, y];
        // a moved point carries its own handles
        const m = f.closed ? n : n - 1;
        for (let e = 0; e < m; e++) {
          const h = o.handles[e];
          if (!h || f.edges[e] !== 'cubic') continue;
          const nh = h.slice();
          if (e === i) { nh[0] = h[0] + ddx; nh[1] = h[1] + ddy; }
          if ((e + 1) % n === i) { nh[2] = h[2] + ddx; nh[3] = h[3] + ddy; }
          f.handles[e] = nh;
        }
      } else if (k.startsWith('h')) {
        const e = parseInt(k.slice(1), 10);
        const h = f.handles[e].slice();
        if (k.endsWith('a')) { h[0] = x; h[1] = y; } else { h[2] = x; h[3] = y; }
        f.handles[e] = h;
      } else if (k.startsWith('b')) {
        const e = parseInt(k.slice(1), 10);
        f.bulge[e] = bulgeFromPoint(f.points[e], f.points[(e + 1) % n], [x, y]);
      }
    }
  }

  async up(e) {
    const d = this.drag;
    if (!d) return;
    while (d.solving) await d.solving;
    this.drag = null;
    const w = this.world(e);
    if (d.kind === 'draft' && this.draft && this.draft.area) {
      const f = this.draftFeature();
      this.draft = null; this.snapped = [];
      if (f && f.w > 0 && f.h > 0) this.host.areaDrawn([f.x, f.y, f.w, f.h]);
      this.host.setTool('select');
      return;
    }
    if (d.kind === 'draft') {
      if (this.draft.kind === 'arc') {
        const L = Math.hypot(this.draft.x1 - this.draft.x0, this.draft.y1 - this.draft.y0);
        if (L * this.stage.view.zoom < 6) { this.draft = null; this.redraw(); return; }
        this.draft.stage = 'bulge';
        this.draft.bulge = 0.0001;
        this.redraw();
        return;
      }
      const f = this.draftFeature();
      this.draft = null;
      this.snapped = [];
      const big = f && (f.kind === 'rect' ? Math.min(f.w, f.h) : 2 * Math.min(f.rx, f.ry)) * this.stage.view.zoom >= 4;
      if (big) this.addFeature(f);
      else this.redraw();
      return;
    }
    if (d.kind === 'marquee') {
      const m = this.marquee;
      this.marquee = null;
      const x0 = Math.min(m.x0, m.x1), x1 = Math.max(m.x0, m.x1), y0 = Math.min(m.y0, m.y1), y1 = Math.max(m.y0, m.y1);
      if ((x1 - x0) * this.stage.view.zoom > 3) {
        const byId = this.byId;
        for (const f of topFeatures(this.geom)) {
          const b = bboxOf(f, byId);
          if (b[0] >= x0 && b[2] <= x1 && b[1] >= y0 && b[3] <= y1) this.sel.add(f.id);
        }
      }
      this.redraw();
      this.host.selected && this.host.selected();
      this.flushPending();
      return;
    }
    if (d.kind === 'move') {
      this.snapped = [];
      if (!d.moved) { this.redraw(); this.flushPending(); return; }
      const ops = d.orig.map((o) => ({ op: 'set', path: ['geometry', 'features', this.index(o.id)], value: this.feature(o.id) }));
      this.host.commit(ops, `moved ${d.orig.length > 1 ? `${d.orig.length} shapes` : (d.orig[0].name || d.orig[0].id)}`);
      return;
    }
    if (d.kind === 'handle') {
      this.snapped = [];
      if (!d.moved) { this.flushPending(); return; }
      const f = this.feature(d.fid);
      this.host.commit([{ op: 'set', path: ['geometry', 'features', this.index(f.id)], value: f }], `reshaped ${f.name || f.id}`);
      return;
    }
    if (d.kind === 'pen-close') { this.finishPen(true); return; }
    if (d.kind === 'pen') { this.redraw(); }
  }

  dbl(e) {
    if (this.tool === 'channel' && this.pen && this.pen.points.length >= 2) {
      const pts = this.pen.points;
      const a = pts[pts.length - 1], b = pts[pts.length - 2];
      if (Math.hypot(a.x - b.x, a.y - b.y) * this.stage.view.zoom < 4) pts.pop();
      this.finishPen(false);
      return;
    }
    if (this.tool === 'pen' && this.pen && this.pen.points.length >= 3) {
      // a double-click places a point twice; drop the duplicate and close
      const pts = this.pen.points;
      if (pts.length >= 2) {
        const a = pts[pts.length - 1], b = pts[pts.length - 2];
        if (Math.hypot(a.x - b.x, a.y - b.y) * this.stage.view.zoom < 4) pts.pop();
      }
      this.finishPen(true);
    }
  }

  cancel() {
    this.drag = null; this.draft = null; this.marquee = null; this.snapped = [];
    this.redraw();
    this.flushPending();
  }

  finishArc() {
    const f = this.draftFeature();
    this.draft = null;
    this.snapped = [];
    if (f && Math.abs(f.bulge[0]) > 1e-3) this.addFeature(f);
    else this.redraw();
  }

  finishPen(close) {
    const pen = this.pen;
    this.drag = null;
    if (!pen) return;
    const n = pen.points.length;
    if ((close && n < 2) || (!close && n < 2)) { this.pen = null; this.redraw(); this.flushPending(); return; }
    let f = this.penFeature(close);
    this.pen = null;
    this.typed = '';
    if (this.tool === 'channel') {
      f = blank('channel', 'draft', this.effectiveRole(), {
        points: f.points, edges: f.edges, handles: f.handles, bulge: f.bulge, closed: false, w: this.host.channelWidth() });
      this.addFeature(f);
      return;
    }
    if (close && n === 2 && f.edges.every((k) => k === 'line')) { this.host.notify('info', 'Two points enclose nothing: add a third, or drag to curve.'); this.redraw(); return; }
    this.addFeature(f);
  }

  addFeature(f) {
    const id = nextId(this.geom);
    f.id = id;
    const n = this.geom.features.filter((x) => x.kind === f.kind).length + 1;
    f.name = f.name || ({ rect: 'Rectangle', ellipse: Math.abs(f.rx - f.ry) < 1e-12 ? 'Circle' : 'Ellipse', path: f.closed ? 'Shape' : 'Line', channel: 'Channel' }[f.kind] + ` ${n}`);
    if (f.role === 'add' || f.role === 'region') f.material = f.material || this.host.material(f.role);
    if (f.role === 'cut' || f.role === 'guide') f.material = null;
    this.geom.features.push(f);
    this.sel.clear(); this.sel.add(id);
    this.redraw();
    this.host.selected && this.host.selected();
    this.host.commit([{ op: 'insert', path: ['geometry', 'features'], value: f }], `drew ${f.name}`);
  }

  fill(w) {
    const h = this.hit(w.view);
    if (!h || !h.fid) return;
    const fid = this.ownerOf(h.fid);
    const f = this.feature(fid);
    if (!f || f.role === 'cut' || f.role === 'guide') { this.host.notify('info', 'Click a shape that adds, or a region, to fill it.'); return; }
    const i = this.index(fid);
    this.host.commit([{ op: 'set', path: ['geometry', 'features', i, 'material'], value: this.fillMaterial }],
      `${f.name || fid}: ${this.fillMaterial}`);
  }

  // -- keys ---------------------------------------------------------------------------
  key(e) {
    if (!this.stage.isShown() || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
    if (document.querySelector('.scrim')) return;
    const k = e.key;
    if (this.pen) {
      if (/^[0-9.]$/.test(k)) { this.typed += k; this.redraw(); e.preventDefault(); return; }
      if (k === 'Backspace') {
        e.preventDefault();
        if (this.typed) this.typed = this.typed.slice(0, -1);
        else { this.pen.points.pop(); if (!this.pen.points.length) this.pen = null; }
        this.redraw(); return;
      }
      if (k === 'Enter') {
        e.preventDefault();
        if (this.typed && this.cursor) {
          // a typed length places the next point along the cursor's direction
          const L = parseFloat(this.typed);
          const last = this.pen.points[this.pen.points.length - 1];
          const a = Math.atan2(this.cursor.y - last.y, this.cursor.x - last.x);
          if (L > 0) this.pen.points.push({ x: last.x + L * Math.cos(a), y: last.y + L * Math.sin(a), in: null, out: null });
          this.typed = '';
          this.redraw();
        } else this.finishPen(false);
        return;
      }
      if (k === 'Escape') { e.preventDefault(); this.pen = null; this.typed = ''; this.redraw(); this.flushPending(); return; }
    }
    if (k === 'Escape') {
      if (this.dimA) { this.dimA = null; this.redraw(); return; }
      if (this.draft) { this.draft = null; this.redraw(); return; }
      if (this.fillMaterial) { this.fillMaterial = null; this.host.changed && this.host.changed(); return; }
      if (this.measure) { this.measure = null; this.redraw(); return; }
      if (this.sel.size) { this.sel.clear(); this.redraw(); this.host.selected && this.host.selected(); return; }
      if (this.tool !== 'select') { this.host.setTool('select'); return; }
    }
    if ((k === 'Delete' || k === 'Backspace') && this.sel.size && this.tool === 'select') { e.preventDefault(); this.host.remove([...this.sel]); return; }
    if ((e.ctrlKey || e.metaKey) && (k === 'd' || k === 'D') && this.sel.size) { e.preventDefault(); this.host.duplicate([...this.sel]); return; }
    if (k.startsWith('Arrow') && this.sel.size && this.tool === 'select') {
      e.preventDefault();
      const step = (e.shiftKey ? 10 : 1) * (this.opts.grid ? this.stage.gridStep : this.grid.dx);
      const dx = k === 'ArrowLeft' ? -step : k === 'ArrowRight' ? step : 0;
      const dy = k === 'ArrowDown' ? -step : k === 'ArrowUp' ? step : 0;
      const ids = [...this.sel].flatMap((id) => this.withInputs(id));
      const ops = ids.map((id) => {
        const f = this.hold(translated(this.feature(id), dx, dy), this.feature(id));
        this.geom.features[this.index(id)] = f;
        return { op: 'set', path: ['geometry', 'features', this.index(id)], value: f };
      });
      this.redraw();
      this.host.commit(ops, 'nudged');
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const tools = { v: 'select', p: 'pen', r: 'rect', e: 'ellipse', a: 'arc', m: 'measure', h: 'hand', c: 'channel', d: 'dimension' };
    if (tools[k.toLowerCase()]) { this.host.setTool(tools[k.toLowerCase()]); return; }
    if (k.toLowerCase() === 'i') { this.host.importFile(); return; }
    if (k.toLowerCase() === 'k') { this.host.toggleCheck(); return; }
    if (k === 'Tab' && this.drag && this.drag.kind === 'draft') { e.preventDefault(); this.host.typeSize(); }
  }

  /** the draft's exact size, typed (Tab while dragging) */
  typedSize(w, h) {
    const d = this.draft;
    if (!d) return;
    const sx = Math.sign(d.x1 - d.x0) || 1, sy = Math.sign(d.y1 - d.y0) || 1;
    d.x1 = d.x0 + sx * w; d.y1 = d.y0 + sy * h;
    const f = this.draftFeature();
    this.draft = null; this.drag = null;
    this.addFeature(f);
  }

  setTool(t) {
    // a pen or channel line in progress ends as what its own tool makes
    if (this.pen && t !== this.tool) this.finishPen(false);
    this.tool = t;
    this.stage.tool = t;
    this.dimA = null;
    this.draft = null; this.measure = null; this.snapped = [];
    this.stage.canvas.style.cursor = t === 'select' ? 'default' : (t === 'hand' ? 'grab' : 'crosshair');
    this.redraw();
  }
}
