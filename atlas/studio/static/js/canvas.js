// A Paper.js stage: one canvas, its own PaperScope, in the physics' length unit.
//
// Paper's coordinates are (x, -y): the world is y-up, the screen y-down, and
// keeping the flip out of the view matrix keeps text upright.  Strokes do not
// scale with the zoom (strokeScaling = false on everything drawn), so a line is
// as thick at any zoom.  Ctrl+wheel zooms about the cursor, wheel pans, a middle
// drag or Space+drag pans, Shift+1 fits.

import { cssVar } from './ui.js';

/* global paper */

export const P = (x, y) => new paper.Point(x, -y);
export const W = (pt) => ({ x: pt.x, y: -pt.y });

export class Stage {
  constructor(canvas, { onDraw = null } = {}) {
    this.canvas = canvas;
    this.scope = new paper.PaperScope();
    this.scope.setup(canvas);
    this.scope.settings.insertItems = false;
    this.project = this.scope.project;
    this.view = this.scope.view;
    this.grid = new this.scope.Layer({ name: 'grid' });
    this.main = new this.scope.Layer({ name: 'main' });
    this.over = new this.scope.Layer({ name: 'over' });
    for (const l of [this.grid, this.main, this.over]) this.project.addLayer(l);
    this.main.activate();
    this.bounds = { x0: 0, y0: 0, x1: 1, y1: 1 };
    this.onDraw = onDraw;
    this.space = false;
    this.panning = null;
    this.fitted = false;
    this.moved = false;           // the person zoomed or panned since the last fit
    this._installNav();
    this.view.onResize = () => { if (!this.fitted) this.fit(); else this.drawGrid(); this.update(); };
    // the canvas fills its box; a hidden tab has no size until it is shown
    this.resizer = new ResizeObserver(() => {
      const box = canvas.parentElement;
      const w = box.clientWidth, ht = box.clientHeight;
      if (w > 0 && ht > 0 && (w !== this.view.viewSize.width || ht !== this.view.viewSize.height)) {
        const first = !this.view.viewSize.width || !this.view.viewSize.height;
        this.view.viewSize = new paper.Size(w, ht);
        // a view fitted while the canvas still had its default size fits again; one the
        // person has zoomed or panned keeps their view
        if (first || !this.fitted || !this.moved) this.fit();
      }
    });
    this.resizer.observe(canvas.parentElement);
  }

  activate() { this.scope.activate(); }

  /** pixels per world unit */
  get scale() { return this.view.zoom; }

  setBounds(x0, y0, x1, y1) {
    const same = this.bounds.x0 === x0 && this.bounds.y0 === y0 && this.bounds.x1 === x1 && this.bounds.y1 === y1;
    this.bounds = { x0, y0, x1, y1 };
    if (!same || !this.fitted) this.fit();
  }

  fit() {
    const { x0, y0, x1, y1 } = this.bounds;
    const vs = this.view.viewSize;
    if (!vs.width || !vs.height) return;
    const w = Math.max(x1 - x0, 1e-9), ht = Math.max(y1 - y0, 1e-9);
    this.view.zoom = Math.min(vs.width / w, vs.height / ht) * 0.88;
    this.view.center = P((x0 + x1) / 2, (y0 + y1) / 2);
    this.fitted = true;
    this.moved = false;
    this.drawGrid();
    this.update();
  }

  zoomAt(factor, viewPt) {
    this.moved = true;
    const v = this.view;
    const before = v.viewToProject(viewPt);
    v.zoom = Math.max(1e-6, Math.min(1e9, v.zoom * factor));
    const after = v.viewToProject(viewPt);
    v.center = v.center.add(before.subtract(after));
    this.drawGrid();
    this.update();
  }

  panBy(dxPx, dyPx) {
    this.moved = true;
    this.view.center = this.view.center.subtract(new paper.Point(dxPx, dyPx).divide(this.view.zoom));
    this.drawGrid();
    this.update();
  }

  /** a spacing of 1, 2 or 5 times a power of ten giving about `px` pixels */
  niceStep(px = 24) {
    const raw = px / this.view.zoom;
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p;
    return 10 * p;
  }

  drawGrid() {
    this.activate();
    this.grid.removeChildren();
    const b = this.view.bounds;
    const step = this.niceStep(24);
    const col = new paper.Color(cssVar('--canvas-grid') || '#EEEDF4');
    const x0 = Math.floor(b.left / step) * step, x1 = b.right;
    const yTop = -b.top, yBot = -b.bottom;
    let n = 0;
    for (let x = x0; x <= x1 && n < 400; x += step, n++) {
      const l = new paper.Path.Line(new paper.Point(x, b.top), new paper.Point(x, b.bottom));
      l.strokeColor = col; l.strokeWidth = 1; l.strokeScaling = false;
      this.grid.addChild(l);
    }
    const ys = Math.floor(yBot / step) * step;
    for (let y = ys; y <= yTop && n < 800; y += step, n++) {
      const l = new paper.Path.Line(P(b.left, y), P(b.right, y));
      l.strokeColor = col; l.strokeWidth = 1; l.strokeScaling = false;
      this.grid.addChild(l);
    }
    this.gridStep = step;
  }

  update() { this.view.requestUpdate(); }

  clear() { this.activate(); this.main.removeChildren(); this.over.removeChildren(); }

  /** a polygon path from rings of [x, y] in world units */
  rings(rings, style = {}) {
    this.activate();
    const paths = rings.filter((r) => r && r.length > 1).map((r) => new paper.Path({
      segments: r.map(([x, y]) => P(x, y)), closed: true, insert: false }));
    const item = paths.length === 1 ? paths[0] : new paper.CompoundPath({ children: paths, insert: false });
    if (paths.length > 1) item.fillRule = 'evenodd';
    return styled(item, style);
  }

  line(points, style = {}) {
    this.activate();
    return styled(new paper.Path({ segments: points.map(([x, y]) => P(x, y)), insert: false }), style);
  }

  viewToWorld(pt) { return W(this.view.viewToProject(pt)); }

  _installNav() {
    const c = this.canvas;
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      const vp = new paper.Point(e.clientX - r.left, e.clientY - r.top);
      if (e.ctrlKey || e.metaKey) this.zoomAt(Math.exp(-e.deltaY * 0.0015), vp);
      else this.panBy(-e.deltaX, -e.deltaY);
    }, { passive: false });
    c.addEventListener('pointerdown', (e) => {
      if (e.button === 1 || (e.button === 0 && (this.space || this.tool === 'hand'))) {
        e.preventDefault();
        this.panning = { x: e.clientX, y: e.clientY };
        c.setPointerCapture(e.pointerId);
        c.style.cursor = 'grabbing';
        e.stopImmediatePropagation();
      }
    }, true);
    c.addEventListener('pointermove', (e) => {
      if (!this.panning) return;
      this.panBy(e.clientX - this.panning.x, e.clientY - this.panning.y);
      this.panning = { x: e.clientX, y: e.clientY };
      e.stopImmediatePropagation();
    }, true);
    const end = (e) => {
      if (!this.panning) return;
      this.panning = null;
      c.style.cursor = '';
      e.stopImmediatePropagation();
    };
    c.addEventListener('pointerup', end, true);
    c.addEventListener('pointercancel', end, true);
    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space' && !/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) && this.isShown()) {
        this.space = true; c.style.cursor = 'grab';
      }
    });
    window.addEventListener('keyup', (e) => {
      if (e.code === 'Space') { this.space = false; if (!this.panning) c.style.cursor = ''; }
    });
  }

  isShown() { return this.canvas.offsetParent !== null; }
}

export function styled(item, s = {}) {
  if (s.fill) item.fillColor = s.fill;
  if (s.fillAlpha !== undefined && item.fillColor) item.fillColor.alpha = s.fillAlpha;
  if (s.stroke) item.strokeColor = s.stroke;
  item.strokeWidth = s.width ?? 1.4;
  item.strokeScaling = false;
  if (s.dash) item.dashArray = s.dash;
  if (s.alpha !== undefined) item.opacity = s.alpha;
  if (s.name) item.name = s.name;
  if (s.data) item.data = s.data;
  return item;
}

/** colours for materials, regions and pieces, in a fixed order */
export const SWATCHES = ['#8E72EE', '#00D7D2', '#F5A524', '#E5484D', '#5B3FC4', '#00A9A5',
  '#B9A6F6', '#7FEBE8', '#E0A85A', '#9B1C20'];
