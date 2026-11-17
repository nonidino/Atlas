// Image tracing (plan §3.6): a picture shown under the canvas at a set scale, to trace
// with the pen.  It belongs to this page only: it is never sent to the server and never
// saved in the case.  Its lower-left corner sits at the drawing's origin; its width is
// typed in the case's units, and its height follows the picture's proportions.

/* global paper */
import { h, fmt, dialog, toast } from '../ui.js';

export class Underlay {
  constructor(stage) {
    this.stage = stage;
    this.layer = new stage.scope.Layer({ name: 'under' });
    stage.project.insertLayer(1, this.layer);      // above the grid, under the shapes
    this.raster = null;
    this.name = '';
    this.width = 0;
    this.opacity = 0.4;
    this.shown = true;
  }

  get active() { return !!this.raster; }

  async pick(defaultWidth, unit) {
    const input = h('input', { type: 'file', accept: 'image/*' });
    const file = await new Promise((ok) => { input.addEventListener('change', () => ok(input.files && input.files[0])); input.click(); });
    if (!file) return false;
    const w = h('input', { class: 'field num', value: fmt(defaultWidth, 4), style: { width: '90px' } });
    const ok = await dialog('Trace an image', [h('div', { class: 'muted' }, 'Shown under the drawing, on this page only: it is never saved in the case.'),
      h('label', { class: 'row' }, 'its width', h('span', {}, w, ` ${unit}`))],
    [{ label: 'Cancel' }, { label: 'Place', primary: true, value: () => Number(w.value) > 0 }]);
    if (!ok) return false;
    const url = await new Promise((res, bad) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = () => bad(r.error); r.readAsDataURL(file); });
    this.place(url, Number(w.value), file.name);
    return true;
  }

  place(url, width, name = 'image') {
    this.stage.activate();
    this.clear();
    const r = new paper.Raster({ source: url, insert: false });
    r.onLoad = () => { this.fit(); this.stage.update(); };
    r.onError = () => toast('error', `${name}: not an image this browser can show.`);
    this.layer.addChild(r);
    this.raster = r; this.name = name; this.width = width;
    this.fit();
  }

  fit() {
    const r = this.raster;
    if (!r || !r.width) return;
    const H = this.width * r.height / r.width;
    r.bounds = new paper.Rectangle(0, -H, this.width, H);
    r.opacity = this.opacity;
    r.visible = this.shown;
  }

  set(opts) {
    Object.assign(this, opts);
    this.fit();
    this.stage.update();
  }

  clear() {
    this.layer.removeChildren();
    this.raster = null;
    this.stage.update();
  }
}

/** the controls, for the Drawing panel */
export function traceSection(under, unit, defaultWidth, rerender) {
  const sec = h('section', { class: 'subpanel' });
  sec.append(h('div', { style: { display: 'flex', alignItems: 'center' } }, h('span', { style: { fontWeight: 700, flex: 1 } }, 'Trace'),
    h('span', { class: 'q', 'data-tip': 'A picture under the drawing at a set scale, to trace with the pen. On this page only.' }, '?')));
  if (!under.active) {
    const b = h('button', { class: 'btn', 'data-tip': 'Choose a picture to trace' }, 'Image to trace…');
    b.addEventListener('click', async () => { if (await under.pick(defaultWidth, unit)) rerender(); });
    sec.append(b);
    return sec;
  }
  const w = h('input', { class: 'field num', value: fmt(under.width, 4), style: { width: '70px' }, 'aria-label': 'Image width' });
  w.addEventListener('change', () => { const v = Number(w.value); if (v > 0) under.set({ width: v }); else w.value = fmt(under.width, 4); });
  const op = h('input', { type: 'range', min: '0.1', max: '1', step: '0.05', value: String(under.opacity), 'aria-label': 'Image opacity', 'data-tip': 'How strongly the picture shows' });
  op.addEventListener('input', () => under.set({ opacity: Number(op.value) }));
  const eye = h('button', { class: 'btn small', 'aria-pressed': String(under.shown), 'data-tip': under.shown ? 'Hide the picture' : 'Show the picture' }, under.shown ? 'Hide' : 'Show');
  eye.addEventListener('click', () => { under.set({ shown: !under.shown }); rerender(); });
  const rm = h('button', { class: 'btn small danger', 'data-tip': 'Remove the picture' }, 'Remove');
  rm.addEventListener('click', () => { under.clear(); rerender(); });
  sec.append(h('div', { class: 'muted', style: { fontSize: '12px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, under.name),
    h('label', { class: 'row' }, 'width', h('span', {}, w, ` ${unit}`)), h('label', { class: 'row' }, 'opacity', op),
    h('div', { style: { display: 'flex', gap: '6px' } }, eye, rm));
  return sec;
}
