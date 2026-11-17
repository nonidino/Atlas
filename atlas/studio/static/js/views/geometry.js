// Geometry, phase P0: the case's shapes as the engine holds them, read-only.
// P1 replaces the read-only canvas with the drawing tools and the feature tree.

import { store, subscribe, family } from '../store.js';
import { h, clear, svgIcon, cssVar, fmt } from '../ui.js';
import { Stage, SWATCHES } from '../canvas.js';

let stage = null;
let root = null;
let selected = null;

function materialColour(name, mats) {
  const i = Math.max(0, mats.indexOf(name));
  return SWATCHES[i % SWATCHES.length];
}

function drawCase() {
  const st = store.state;
  if (!stage || !st) return;
  const d = st.drawing;
  stage.clear();
  stage.main.activate();
  const fill = cssVar('--shape-fill'), stroke = cssVar('--shape-stroke');
  const canvasBg = cssVar('--canvas');
  const dom = d.outline ? [d.outline] : [[[0, 0], [d.width, 0], [d.width, d.height], [0, d.height]]];
  const domain = stage.rings(dom, { fill, stroke, width: 1.6, name: 'domain' });
  stage.main.addChild(domain);
  // a region holds the domain's cells inside it: drawn clipped to the domain
  let solid = domain.clone({ insert: false });
  for (const ring of d.holes) solid = solid.subtract(stage.rings([ring]), { insert: false });
  const mats = Object.keys(st.case.materials || {});
  d.regions.forEach((r) => {
    const c = materialColour(r.material, mats);
    const raw = stage.rings(r.rings);
    const it = raw.intersect(solid, { insert: false });
    stage.main.addChild(Object.assign(it, {}));
    it.fillColor = new paper.Color(c); it.fillColor.alpha = 0.45;
    it.strokeColor = c; it.strokeWidth = selected === `region:${r.id}` ? 2.6 : 1; it.strokeScaling = false;
  });
  d.holes.forEach((ring, i) => {
    stage.main.addChild(stage.rings([ring], { fill: canvasBg, stroke, width: selected === `hole:${i}` ? 2.6 : 1.6 }));
  });
  if (selected === 'domain') stage.main.addChild(stage.rings(dom, { stroke: cssVar('--accent'), width: 2.6 }));
  stage.setBounds(0, 0, d.width, d.height);
  stage.update();
}

function featureList() {
  const st = store.state;
  const box = h('div', { class: 'items' });
  if (!st) return box;
  const d = st.drawing;
  const unit = st.meta.unit;
  const mats = Object.keys(st.case.materials || {});
  const item = (key, sw, name, meta, swClass = '') => {
    const b = h('button', { class: 'item', 'aria-selected': String(selected === key) },
      h('span', { class: `sw ${swClass}`, style: sw ? { background: sw } : {} }),
      h('span', { class: 'name' }, name), h('span', { class: 'meta' }, meta));
    b.addEventListener('click', () => { selected = selected === key ? null : key; render(); });
    return b;
  };
  box.append(item('domain', cssVar('--accent'), d.outline ? 'Outline' : 'Grid',
    `${fmt(d.width)} × ${fmt(d.height)}`));
  d.holes.forEach((_r, i) => box.append(item(`hole:${i}`, null, `Hole ${i + 1}`, 'cut', 'cut')));
  d.regions.forEach((r) => box.append(item(`region:${r.id}`, materialColour(r.material, mats), r.id, r.material)));
  return box;
}

function inspector() {
  const st = store.state;
  const box = h('aside', { class: 'panel scroll', style: { width: '240px', flex: 'none' } });
  if (!st) return box;
  const d = st.drawing;
  const unit = st.meta.unit;
  const fam = family();
  if (selected && selected.startsWith('region:')) {
    const r = st.case.regions.find((x) => `region:${x.id}` === selected);
    if (r) {
      box.append(h('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
        h('span', { style: { fontSize: '17px', fontWeight: 700, flex: 1 } }, r.id),
        h('span', { class: 'tag region', 'data-tip': 'A region: inside the domain, with its own material' }, 'region')));
      box.append(h('div', { class: 'kv' }, h('span', { class: 'k' }, 'material'), h('span', {}, r.material)));
      const props = st.case.materials[r.material] || {};
      const kv = h('div', { class: 'kv' });
      for (const [k, v] of Object.entries(props)) kv.append(h('span', { class: 'k' }, k), h('span', {}, fmt(v, 4)));
      box.append(kv);
    }
  } else {
    box.append(h('div', { style: { fontSize: '17px', fontWeight: 700 } }, fam ? fam.short[0].toUpperCase() + fam.short.slice(1) : 'Case'));
    box.append(h('div', { class: 'kv' },
      h('span', { class: 'k', 'data-tip': 'Cells along x and y' }, 'grid'), h('span', {}, `${d.nx} × ${d.ny}`),
      h('span', { class: 'k', 'data-tip': 'The side of one cell' }, 'cell'), h('span', {}, `${fmt(d.dx, 4)} ${unit}`),
      h('span', { class: 'k' }, 'size'), h('span', {}, `${fmt(d.width)} × ${fmt(d.height)}`),
      h('span', { class: 'k' }, 'holes'), h('span', {}, String(d.holes.length)),
      h('span', { class: 'k' }, 'regions'), h('span', {}, String(d.regions.length))));
  }
  box.append(h('span', { style: { flex: 1 } }));
  box.append(h('div', { class: 'note info' }, 'Drawing arrives in phase P1: pen, rectangle, ellipse, arc, combine, materials and the check.'));
  return box;
}

function render() {
  if (!root) return;
  const st = store.state;
  const featuresBox = root.querySelector('#geo-features');
  if (featuresBox) { clear(featuresBox).append(featureList()); }
  const insp = root.querySelector('#geo-inspector');
  if (insp) insp.replaceWith(Object.assign(inspector(), { id: 'geo-inspector' }));
  const foot = root.querySelector('#geo-foot');
  if (foot && st) {
    clear(foot);
    const errs = st.issues.filter((i) => i.where === 'geometry' && i.severity === 'error').length;
    foot.append(h('span', { class: errs ? 'err-text' : 'ok-text' }, errs ? `${errs} geometry problems` : '✓ shape ready'),
      h('span', { class: 'spacer' }),
      h('span', {}, h('kbd', {}, 'Space'), ' pan'), h('span', {}, h('kbd', {}, 'Ctrl'), '+wheel zoom'),
      h('span', {}, h('kbd', {}, 'Shift+1'), ' fit'));
  }
  drawCase();
}

export function mount(el) {
  root = el;
  const canvas = h('canvas', { 'aria-label': 'The case geometry' });
  const rail = h('nav', { class: 'rail', 'aria-label': 'Geometry tools' },
    h('button', { class: 'tool', 'aria-pressed': 'true', 'aria-label': 'Select', 'data-tip': 'Select · V. Click a shape; drag handles to reshape.' },
      svgIcon('M5 3l13 8-6 1.5L9.5 19z')),
    h('span', { class: 'fill' }));
  const features = h('aside', { class: 'panel', style: { width: '196px', flex: 'none' } },
    h('div', { class: 'label' }, 'Features', h('span', { class: 'q', 'data-tip': 'Applied top to bottom.' }, '?')),
    h('div', { id: 'geo-features', style: { minHeight: 0, overflow: 'auto' } }),
    h('div', { class: 'legend' },
      h('span', {}, h('i', { style: { background: 'var(--accent)' } }), 'add'),
      h('span', {}, h('i', { style: { border: '1.5px solid var(--err)' } }), 'cut'),
      h('span', {}, h('i', { style: { background: 'var(--ready)' } }), 'region')));
  const stageEl = h('main', { class: 'stage' },
    h('div', { class: 'bar' }, h('span', { class: 'muted' }, 'As the engine holds it'), h('span', { class: 'spacer' }),
      h('span', { class: 'muted', 'data-tip': 'Units for every length on screen', id: 'geo-unit' }, '')),
    h('div', { class: 'canvas-wrap' }, canvas),
    h('div', { class: 'foot', id: 'geo-foot' }));
  el.append(rail, h('div', { class: 'work' }, features, stageEl, Object.assign(inspector(), { id: 'geo-inspector' })));
  stage = new Stage(canvas);
  window.addEventListener('keydown', (e) => {
    if (e.shiftKey && e.code === 'Digit1' && stage.isShown()) { stage.fit(); }
  });
  window.addEventListener('themechange', () => { stage.drawGrid(); drawCase(); });
  subscribe((what) => {
    if (what === 'state' || what === 'catalog') {
      const u = root.querySelector('#geo-unit');
      if (u && store.state) u.textContent = store.state.meta.unit;
      render();
    }
    if (what === 'tab' && store.tab === 'geometry') { stage.view.viewSize = stage.view.viewSize; drawCase(); }
  });
}
