// Pieces, phases P0-P1: the windows the case is cut into, as the engine holds them.
// P4 brings the four-sided pieces, their charts and certificates, and an expert
// per piece.

import { store, subscribe, send } from '../store.js';
import { h, clear, cssVar } from '../ui.js';
import { Stage, SWATCHES } from '../canvas.js';

let root = null;
let stage = null;

function draw() {
  const st = store.state;
  if (!stage || !st) return;
  const d = st.drawing;
  stage.clear();
  const dom = d.outline ? [d.outline] : [[[0, 0], [d.width, 0], [d.width, d.height], [0, d.height]]];
  stage.main.addChild(stage.rings(dom, { fill: cssVar('--shape-fill'), stroke: cssVar('--shape-stroke'), width: 1.4 }));
  d.holes.forEach((ring) => stage.main.addChild(stage.rings([ring], { fill: cssVar('--canvas'), stroke: cssVar('--shape-stroke') })));
  d.windows.forEach((w, i) => {
    const c = SWATCHES[i % SWATCHES.length];
    stage.main.addChild(stage.rings(w.rings, { fill: c, fillAlpha: 0.18, stroke: c, width: 1.6, dash: [5, 3] }));
  });
  stage.setBounds(0, 0, d.width, d.height);
  stage.update();
}

function render() {
  const st = store.state;
  const list = root?.querySelector('#pc-list');
  if (!list || !st) return;
  clear(list);
  const c = st.case;
  st.drawing.windows.forEach((w, i) => {
    list.append(h('div', { class: 'item' }, h('span', { class: 'sw', style: { background: SWATCHES[i % SWATCHES.length] } }),
      h('span', { class: 'name' }, w.id)));
  });
  if (!st.drawing.windows.length) list.append(h('div', { class: 'empty' }, 'No windows yet.'));
  const how = root.querySelector('#pc-how');
  clear(how);
  if (c.layout) how.append(h('div', { class: 'note info' }, `Automatic: cut ${c.layout.cut === 'materials' ? 'one per material' : `along the shape into ${c.layout.along}`}` + (c.layout.how ? ` (${c.layout.how})` : '')));
  else how.append(h('div', { class: 'note info' }, 'The case\'s own windows.'));
  const fixWin = st.issues.find((i) => i.fix === 'windows');
  if (fixWin) {
    const b = h('button', { class: 'btn primary' }, 'Cut automatically');
    b.addEventListener('click', () => send('case.fix', { keys: ['windows'] }));
    how.append(b);
  }
  draw();
}

export function mount(el) {
  root = el;
  const canvas = h('canvas', { 'aria-label': 'The pieces' });
  el.append(h('div', { class: 'work' },
    h('aside', { class: 'panel', style: { width: '220px', flex: 'none' } },
      h('div', { class: 'label' }, 'Windows', h('span', { class: 'q', 'data-tip': 'The pieces the case is solved on, each by its own solver, coupled at their seams.' }, '?')),
      h('div', { id: 'pc-list', class: 'items' }), h('div', { class: 'rule' }), h('div', { id: 'pc-how', style: { display: 'flex', flexDirection: 'column', gap: '6px' } })),
    h('main', { class: 'stage' }, h('div', { class: 'bar' }, h('span', { class: 'muted' }, 'Windows over the domain')),
      h('div', { class: 'canvas-wrap' }, canvas)),
    h('aside', { class: 'panel', style: { width: '240px', flex: 'none' } },
      h('div', { class: 'note info' }, 'Four-sided pieces, their charts and certificates, and an expert per piece arrive in phase P4.'))));
  stage = new Stage(canvas);
  window.addEventListener('themechange', () => { stage.drawGrid(); draw(); });
  subscribe((what) => { if (what === 'state' || (what === 'tab' && store.tab === 'pieces')) render(); });
}
