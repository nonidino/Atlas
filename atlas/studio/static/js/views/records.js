// Records, phase P6: the chart operator's registered evaluations, read-only.

import { h } from '../ui.js';

export function mount(el) {
  el.append(h('div', { class: 'work' }, h('div', { class: 'placeholder' },
    h('h2', {}, 'Chart-operator records'),
    h('div', {}, 'The registered evaluations (two-level, squares, hybrid, forms), their gates, prices and ρ arrive in phase P6. Nothing here re-runs anything.'))));
}
