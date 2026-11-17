// What the page knows: the server's catalog and state, the last run frames, and
// the page's own view state (tab, theme).  Views subscribe and redraw.

import * as net from './net.js';

const subs = new Set();

export const store = {
  catalog: null,          // kinds of simulation and examples (never changes)
  state: null,            // the case, its problems, the status chip, the drawing
  status: null,           // the chip, also pushed on its own while a job runs
  run: null,              // the last run frame
  fields: null,           // the last fields frame
  compile: null,          // the last compile frame
  sweep: null,            // the last sweep frame: one run per parameter value
  tab: 'geometry',
  connected: false,
};

export function subscribe(fn) { subs.add(fn); return () => subs.delete(fn); }

export function changed(what) {
  for (const fn of subs) {
    try { fn(what); } catch (err) { console.error(err); }
  }
}

export function family(id = store.state?.meta?.family) {
  return store.catalog?.families.find((f) => f.id === id) || null;
}

export function bind() {
  net.on('catalog', (m) => { store.catalog = m; changed('catalog'); });
  net.on('state', (m) => { store.state = m; store.status = m.status; changed('state'); });
  net.on('status', (m) => { store.status = m.status; changed('status'); });
  net.on('run', (m) => { store.run = m; changed('run'); });
  net.on('fields', (m) => { store.fields = m; changed('fields'); });
  net.on('compile', (m) => { store.compile = m; changed('compile'); });
  net.on('sweep', (m) => { store.sweep = m; changed('sweep'); });
  net.on('connection', (m) => { store.connected = m.connected; changed('connection'); });
}

/** Send a message; a refusal is already a notice from the server. */
export async function send(type, args = {}) {
  return net.request(type, args);
}

/** One edit, as data (`session.apply_ops`). */
export function edit(ops, label = 'edit', defaults = false) {
  return net.request('edit.apply', { ops, label, defaults });
}
