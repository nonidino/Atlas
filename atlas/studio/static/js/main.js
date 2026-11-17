// The studio's page: connect, then let each view draw what the server sends.

import * as net from './net.js';
import { store, bind, subscribe } from './store.js';
import { installTips, toast } from './ui.js';
import * as header from './header.js';
import * as geometry from './views/geometry.js';
import * as physics from './views/physics.js';
import * as pieces from './views/pieces.js';
import * as run from './views/run.js';
import * as records from './views/records.js';

const view = (name) => document.querySelector(`.view[data-view="${name}"]`);

installTips();
bind();
header.install();
geometry.mount(view('geometry'));
physics.mount(view('physics'));
pieces.mount(view('pieces'));
run.mount(view('run'));
records.mount(view('records'));

net.on('notice', (m) => toast(m.kind, m.text, m.ms));
let lost = false;
net.on('connection', (m) => {
  if (!m.connected) { lost = true; toast('warning', 'The connection to the server dropped; reconnecting…', 4000); }
  else if (lost) { lost = false; toast('success', m.resumed ? 'Reconnected: the same case.' : 'Reconnected to a new session.', 3000); }
});
net.start();

// for the page checks: the page's own state, read without a screenshot
window.atlas = { store, net, subscribe };
