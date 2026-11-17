// The page's one connection to the server: requests with replies, and pushes.
//
//   const r = await net.request('case.example', {key: 'bend-3'});
//   net.on('state', s => ...);
//
// The socket reconnects on its own and asks for its own session back (kept by the
// server for an hour), so a dropped connection or a reload never loses the case.

const listeners = new Map();
const pending = new Map();
let sock = null;
let nextId = 1;
let sessionId = null;
let backoff = 400;
let connected = false;

try { sessionId = sessionStorage.getItem('atlas.session'); } catch (e) { sessionId = null; }

function emit(type, msg) {
  for (const fn of listeners.get(type) || []) {
    try { fn(msg); } catch (err) { console.error(err); }
  }
}

export function on(type, fn) {
  if (!listeners.has(type)) listeners.set(type, new Set());
  listeners.get(type).add(fn);
  return () => listeners.get(type).delete(fn);
}

export function isConnected() { return connected; }

function connect() {
  const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
  sock = new WebSocket(url);
  sock.onopen = () => {
    backoff = 400;
    sock.send(JSON.stringify({ type: 'hello', session: sessionId }));
  };
  sock.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    if (msg.type === 'hello') {
      sessionId = msg.session;
      try { sessionStorage.setItem('atlas.session', sessionId); } catch (e) { /* private window */ }
      connected = true;
      emit('connection', { connected: true, resumed: msg.resumed });
      return;
    }
    if (msg.type === 'reply') {
      const p = pending.get(msg.re);
      if (p) { pending.delete(msg.re); p(msg); }
      emit('reply', msg);
      return;
    }
    emit(msg.type, msg);
  };
  sock.onclose = () => {
    const was = connected;
    connected = false;
    for (const [id, p] of pending) p({ type: 'reply', re: id, ok: false, error: 'the connection to the server closed' });
    pending.clear();
    if (was) emit('connection', { connected: false });
    setTimeout(connect, backoff);
    backoff = Math.min(5000, backoff * 1.6);
  };
  sock.onerror = () => { /* onclose follows and reconnects */ };
}

export function request(type, args = {}) {
  return new Promise((resolve) => {
    if (!sock || sock.readyState !== WebSocket.OPEN || !connected) {
      resolve({ ok: false, error: 'not connected to the server' });
      return;
    }
    const id = nextId++;
    pending.set(id, resolve);
    sock.send(JSON.stringify({ id, type, ...args }));
  });
}

export function start() { connect(); }
