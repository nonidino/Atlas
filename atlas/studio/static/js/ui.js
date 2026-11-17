// Small DOM helpers shared by every view: an element builder, the hover
// explanations, notices, menus, dialogs and the theme.

export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

export function svgIcon(d, size = 20, extra = '') {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('width', size); s.setAttribute('height', size);
  s.setAttribute('viewBox', '0 0 24 24'); s.setAttribute('fill', 'none');
  s.setAttribute('stroke', 'currentColor'); s.setAttribute('stroke-width', '1.8');
  s.setAttribute('stroke-linecap', 'round'); s.setAttribute('stroke-linejoin', 'round');
  s.setAttribute('aria-hidden', 'true');
  s.innerHTML = `<path d="${d}"></path>${extra}`;
  return s;
}

export function fmt(x, digits = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return '–';
  if (typeof x !== 'number') return String(x);
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e5 || a < 1e-3) return x.toExponential(Math.max(0, digits - 1)).replace('e+', 'e');
  return Number(x.toPrecision(digits)).toString();
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

// -- hover explanations --------------------------------------------------------
// Every element with data-tip explains itself on hover and on keyboard focus.  One
// floating box, kept inside the window (the old Case menu ran off its edge).
let tipTimer = null;
let tipFor = null;
function tipEl() { return document.getElementById('tip'); }

function placeTip(target) {
  const tip = tipEl();
  const text = target.getAttribute('data-tip');
  if (!text) return;
  tip.textContent = text;
  tip.classList.add('on');
  const r = target.getBoundingClientRect();
  const w = tip.offsetWidth, ht = tip.offsetHeight;
  let x = r.left + r.width / 2 - w / 2;
  let y = r.bottom + 6;
  if (y + ht > window.innerHeight - 4) y = r.top - ht - 6;
  x = Math.max(4, Math.min(window.innerWidth - w - 4, x));
  y = Math.max(4, y);
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

export function hideTip() {
  clearTimeout(tipTimer);
  tipFor = null;
  const tip = tipEl();
  if (tip) tip.classList.remove('on');
}

export function installTips() {
  const show = (ev) => {
    const t = ev.target.closest && ev.target.closest('[data-tip]');
    if (!t || t === tipFor) return;
    hideTip();
    tipFor = t;
    tipTimer = setTimeout(() => { if (tipFor === t && t.isConnected) placeTip(t); }, 260);
  };
  const leave = (ev) => {
    const t = ev.target.closest && ev.target.closest('[data-tip]');
    if (t && t === tipFor && !(ev.relatedTarget && t.contains(ev.relatedTarget))) hideTip();
  };
  document.addEventListener('mouseover', show);
  document.addEventListener('mouseout', leave);
  document.addEventListener('focusin', show);
  document.addEventListener('focusout', leave);
  document.addEventListener('mousedown', hideTip, true);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hideTip(); });
}

// -- notices -------------------------------------------------------------------
export function toast(kind, text, ms = 5000) {
  const box = document.getElementById('toasts');
  const close = h('button', { class: 'x', 'aria-label': 'Dismiss' }, '×');
  const el = h('div', { class: `toast ${kind}`, role: kind === 'error' ? 'alert' : 'status' },
    h('span', { class: 'dot' }), h('span', { class: 'text' }, text), close);
  close.addEventListener('click', () => el.remove());
  box.append(el);
  while (box.children.length > 4) box.firstChild.remove();
  setTimeout(() => el.remove(), ms);
}

// -- menus -----------------------------------------------------------------------
let openMenu = null;
export function closeMenu() {
  if (openMenu) { openMenu.remove(); openMenu = null; }
}

/** items: [{label, kb, tip, action, sub (items), disabled}] | 'sep' | {header} */
export function menu(anchor, items, { align = 'right' } = {}) {
  closeMenu();
  const el = h('div', { class: 'menu', role: 'menu' });
  for (const it of items) {
    if (it === 'sep') { el.append(h('div', { class: 'sep' })); continue; }
    if (it.header) { el.append(h('div', { class: 'mh' }, it.header)); continue; }
    const b = h('button', { class: 'mi', role: 'menuitem', 'data-tip': it.tip || null,
      disabled: it.disabled || null },
      h('span', {}, it.label), it.kb ? h('span', { class: 'kb' }, it.kb) : null,
      it.sub ? h('span', { class: 'sub' }, '›') : null);
    b.addEventListener('click', (e) => {
      e.stopPropagation();
      if (it.sub) { menu(b, it.sub, { align: 'side' }); return; }
      closeMenu();
      it.action && it.action();
    });
    el.append(b);
  }
  document.body.append(el);
  const r = anchor.getBoundingClientRect();
  const w = el.offsetWidth, ht = el.offsetHeight;
  let x = align === 'side' ? r.left - w - 4 : (align === 'left' ? r.left : r.right - w);
  if (align === 'side' && x < 4) x = r.right + 4;
  let y = align === 'side' ? r.top : r.bottom + 4;
  x = Math.max(4, Math.min(window.innerWidth - w - 4, x));
  y = Math.max(4, Math.min(window.innerHeight - ht - 4, y));
  el.style.left = `${x}px`; el.style.top = `${y}px`;
  openMenu = el;
  const first = el.querySelector('.mi:not([disabled])');
  if (first) first.focus({ preventScroll: true });
  return el;
}
document.addEventListener('mousedown', (e) => {
  if (openMenu && !openMenu.contains(e.target)) closeMenu();
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeMenu(); });

// -- dialogs ---------------------------------------------------------------------
export function dialog(title, body, buttons = [{ label: 'Close' }]) {
  let closer = null;
  const p = new Promise((resolve) => {
    const foot = h('footer');
    const scrim = h('div', { class: 'scrim' });
    const done = (v) => { scrim.remove(); resolve(v); };
    closer = done;
    for (const b of buttons) {
      const btn = h('button', { class: `btn ${b.primary ? 'primary' : ''}` }, b.label);
      btn.addEventListener('click', () => {
        const v = b.value !== undefined ? (typeof b.value === 'function' ? b.value() : b.value) : null;
        if (v === false) return;                       // the button's own check refused
        done(v);
      });
      foot.append(btn);
    }
    const close = h('button', { class: 'btn small', 'aria-label': 'Close' }, '×');
    close.addEventListener('click', () => done(null));
    const dlg = h('div', { class: 'dialog', role: 'dialog', 'aria-modal': 'true', 'aria-label': title },
      h('header', {}, h('h2', {}, title), close), h('div', { class: 'body' }, body), foot);
    scrim.append(dlg);
    scrim.addEventListener('mousedown', (e) => { if (e.target === scrim) done(null); });
    scrim.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { e.stopPropagation(); done(null); }
      if (e.key === 'Enter' && e.target.tagName === 'INPUT') {
        const p = buttons.find((b) => b.primary);
        if (p) { foot.querySelector('.primary').click(); }
      }
    });
    document.body.append(scrim);
    const focus = dlg.querySelector('input, select, textarea') || foot.querySelector('.primary') || close;
    focus.focus();
  });
  p.close = (v = null) => closer && closer(v);
  return p;
}

// -- the theme -------------------------------------------------------------------
export function theme() {
  const set = document.documentElement.dataset.theme;
  if (set === 'light' || set === 'dark') return set;
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('atlas.theme', t); } catch (e) { /* private window */ }
  window.dispatchEvent(new CustomEvent('themechange', { detail: t }));
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function download(name, text, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h('a', { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
