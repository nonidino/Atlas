// The header: the five tabs, the status chip and its problems list, Run, the theme,
// and the Case menu (new, examples, open, save, import, export).

import { store, subscribe, send, changed, family } from './store.js';
import { h, menu, dialog, toast, setTheme, theme, download, clear, refreshTip } from './ui.js';

export function showTab(name) {
  store.tab = name;
  for (const b of document.querySelectorAll('.tab')) {
    b.setAttribute('aria-selected', String(b.dataset.tab === name));
  }
  for (const v of document.querySelectorAll('.view')) v.hidden = v.dataset.view !== name;
  changed('tab');
}

function runActive() {
  const s = store.run?.status;
  return s === 'starting' || s === 'running' || s === 'stopping';
}

export async function runOrStop() {
  if (runActive()) { await send('run.stop'); return; }
  const r = await send('run.start');
  if (r.ok) showTab('run');
}

async function openDialog() {
  const r = await send('case.files');
  if (!r.ok) return;
  const list = h('div', { class: 'items', style: { maxHeight: '320px' } });
  let picked = null;
  let p = null;
  if (!r.files.length) list.append(h('div', { class: 'empty' }, 'No saved cases yet.'));
  for (const f of r.files) {
    const b = h('button', { class: 'item', 'aria-selected': 'false' },
      h('span', { class: 'name' }, f.name), h('span', { class: 'meta' }, f.modified.replace('T', ' ')));
    b.addEventListener('click', () => {
      for (const x of list.children) x.setAttribute('aria-selected', 'false');
      b.setAttribute('aria-selected', 'true');
      picked = f.name;
    });
    b.addEventListener('dblclick', () => p.close(f.name));
    list.append(b);
  }
  p = dialog('Open a case', [h('div', { class: 'muted' }, 'Saved in out/workbench/cases.'), list],
    [{ label: 'Cancel' }, { label: 'Open', primary: true, value: () => picked || false }]);
  const name = await p;
  if (name) await send('case.open', { name });
}

async function saveAsDialog() {
  const input = h('input', { class: 'field', style: { width: '100%' },
    value: store.state?.meta?.file?.replace(/\.json$/, '') || store.state?.meta?.key || '' });
  const name = await dialog('Save as', [
    h('label', { class: 'muted' }, 'File name, in out/workbench/cases'), input],
  [{ label: 'Cancel' }, { label: 'Save', primary: true, value: () => input.value.trim() || false }]);
  if (name) await send('case.save', { name });
}

async function save() {
  if (store.state?.meta?.file) await send('case.save');
  else await saveAsDialog();
}

function importDialog() {
  const input = h('input', { type: 'file', accept: '.json,application/json' });
  input.addEventListener('change', async () => {
    const f = input.files && input.files[0];
    if (!f) return;
    const text = await f.text();
    await send('case.import', { text, name: f.name });
  });
  input.click();
}

async function exportCase() {
  const r = await send('case.export');
  if (r.ok) download(r.name, r.text);
}

async function historyDialog() {
  const r = await send('case.history');
  if (!r.ok) return;
  await dialog('This session', [h('pre', { class: 'mono', style: { fontSize: '11.5px', margin: 0, whiteSpace: 'pre-wrap' } },
    r.lines.join('\n') || 'Nothing yet.')]);
}

export function problemsDialog() {
  const st = store.state;
  if (!st) return;
  const rows = st.issues || [];
  const body = h('div', { class: 'issues' });
  if (!rows.length) body.append(h('div', { class: 'note ok' }, 'Everything needed to run is set.'));
  const fixable = rows.filter((i) => i.fix).map((i) => i.fix);
  let p = null;
  for (const i of rows) {
    const acts = h('div', { class: 'acts' });
    if (i.fix) {
      const b = h('button', { class: 'btn small primary', 'data-tip': i.fix_label }, 'Fix');
      b.addEventListener('click', async () => { p.close(); await send('case.fix', { keys: [i.fix] }); });
      acts.append(b);
    }
    const show = h('button', { class: 'btn small', 'data-tip': 'Open the tab where this is set' }, 'Show me');
    show.addEventListener('click', () => { p.close(); showTab(i.where); });
    acts.append(show);
    body.append(h('div', { class: `issue ${i.severity}` }, h('span', { class: 'sev' }),
      h('span', { class: 'msg' }, i.message), acts));
  }
  const buttons = [{ label: 'Close' }];
  if (fixable.length > 1) buttons.unshift({ label: `Fix all ${new Set(fixable).size}`, value: 'all' });
  const n = rows.length;
  p = dialog(n ? `${n} ${n === 1 ? 'note' : 'notes'}` : 'Ready', [body], buttons);
  p.then(async (v) => { if (v === 'all') await send('case.fix', { keys: [...new Set(fixable)] }); });
}

function caseMenu(anchor) {
  const cat = store.catalog;
  if (!cat) return;
  const fid = store.state?.meta?.family;
  const kinds = cat.families.filter((f) => f.runs).map((f) => ({
    label: f.short[0].toUpperCase() + f.short.slice(1), tip: f.label,
    action: () => send('case.new', { family: f.id }) }));
  const exOf = (fam) => cat.examples.filter((e) => e.family === fam).map((e) => ({
    label: e.key, tip: e.shows + (e.fast ? ' (the fast example)' : '') + (e.fixed ? ' Read-only.' : ''),
    action: () => send('case.example', { key: e.key }) }));
  const others = cat.families.filter((f) => f.runs && f.id !== fid).map((f) => ({
    label: f.short[0].toUpperCase() + f.short.slice(1), sub: exOf(f.id) }));
  const fam = family();
  menu(anchor, [
    { label: 'New', sub: kinds, tip: 'A new case of a kind, ready to run; Undo brings this one back' },
    { label: 'Start over', tip: 'A new case of this kind', action: () => send('case.start_over') },
    'sep',
    { header: fam ? `${fam.short} examples` : 'Examples' },
    ...exOf(fid),
    { label: 'Other kinds', sub: others },
    ...(fam?.fast_example ? [{ label: 'Fast example', tip: 'This kind\'s fastest honest setup', action: () => send('case.fast') }] : []),
    'sep',
    { label: 'Open…', kb: 'Ctrl+O', action: openDialog },
    { label: 'Save', kb: 'Ctrl+S', action: save },
    { label: 'Save as…', action: saveAsDialog },
    { label: 'Revert', tip: 'Back to the saved file', action: () => send('case.revert') },
    'sep',
    { label: 'Import JSON…', action: importDialog },
    { label: 'Export JSON', action: exportCase },
    'sep',
    { label: 'History', tip: 'What this session did', action: historyDialog },
  ]);
}

function render() {
  const st = store.state;
  const chip = document.getElementById('chip');
  const status = store.status || st?.status;
  if (status) {
    chip.className = `chip ${status.kind === 'ok' ? '' : status.kind}`;
    chip.textContent = status.text;
    chip.setAttribute('data-tip', status.tip + (status.kind !== 'busy' ? ' · click for the list' : ''));
  }
  const run = document.getElementById('run');
  const active = runActive();
  run.textContent = active ? 'Stop' : 'Run';
  run.classList.toggle('stop', active);
  run.setAttribute('data-tip', active ? 'Stop after the step in progress' : 'Run the case · Ctrl+Enter');
  refreshTip(run);
  refreshTip(chip);
  const label = document.getElementById('filelabel');
  if (st) {
    const m = st.meta;
    clear(label);
    label.append(h('b', {}, m.file ? m.file.replace(/\.json$/, '') : m.key));
    label.append(m.read_only ? ' · read-only' : (m.dirty ? ' · unsaved' : (m.file ? ' · saved' : '')));
    label.setAttribute('data-tip', `${m.family_label} case` + (m.file ? `, saved as ${m.file}` : ', not saved to a file yet'));
  }
}

export function install() {
  for (const b of document.querySelectorAll('.tab')) b.addEventListener('click', () => showTab(b.dataset.tab));
  document.getElementById('chip').addEventListener('click', problemsDialog);
  document.getElementById('run').addEventListener('click', runOrStop);
  document.getElementById('theme').addEventListener('click', () => setTheme(theme() === 'dark' ? 'light' : 'dark'));
  document.getElementById('casemenu').addEventListener('click', (e) => caseMenu(e.currentTarget));
  document.addEventListener('keydown', (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key === 'Enter') { e.preventDefault(); runOrStop(); }
    else if (mod && (e.key === 's' || e.key === 'S')) { e.preventDefault(); save(); }
    else if (mod && (e.key === 'o' || e.key === 'O')) { e.preventDefault(); openDialog(); }
    else if (!typing && mod && e.shiftKey && (e.key === 'z' || e.key === 'Z')) { e.preventDefault(); send('edit.redo'); }
    else if (!typing && mod && (e.key === 'z' || e.key === 'Z')) { e.preventDefault(); send('edit.undo'); }
    else if (!typing && mod && (e.key === 'y' || e.key === 'Y')) { e.preventDefault(); send('edit.redo'); }
  });
  subscribe((what) => { if (what !== 'fields') render(); });
}

export { save, openDialog, saveAsDialog, toast };
