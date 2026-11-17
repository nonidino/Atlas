# Atlas studio

One app to draw a domain, choose its physics, cut it into pieces, run it, and
read the guarantees: the integrated UI of
`wiki/concepts/Atlas 0.1/atlas-0.1-proposal/demo/integrated-ui-plan.md`, which
replaces the Panel workbench (`atlas/workbench/app.py` and its page modules).

```bash
python -m atlas.studio --open        # http://127.0.0.1:8021/
```

`--port`, `--host` and `--cases-dir` are optional; cases are saved to
`out/workbench/cases/`, the folder the workbench used, so a case saved in either
opens in the other. `--fetch-fonts` downloads Ranade from Fontshare and exits.

## What is built

| phase | what | state |
|---|---|---|
| P0 | the shell: FastAPI and one JavaScript page over a WebSocket; the header, the two themes, hover explanations; open and save of case files bit for bit; the vendored libraries pinned by hash; the bundle's pins and self-test | built |

## How it works

**One page, one socket.** The server (`server.py`) serves `static/index.html`, its
plain JavaScript modules (no build step, nothing from npm) and the vendored files,
and holds one `Session` per open page. Every action the page takes is a message
over `/ws` (`protocol.py`): the server checks the message against the handler's
arguments, applies it through the session, and answers with a reply, after sending
what changed, the case with its problems and the geometry to draw (`state`).
Notices, run progress, fields and compile verdicts are pushed as they happen.

**The session** (`session.py`) is the retired workbench's `Workbench` without its
GUI library: a case is what it simulates (another kind starts a new case of it);
every edit is validated as a whole case and refused whole; windows that follow the
domain are cut again when it changes; a newly drawn shape or physics gets what it
lacks to run, each thing said; one Undo restores a whole case with its file and
key; a run is a commit, and the page says which version is marching.

**Edits are data.** `edit.apply` carries a list of operations on the case as JSON
(`set`, `insert`, `remove`, `move`, each with a path from the root); the result is
validated as a `CaseSpec` before anything is kept. A test drives every message
without a browser (`tests/test_studio_shell.py`).

**The price is a ratio.** A run's record keeps every per-step time; the page is
sent only what `runcards.py` derives from it: the full domain's mean step time over
each decomposed arm's, each arm's own spread (90th over 10th percentile, the A/A
noise), and the registered checks with their tolerances. No raw seconds reach the
page.

**The page reconnects to its own case.** A dropped socket or a reload in the same
tab asks for its session back (kept for an hour), so nothing is lost by a theme
switch, which never reloads anyway.

## The page

| file | role |
|---|---|
| `static/index.html` | the shell: the header (tabs, status chip, Run, theme, Case) and the five views |
| `static/css/tokens.css` | the two themes as tokens: the owner's palette, the dark values from the mockup canvas |
| `static/css/app.css` | the layout at 1090 x 620: 48 px header, 52 px rail, 8 px gutters |
| `static/js/net.js` | the socket: requests with replies, pushes, reconnection |
| `static/js/store.js` | what the page knows (catalog, state, run frames) and the edit helper |
| `static/js/ui.js` | the element builder, hover explanations, notices, menus, dialogs, the theme |
| `static/js/glossary.js` | every shorthand the page shows, explained (G0 to G3, TL0, H2r, det J, Fo, Pe, A/A, ...) |
| `static/js/header.js` | the tabs, the status chip and its problems list with one-click fixes, Run, the Case menu |
| `static/js/canvas.js` | a Paper.js stage in the physics' length unit: zoom, pan, fit, grid |
| `static/js/views/*.js` | the five views: geometry, physics, pieces, run, records |

## The vendored files

Paper.js 0.12.18 (MIT), planegcs 1.3.0 (LGPL-2.1), and the JetBrains Mono and
Instrument Sans fonts (OFL), each unmodified with its licence beside it and pinned
by sha256 in `vendor.py`; the root `THIRD-PARTY-NOTICES.md` lists them. **Ranade is
not committed**: its licence (ITF Free Font License, section 02) forbids distribution
through a repository, so `fonts.py` fetches it from Fontshare at install time and the
page falls back to Instrument Sans without it. The server writes the page's font CSS
from the files present, so a missing font is never a failed request.

## Tests

`tests/test_studio_shell.py`: every vendored file against its hash; the page and every
file it and its modules name served from this machine with the MIME types a browser
requires; no file of the page names the network; every message the page sends has a
handler, and every handler refuses arguments it does not take; edits, refusals, undo;
every example opened and saved bit for bit; a run over the socket ending in cards with
no raw seconds. The installer's self-test (`run.py --check`) starts the studio's own
server and runs a case over its WebSocket.

## Known limitations

- **The geometry tab is read-only until P1.** It draws the case as the engine holds it.
- **Case files are written in text mode** (`CaseSpec.save`, unchanged), so a file is
  CRLF on Windows and LF elsewhere; open-then-save is bit for bit on each.
