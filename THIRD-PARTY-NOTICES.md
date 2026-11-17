# Third-party notices

Atlas itself carries no licence yet (the owner's decision, 2026-11-16: not now). The
files below are other people's work, copied unmodified into this repository so the
studio's page needs no network and no npm. Each keeps its own licence beside it, and
each is pinned by sha256 in `atlas/studio/vendor.py`, which the tests and the
installer's self-test check.

| package | version | where it lives | licence | source |
|---|---|---|---|---|
| Paper.js | 0.12.18 | `atlas/studio/static/vendor/paper/` (`paper-core.min.js`) | MIT, Copyright (c) 2011 - 2020, Jürg Lehni & Jonathan Puckey (`LICENSE.txt` beside it) | <https://cdn.jsdelivr.net/npm/paper@0.12.18/> |
| planegcs | 1.3.0 | `atlas/studio/static/vendor/planegcs/` | GNU Lesser General Public License 2.1 (the package's `LICENSE` file, beside it; its `package.json` says LGPL-2.0-or-later). A WebAssembly build of FreeCAD's PlaneGCS solver, unmodified; its source is FreeCAD's `src/Mod/Sketcher/App/planegcs` and the package's own repository (`github.com/Salusoft89/planegcs`) | <https://cdn.jsdelivr.net/npm/@salusoft89/planegcs@1.3.0/> |
| JetBrains Mono (Fontsource) | 5.3.0 | `atlas/studio/static/fonts/jetbrains-mono/` | SIL Open Font License 1.1, The JetBrains Mono Project Authors (`LICENSE` beside it) | <https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5.3.0/> |
| Instrument Sans (Fontsource) | 5.3.0 | `atlas/studio/static/fonts/instrument-sans/` | SIL Open Font License 1.1, The Instrument Sans Project Authors (`LICENSE` beside it) | <https://cdn.jsdelivr.net/npm/@fontsource/instrument-sans@5.3.0/> |

**Not here: Ranade.** The page's own font is Ranade, from Fontshare, under the ITF
Free Font License, whose section 02 forbids distributing the font through a
repository or a public server. It is never committed (`.gitignore`); the installer
downloads it from Fontshare onto the machine it is used on
(`atlas/studio/fonts.py`), and without it the page is drawn in Instrument Sans.

**Not here either: Gmsh** (GPL-2.0-or-later), which the installer installs from PyPI
as an optional step, and the build repository's wind-farm solver, which the
workbench bundle carries with that repository's own MIT licence (`vendor/LICENSE`).
