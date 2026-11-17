"""The third-party files the studio's page loads, each pinned by sha256.

The page has no build step and installs nothing from npm: every library it uses
is a file in ``static/``, copied unmodified from the package named here, with its
own licence beside it (the root ``THIRD-PARTY-NOTICES.md`` lists them).  The owner
approved each download on 2026-11-16 (plan page ``integrated-ui-plan.md``, section 10).

`check()` hashes every file against this table; the tests and the installer's
self-test both call it, so a file changed by hand, by a line-ending conversion or
by a half-finished copy is caught before the page is served.

**Ranade is not here.**  Fontshare marks it "Closed Source / ITF Free Font
License", whose section 02 forbids distributing the font through "a repository" or
"publicly accessible servers".  It is fetched at install time from Fontshare itself
(`fonts.py`), so each person obtains their own copy, and it is never committed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@dataclass(frozen=True)
class Package:
    name: str
    version: str
    source: str
    licence: str
    #: path under static/ -> sha256 of the file as downloaded
    files: tuple[tuple[str, str], ...]


PACKAGES: tuple[Package, ...] = (
    Package(
        "Paper.js", "0.12.18", "https://cdn.jsdelivr.net/npm/paper@0.12.18/",
        "MIT (Juerg Lehni and Jonathan Puckey)",
        (("vendor/paper/paper-core.min.js",
          "3a8c73ecdf3929a7fe05f96b82e8146193ea449c5c3e889c5f4fe0184a30ed96"),
         ("vendor/paper/LICENSE.txt",
          "f8833a38807922b4652421eb0cf29065400c55abd6f4832b9c3250dae87e1958"))),
    Package(
        "planegcs", "1.3.0", "https://cdn.jsdelivr.net/npm/@salusoft89/planegcs@1.3.0/",
        "LGPL-2.1 (the LICENSE file; package.json says LGPL-2.0-or-later); a WebAssembly "
        "build of FreeCAD's PlaneGCS, unmodified",
        (("vendor/planegcs/LICENSE",
          "20c17d8b8c48a600800dfd14f95d5cb9ff47066a9641ddeab48dc54aec96e331"),
         ("vendor/planegcs/package.json",
          "b3b0b656e303900e579d222a02e285b331b3bdfd2889ecd51a2f1783b6667652"),
         ("vendor/planegcs/dist/index.js",
          "05fef485d533fd3f46c63c1cff6e1770295c09edc59592fd8227939b763df4a8"),
         ("vendor/planegcs/dist/planegcs_dist/constraint_param_index.js",
          "f47b001bce8ca26221c3590a5ee6734a08cd53393f0b0dd90a114067d868555a"),
         ("vendor/planegcs/dist/planegcs_dist/constraints.js",
          "b2c9603a34111cbac194ad30cfa7c388ae10485743e17a630ea401f1cac76411"),
         ("vendor/planegcs/dist/planegcs_dist/enums.js",
          "ed3bc0783f6136d57618f15647581ff5fd7d4ae8e51f36ad347c16dfd752f398"),
         ("vendor/planegcs/dist/planegcs_dist/gcs_system.js",
          "66ee867255e543d3847eceecaf35e0cbe4e760a5ebd49b735fed5f25aa1370eb"),
         ("vendor/planegcs/dist/planegcs_dist/id.js",
          "ad92fff0948a4e70521aefc5bb5d431e51dfb09debe011d8b52897bd74601c97"),
         ("vendor/planegcs/dist/planegcs_dist/planegcs.js",
          "d541e52942d9480438ea44f2ca05bb07dcc216e99da29d7aa31d3cc60ad6ef81"),
         ("vendor/planegcs/dist/planegcs_dist/planegcs.wasm",
          "af9d1874fc895726bd267703a698a86c82d35b7415a02fe5184ebf65ddbc04af"),
         ("vendor/planegcs/dist/sketch/emsc_vectors.js",
          "0e779089c582ecea186c3b5d0210111ce411fb15b3dce416f5d6f6b8d76cf2c3"),
         ("vendor/planegcs/dist/sketch/gcs_wrapper.js",
          "8081a702f9cb1ac997b0dc75e304187d28a9e171d558e26aaf6a70dd3d8e3263"),
         ("vendor/planegcs/dist/sketch/geom_params.js",
          "76b42afcf00e674abcb1d10e8aca15ddbf28bc299206ac8d9400ee7dda7d78e8"),
         ("vendor/planegcs/dist/sketch/sketch_index.js",
          "60985b932918d592bd7d63ff9e89b825ef617b29d3bb66f568d2348e0c62c276"),
         ("vendor/planegcs/dist/sketch/sketch_primitive.js",
          "0910914f40718184a36ec1c0fb6947fbe26bbc941b8b04d35f96e2230a6c2959"))),
    Package(
        "JetBrains Mono (Fontsource)", "5.3.0",
        "https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5.3.0/",
        "SIL Open Font License 1.1 (The JetBrains Mono Project Authors)",
        (("fonts/jetbrains-mono/LICENSE",
          "403581b69dac5cff4079205e01c6b467e56af449ecbd7247693ddb1baafa005b"),
         ("fonts/jetbrains-mono/jetbrains-mono-latin-400-normal.woff2",
          "14425ba9c695763c1547f48a206b7aa60350a33ae23de09f0407877f3fcd89eb"),
         ("fonts/jetbrains-mono/jetbrains-mono-latin-600-normal.woff2",
          "400c6bfda18d5d14acad1c15d6dcb9f8e13c015e7286317e0b9a482539bef147"),
         ("fonts/jetbrains-mono/jetbrains-mono-latin-ext-400-normal.woff2",
          "505dfba8ecbe77e82765f36d317ed7ef4ac42719dc5f4ae68d1c483fd22d0d14"),
         ("fonts/jetbrains-mono/jetbrains-mono-latin-ext-600-normal.woff2",
          "90899aac1c15552c028fd205376bb537a2a5205164fe8626efabad77d7260c20"),
         ("fonts/jetbrains-mono/jetbrains-mono-greek-400-normal.woff2",
          "365b7a8eaff022b13ed0561fde3890a04a30507d773ecd5adb3aca8e766a285c"),
         ("fonts/jetbrains-mono/jetbrains-mono-greek-600-normal.woff2",
          "de02415369898d9aa2d9f30379b6c201fc1c0a7017589d973747d993a68f709b"))),
    Package(
        "Instrument Sans (Fontsource)", "5.3.0",
        "https://cdn.jsdelivr.net/npm/@fontsource/instrument-sans@5.3.0/",
        "SIL Open Font License 1.1 (The Instrument Sans Project Authors); the page's "
        "font when Ranade is not installed",
        (("fonts/instrument-sans/LICENSE",
          "c27a3c53c3beed7f5c26853afa15991478ff7145d3754a36b0382f84e10c0d03"),
         ("fonts/instrument-sans/instrument-sans-latin-400-normal.woff2",
          "9a91efaa3599a6ef01b2f49e9ec670e3a2f39baeef09c73198f1052cf3bb0850"),
         ("fonts/instrument-sans/instrument-sans-latin-500-normal.woff2",
          "9d3547da16d950d12a677c772a3ff9b69846e60e7dcd1b5efd540993f7286532"),
         ("fonts/instrument-sans/instrument-sans-latin-600-normal.woff2",
          "efb770916eda95a0e31d80cccde9651d4e8617010b21220e9f227c4820d675da"),
         ("fonts/instrument-sans/instrument-sans-latin-700-normal.woff2",
          "bcf650536cae57403741af193e1e237c245f8929b936f03601279079a64fd699"))),
)


def files() -> dict[str, str]:
    """Every vendored path under static/, with its sha256."""
    return {p: h for pkg in PACKAGES for p, h in pkg.files}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def check(static: str = STATIC) -> list[str]:
    """What is wrong with the vendored files, in words; empty when every file is
    present and hashes to its pinned value."""
    bad = []
    for rel, want in files().items():
        full = os.path.join(static, *rel.split("/"))
        if not os.path.isfile(full):
            bad.append(f"{rel}: missing")
            continue
        got = sha256(full)
        if got != want:
            bad.append(f"{rel}: sha256 {got[:12]}..., pinned {want[:12]}...")
    return bad


__all__ = ["PACKAGES", "Package", "STATIC", "check", "files", "sha256"]
