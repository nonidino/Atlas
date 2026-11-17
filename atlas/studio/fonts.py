"""The page's fonts: Ranade when it is installed, Instrument Sans when it is not.

**Ranade** (the owner's choice for the page) is Fontshare's, under the ITF Free
Font License.  Its section 02 forbids distributing the font "through another font
website, font library, marketplace, repository" or "publicly accessible servers",
and says that anyone wishing to use it must obtain their own copy from Fontshare.
So it is never committed (``.gitignore``), and `fetch_ranade` downloads it from
Fontshare's own API onto the machine it will be used on, at install time.  The
files are checked against the hashes of the copies the owner took from Fontshare
on 2026-11-16 (`RANADE`); Fontshare's API served the same bytes that day.

**Instrument Sans** (OFL, vendored) is the fallback: with no network at install
time, or a Fontshare that has changed its files, the page is drawn in it.  The CSS
the page loads (`font_css`) names only files that are present, so a missing font is
a quiet fallback, never a failed request in the page's console.

**JetBrains Mono** (OFL, vendored) sets every number.
"""

from __future__ import annotations

import os
import re
import urllib.request

from .vendor import STATIC, sha256

RANADE_DIR = os.path.join(STATIC, "fonts", "ranade")
#: weight -> sha256 of Fontshare's woff2, as served by its API on 2026-11-16
RANADE: dict[int, str] = {
    400: "8df8657d91de903cc26d42b7a17df0129dc2f8a20013bd65da8b4d8b0b2a92c3",
    500: "172b85b509d277ba35c1743450d09e821d2172cb36596db77fbc7a9cce013fac",
    700: "3812a59d9c9df3c8cb330101862e0455e21405fe9e0d16aba6e82c8773958850",
}
API = "https://api.fontshare.com/v2/css?f[]=ranade@400,500,700&display=swap"

_JB = ("latin", "U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,"
       "U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD"), \
      ("latin-ext", "U+0100-02BA,U+02BD-02C5,U+02C7-02CC,U+02CE-02D7,U+02DD-02FF,U+0304,"
       "U+0308,U+0329,U+1D00-1DBF,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20C0,"
       "U+2113,U+2C60-2C7F,U+A720-A7FF"), \
      ("greek", "U+0370-0377,U+037A-037F,U+0384-038A,U+038C,U+038E-03A1,U+03A3-03FF")


def ranade_path(weight: int) -> str:
    return os.path.join(RANADE_DIR, f"ranade-{weight}.woff2")


def ranade_installed() -> bool:
    """All three weights present and each the file Fontshare served."""
    for w, want in RANADE.items():
        p = ranade_path(w)
        if not os.path.isfile(p) or sha256(p) != want:
            return False
    return True


def fetch_ranade(timeout: float = 20.0, log=print) -> bool:
    """Download Ranade 400, 500 and 700 from Fontshare's API into ``RANADE_DIR``.

    Returns True when all three are installed and match `RANADE`.  Any failure (no
    network, a changed API, a file that does not hash as expected) leaves the folder
    without that weight and returns False: the page then uses Instrument Sans.
    Never raises."""
    if ranade_installed():
        log("    ok       Ranade already installed (from Fontshare)")
        return True
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0 (Atlas installer)")]
        css = opener.open(API, timeout=timeout).read().decode("utf-8", "replace")
        blocks = re.findall(r"@font-face\s*\{(.*?)\}", css, re.S)
        urls: dict[int, str] = {}
        for b in blocks:
            w = re.search(r"font-weight:\s*(\d+)", b)
            u = re.search(r"url\('([^']+\.woff2)'\)", b)
            if w and u:
                url = u.group(1)
                urls[int(w.group(1))] = ("https:" + url) if url.startswith("//") else url
        os.makedirs(RANADE_DIR, exist_ok=True)
        ok = True
        for w, want in RANADE.items():
            if w not in urls:
                log(f"    absent   Ranade {w}: Fontshare's API did not list it")
                ok = False
                continue
            data = opener.open(urls[w], timeout=timeout).read()
            tmp = ranade_path(w) + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(data)
            if sha256(tmp) != want:
                os.remove(tmp)
                log(f"    absent   Ranade {w}: Fontshare served different bytes than the "
                    f"ones checked; the page uses Instrument Sans")
                ok = False
                continue
            os.replace(tmp, ranade_path(w))
        if ok:
            log("    ok       Ranade 400, 500, 700 from Fontshare (ITF Free Font License; "
                "never committed)")
        return ok and ranade_installed()
    except Exception as exc:                      # no network: the fallback font
        log(f"    absent   Ranade (could not reach Fontshare: {type(exc).__name__}); the "
            f"page uses Instrument Sans")
        return False


def font_css() -> str:
    """The @font-face rules for the fonts present on this machine."""
    rules = []
    if ranade_installed():
        for w in sorted(RANADE):
            rules.append("@font-face{font-family:'Ranade';src:url('/static/fonts/ranade/"
                         f"ranade-{w}.woff2') format('woff2');font-weight:{w};"
                         "font-style:normal;font-display:swap}")
    inst = os.path.join(STATIC, "fonts", "instrument-sans")
    for w in (400, 500, 600, 700):
        f = f"instrument-sans-latin-{w}-normal.woff2"
        if os.path.isfile(os.path.join(inst, f)):
            rules.append("@font-face{font-family:'Instrument Sans';src:url('/static/fonts/"
                         f"instrument-sans/{f}') format('woff2');font-weight:{w};"
                         "font-style:normal;font-display:swap}")
    jb = os.path.join(STATIC, "fonts", "jetbrains-mono")
    for w in (400, 600):
        for sub, rng in _JB:
            f = f"jetbrains-mono-{sub}-{w}-normal.woff2"
            if os.path.isfile(os.path.join(jb, f)):
                rules.append("@font-face{font-family:'JetBrains Mono';src:url('/static/fonts/"
                             f"jetbrains-mono/{f}') format('woff2');font-weight:{w};"
                             f"font-style:normal;font-display:swap;unicode-range:{rng}}}")
    return "\n".join(rules) + "\n"


__all__ = ["API", "RANADE", "RANADE_DIR", "fetch_ranade", "font_css", "ranade_installed"]
