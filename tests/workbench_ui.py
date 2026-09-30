"""Driving the Atlas Workbench's page from a test, without a browser.

Every control in the page is a Panel widget whose watcher or ``on_click`` is the
code a click in the browser runs, and the canvas's clicks arrive as Bokeh Tap
events at `GeometryEditor._on_tap`.  These helpers find a widget by its label
under a layout, read the text a layout shows, click, and tap the canvas.  What
they cannot see is whether the page draws it: that is checked by clicking in a
served page (the project's rule since the RaceLab control that did nothing).
"""

from __future__ import annotations

from types import SimpleNamespace

import panel as pn


def walk(layout):
    """Every object under a Panel layout, the layout first, depth first."""
    stack = [layout]
    while stack:
        o = stack.pop(0)
        yield o
        stack[0:0] = list(getattr(o, "objects", None) or [])


def texts(layout) -> str:
    """Every label, text, title and option a layout holds (a repr of a Panel layout
    shows neither widget names nor text)."""
    out: list[str] = []
    for o in walk(layout):
        for attr in ("name", "object", "title"):
            v = getattr(o, attr, None)
            if isinstance(v, str):
                out.append(v)
        opts = getattr(o, "options", None)
        if isinstance(opts, (dict, list)):
            out.extend(str(k) for k in opts)
    return "\n".join(out)


def widgets(layout, kind=pn.widgets.Widget, name: str | None = None,
            prefix: str | None = None) -> list:
    """The widgets of ``kind`` under ``layout``, by label or its start."""
    return [o for o in walk(layout) if isinstance(o, kind)
            and (name is None or o.name == name)
            and (prefix is None or str(o.name).startswith(prefix))]


def widget(layout, name: str, kind=pn.widgets.Widget):
    """The one widget labelled ``name`` under ``layout``."""
    found = widgets(layout, kind, name=name)
    assert len(found) == 1, (name, [o.name for o in widgets(layout)])
    return found[0]


def click(button) -> None:
    """A click: what the page sends is the button's click count going up."""
    button.clicks += 1


def tap(ed, x: float, y: float) -> None:
    """A click on the canvas at ``(x, y)``, in cells."""
    ed._on_tap(SimpleNamespace(x=x, y=y))


def double_tap(ed, x: float, y: float) -> None:
    """A double click: the page sends its own tap first, then the double tap."""
    ev = SimpleNamespace(x=x, y=y)
    ed._on_tap(ev)
    ed._on_double_tap(ev)
