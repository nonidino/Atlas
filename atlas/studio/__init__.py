"""Atlas studio: one app to draw a domain, choose its physics, cut it into pieces,
run it and read the guarantees.

The integrated UI of ``wiki/concepts/Atlas 0.1/atlas-0.1-proposal/demo/
integrated-ui-plan.md``: a FastAPI server (`server.py`) and one plain JavaScript page
(``static/``), with every edit a WebSocket message (`protocol.py`) applied to a case
by `session.Session`.  The engine below it is the workbench's (``spec``, the eight
families, the compile, the runner) and the chart operator's (``atlas/chartop``),
called and never edited; registered evaluations hash some of those modules.

    python -m atlas.studio --open
"""

PORT = 8021

__all__ = ["PORT"]
