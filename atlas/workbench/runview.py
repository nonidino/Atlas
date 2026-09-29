"""The Run & compare and Results steps: what a run draws while it marches, and after.

Everything here is drawn from a `runner.CaseRun` -- its live `Progress` while it
marches and its record when it is done -- so the page cannot show a number the
record does not hold.  Rebuilt whenever the step is shown; while the step is
showing, `RunPanel.update` is called by the page's periodic callback.
"""

from __future__ import annotations

import html
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import (ColorBar, ColumnDataSource, LinearColorMapper, Range1d)
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

from .runner import ARM_LABELS, CaseRun

if TYPE_CHECKING:                                             # pragma: no cover
    from .app import Workbench

ARM_COLOURS = {"serial": "#6b7280", "parallel": "#0e6874", "full": "#b45309"}


def _diverging(n: int = 101) -> tuple[str, ...]:
    """Blue through white to red, for a difference centred on zero."""
    lo, mid, hi = (33, 102, 172), (247, 247, 247), (178, 24, 43)
    out = []
    for i in range(n):
        t = i / (n - 1)
        a, b, s = (lo, mid, 2 * t) if t < 0.5 else (mid, hi, 2 * (t - 0.5))
        out.append("#%02x%02x%02x" % tuple(int(round(a[k] + (b[k] - a[k]) * s))
                                           for k in range(3)))
    return tuple(out)


DIVERGING = _diverging()


def fmt(x: Any, spec: str = ".3g") -> str:
    if x is None:
        return "-"
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "-"
        return format(float(x), spec)
    return str(x)


VERDICT_COLOURS = {"pass": "#15803d", "FAIL": "#b91c1c", "not measured": "#6b7280"}


def html_table(df: pd.DataFrame, wrap: tuple[str, ...] = ()) -> str:
    """A plain table that wraps its text, for results that do not change once drawn.

    Tabulator gives each column the width of its longest cell and scrolls
    sideways, which hid the last columns of the results on a 1280-pixel screen;
    a table of a handful of rows reads better as HTML."""
    if df.empty:
        return "<i>nothing yet</i>"
    th = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = html.escape(str(r[c]))
            if c == "verdict":
                v = (f"<b style='color:{VERDICT_COLOURS.get(str(r[c]), '#111827')}'>"
                     f"{v}</b>")
            cls = " class='wrap'" if c in wrap else ""
            tds.append(f"<td{cls}>{v}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return ("<table class='wb-table'><thead><tr>" + th + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


TABLE_CSS = """
table.wb-table { border-collapse: collapse; font-size: 13px; width: 100%; }
table.wb-table th { text-align: left; background: #f1f5f9; color: #1f2937;
                    padding: 5px 10px; border-bottom: 1px solid #cbd5e1; font-weight: 600; }
table.wb-table td { padding: 4px 10px; border-bottom: 1px solid #e2e8f0; white-space: nowrap;
                    vertical-align: top; }
table.wb-table td.wrap { white-space: normal; min-width: 180px; }
"""


def table_pane(df: pd.DataFrame, wrap: tuple[str, ...] = ()) -> pn.pane.HTML:
    return pn.pane.HTML(html_table(df, wrap), sizing_mode="stretch_width",
                        stylesheets=[TABLE_CSS], margin=(0, 10))


class RunPanel:
    """The live half of the Run & compare step, for one run (or none yet)."""

    def __init__(self, wb: "Workbench", run: CaseRun | None):
        self.wb = wb
        self.run = run
        spec = run.spec if run is not None else wb.spec
        d = spec.domain
        self.module = run.module if run is not None else None
        unit = getattr(self.module, "LENGTH_UNIT", "") if self.module else ""
        lx, ly = d.nx * d.dx, d.ny * d.dx
        self.extent = (lx, ly)
        self.xr, self.yr = Range1d(0, lx), Range1d(0, ly)
        self.field_mapper = LinearColorMapper(palette=Viridis256, low=0.0, high=1.0)
        self.diff_mapper = LinearColorMapper(palette=DIVERGING, low=-1.0, high=1.0)
        label = getattr(self.module, "FIELD_LABEL", "field") if self.module else "field"
        titles = {"decomposed": f"Decomposed: {label}", "full": f"Full domain: {label}",
                  "difference": "Decomposed minus full"}
        self.src = {k: ColumnDataSource(dict(image=[np.zeros((2, 2), np.float32)], x=[0.0],
                                             y=[0.0], dw=[lx], dh=[ly]))
                    for k in titles}
        self.figs = {}
        aspect = ly / max(lx, 1e-12)
        w = 300
        h = int(min(max(w * aspect, 110), 330)) + 60
        for k, t in titles.items():
            p = figure(title=t, width=w, height=h, x_range=self.xr, y_range=self.yr,
                       tools="pan,wheel_zoom,reset,save", toolbar_location="above",
                       x_axis_label=f"x ({unit})" if unit else "x",
                       y_axis_label=f"y ({unit})" if unit else "y")
            mapper = self.diff_mapper if k == "difference" else self.field_mapper
            p.image(image="image", x="x", y="y", dw="dw", dh="dh", source=self.src[k],
                    color_mapper=mapper)
            p.add_layout(ColorBar(color_mapper=mapper, width=8, padding=2), "right")
            p.toolbar.logo = None
            p.title.text_font_size = "12px"
            self.figs[k] = p
        series = dict(getattr(self.module, "SERIES", {}) or {}) if self.module else {}
        self.series_key = next(iter(series), None)
        title, _, ylabel = series.get(self.series_key, "per-step metric").partition("|")
        self.series_fig = figure(title=title.strip(), width=440, height=250,
                                 tools="pan,wheel_zoom,reset,save", toolbar_location="above",
                                 x_axis_label=run.step_label if run is not None else "step",
                                 y_axis_label=ylabel.strip())
        self.series_fig.toolbar.logo = None
        self.series_fig.title.text_font_size = "12px"
        self.series_src = {}
        arms = run.arms if run is not None else ()
        for a in arms:
            s = ColumnDataSource(dict(step=[], value=[]))
            self.series_fig.line("step", "value", source=s, color=ARM_COLOURS[a],
                                 line_width=2, legend_label=ARM_LABELS[a])
            self.series_src[a] = s
        if arms:
            self.series_fig.legend.location = "top_right"
            self.series_fig.legend.label_text_font_size = "11px"
        #: styles B, C and D iterate inside every step: their latest iteration
        #: history, on a log scale (the plan's "the convergence curve is itself a
        #: display")
        self.conv_fig = None
        self.conv_src = {}
        style = run.spec.coupling.style if run is not None else ""
        if run is not None and style in ("B", "C", "D"):
            self.conv_fig = figure(title="Convergence in the last step (update over the "
                                         "field's scale)",
                                   width=440, height=250, y_axis_type="log",
                                   tools="pan,wheel_zoom,reset,save", toolbar_location="above",
                                   x_axis_label="iteration")
            self.conv_fig.toolbar.logo = None
            self.conv_fig.title.text_font_size = "12px"
            for a in run.arms:
                if a == "full":
                    continue
                s = ColumnDataSource(dict(it=[], value=[]))
                self.conv_fig.line("it", "value", source=s, color=ARM_COLOURS[a],
                                   line_width=2, legend_label=ARM_LABELS[a])
                self.conv_fig.scatter("it", "value", source=s, color=ARM_COLOURS[a], size=4)
                self.conv_src[a] = s
            if self.conv_src:
                self.conv_fig.legend.location = "top_right"
                self.conv_fig.legend.label_text_font_size = "11px"
        self.table = table_pane(pd.DataFrame())
        self.progress = pn.indicators.Progress(value=0, max=100, width=260, active=False)
        self.status = pn.pane.HTML(sizing_mode="stretch_width", margin=(0, 10),
                                   styles={"font-size": "13px"})
        self._field_version = -1
        self._n_series = -1
        if run is not None:
            self.update()

    # -- live ------------------------------------------------------------------

    def update(self) -> None:
        run = self.run
        if run is None:
            return
        p = run.progress()
        pct = 100.0 * p.step / max(p.steps, 1)
        self.progress.value = int(round(pct))
        self.progress.active = run.active
        self.status.object = self._status_html(run, p)
        if p.field_version != self._field_version and p.fields:
            self._draw_fields(p.fields)
            self._field_version = p.field_version
        n = sum(len(v) for v in p.seconds.values())
        if n != self._n_series:
            self._draw_series(p)
            self.table.object = html_table(timing_frame(run, p))
            self._n_series = n
            for a, src in self.conv_src.items():
                vals = [max(v, 1e-18) for v in p.convergence.get(a, [])]
                src.data = dict(it=list(range(1, len(vals) + 1)), value=vals)

    def _status_html(self, run: CaseRun, p) -> str:
        changed = run.committed_json != self.wb.spec.to_json()
        state = {"starting": "Starting", "running": "Marching", "stopping": "Stopping",
                 "done": "Finished", "stopped": "Stopped", "failed": "Failed",
                 "created": "Waiting"}.get(p.status, p.status)
        arm = f" &middot; now: {html.escape(ARM_LABELS.get(p.arm, p.arm))}" if run.active else ""
        out = (f"<b>{state}</b> {html.escape(run.label())} &middot; "
               f"{html.escape(run.step_label)} <b>{p.step}</b> of {p.steps}{arm}<br>"
               f"<span style='opacity:0.8'>{html.escape(p.message)}</span>")
        if p.error:
            out += f"<br><b style='color:#b91c1c'>{html.escape(p.error)}</b>"
        if changed:
            out += ("<br><span style='color:#92400e'>The case has changed since this run "
                    "was committed; the run marches the committed copy. Run again to march "
                    "the new version.</span>")
        return out

    def _draw_fields(self, fields: dict[str, np.ndarray]) -> None:
        lx, ly = self.extent
        dec = "parallel" if "parallel" in fields else ("serial" if "serial" in fields else None)
        shown = {"decomposed": fields.get(dec) if dec else None, "full": fields.get("full"),
                 "difference": fields.get("difference")}
        vals = [f for k, f in shown.items() if f is not None and k != "difference"]
        if vals:
            lo = float(min(np.nanmin(f) for f in vals))
            hi = float(max(np.nanmax(f) for f in vals))
            if hi - lo < 1e-12:
                hi = lo + 1e-12
            self.field_mapper.low, self.field_mapper.high = lo, hi
        if shown["difference"] is not None:
            m = float(np.nanmax(np.abs(shown["difference"])))
            m = max(m, 1e-15)
            self.diff_mapper.low, self.diff_mapper.high = -m, m
        for k, f in shown.items():
            if f is None:
                continue
            self.src[k].data = dict(image=[f], x=[0.0], y=[0.0], dw=[lx], dh=[ly])
        self.figs["difference"].title.text = (
            "Decomposed minus full" + (
                f" (max |d| {fmt(float(np.nanmax(np.abs(shown['difference']))), '.2g')})"
                if shown["difference"] is not None else ""))

    def _draw_series(self, p) -> None:
        if self.series_key is None:
            return
        for a, src in self.series_src.items():
            vals = p.series.get(a, {}).get(self.series_key, [])
            src.data = dict(step=list(range(1, len(vals) + 1)), value=vals)

    # -- layout ------------------------------------------------------------------

    def fields_view(self):
        return pn.FlexBox(*[pn.pane.Bokeh(f) for f in self.figs.values()],
                          flex_wrap="wrap", gap="6px 10px", sizing_mode="stretch_width")

    def live_view(self):
        return pn.Column(
            pn.Row(self.progress, self.status, sizing_mode="stretch_width"),
            self.fields_view(),
            pn.FlexBox(pn.Column(pn.pane.Markdown(
                f"**Time per {html.escape(self.run.step_label)}, each arm** (the step "
                "alone; t_full / t_arm above 1 is faster than the full domain)",
                margin=(0, 10)), self.table, width=520),
                *([pn.pane.Bokeh(self.conv_fig)] if self.conv_fig is not None else []),
                # a steady run's repeats all give the same answer, and autoscaled to
                # a round-off spread that plot drew two equal answers as two lines
                *([] if self.run.step_label == "timed repeat"
                  else [pn.pane.Bokeh(self.series_fig)]),
                flex_wrap="wrap", gap="6px 16px", sizing_mode="stretch_width"),
            sizing_mode="stretch_width")


def timing_frame(run: CaseRun, p) -> pd.DataFrame:
    """The live table, in the run's own unit: macro-steps for a march, timed repeats
    for a steady solve; and the work inside one -- the wind farm's sub-steps, or the
    iterations of an iterated style."""
    rows = []
    full = p.seconds.get("full") or []
    t_full = float(np.mean(full)) if full else None
    unit = run.step_label
    work = "substeps" if any("substeps" in p.series.get(a, {}) for a in run.arms) \
        else "iterations"
    for a in run.arms:
        s = p.seconds.get(a) or []
        mean = float(np.mean(s)) if s else None
        w = (p.series.get(a, {}).get(work) or [None])[-1]
        rows.append({"arm": ARM_LABELS[a], f"{unit}s": len(s),
                     f"mean s per {unit}": fmt(mean, ".4g"),
                     "median": fmt(float(np.median(s)) if s else None, ".4g"),
                     "t_full / t_arm": fmt(t_full / mean if (t_full and mean) else None,
                                           ".3g"),
                     ("sub-steps (last)" if work == "substeps" else "iterations (last)"):
                         fmt(w, ".0f")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


def results_frames(res: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The results table and the checks table, straight from a run's record."""
    timing, metrics = res["timing"], res.get("metrics", {})
    rows = []
    for a in res["arms"]:
        t = timing.get(a, {})
        m = metrics.get(a, {})
        row = {"arm": ARM_LABELS[a],
               f"seconds per {res.get('step_label', 'macro-step')}": fmt(t.get("mean"),
                                                                          ".4g"),
               "median": fmt(t.get("median"), ".4g"),
               "t_full / t_arm (measured now)": fmt(t.get("speedup_vs_full"), ".3g")}
        for key, label, spec in METRIC_COLUMNS.get(res["family"], ()):
            row[label] = fmt(m.get(key), spec)
        rows.append(row)
    checks = pd.DataFrame([{"check": c["title"], "measured": fmt(c["value"], ".3g"),
                            "tolerance": ("exact" if c["tolerance"] is None
                                          else fmt(c["tolerance"], ".3g")),
                            "verdict": c["verdict"], "detail": c.get("detail", ""),
                            "registered": c["registered"]}
                           for c in res.get("checks", [])])
    return pd.DataFrame(rows), checks


#: family -> (metric key, column title, format) for the results table
METRIC_COLUMNS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "incompressible-2d": (("farm_power", "farm power (last 5 steps)", ".5g"),
                          ("farm_power_vs_full", "farm power vs full", "+.2%"),
                          ("rms_velocity_difference", "rms velocity difference", ".3g")),
    "conduction-2d": (("heat_in", "heat in (W per m of depth)", ".7g"),
                      ("heat_in_vs_full", "heat in vs full", "+.2e"),
                      ("heat_in_vs_closed_form", "vs closed form", "+.2e"),
                      ("max_temperature_difference_K", "max |T - T_full| (K)", ".3g"),
                      ("iterations_last", "iterations (last step)", ".0f")),
    "electric-2d": (("current_A", "circuit current (A)", ".7g"),
                    ("current_vs_full", "current vs full", "+.2e"),
                    ("plate_resistance_ohm", "plate resistance (ohm)", ".5g"),
                    ("plate_heat_W", "heat in the plate (W)", ".5g"),
                    ("iterations_last", "iterations", ".0f")),
}


def machine_line(res: dict[str, Any]) -> str:
    m = res.get("machine", {})
    pw = m.get("power", {})
    ac = {True: "on AC power", False: "**on battery**", None: "power source not read"}[pw.get("ac")]
    others = m.get("other_python_processes", []) or []
    names = []
    for o in others:
        cmd = o.get("command", "")
        tail = os.path.basename(cmd.split()[-1].strip('"')) if cmd else ""
        names.append(f"pid {o.get('pid')}" + (f" ({tail})" if tail else ""))
    after = res.get("machine_after", {}).get("power", {})
    changed = (" (the power source changed during the run)"
               if after.get("ac") is not None and after.get("ac") != pw.get("ac") else "")
    return (f"Measured on **{m.get('host', '?')}** ({m.get('cpu_count', '?')} logical cores), "
            f"{ac}{changed}; other Python processes at the start: "
            + (", ".join(names) if names else "none") + ".")


def results_view(wb: "Workbench", res: dict[str, Any] | None):
    if res is None:
        return pn.pane.Alert("No run has finished in this session yet. Run the case from "
                             "**5. Run & compare** (or the Run menu); its results appear here "
                             "and are saved beside the case file.", alert_type="info",
                             sizing_mode="stretch_width")
    table, checks = results_frames(res)
    n_fail = sum(1 for c in res.get("checks", []) if c["passed"] is False)
    n_na = sum(1 for c in res.get("checks", []) if c["passed"] is None)
    unit = res.get("step_label", "macro-step")
    style = (res.get("case") or {}).get("coupling", {}).get("style") or res.get("style", "?")
    head = (f"**{html.escape(res['case_name'])}**, committed at {res['committed_at'][11:19]}, "
            f"finished {res['finished'][11:19]} after **{res['steps_done']}** of "
            f"{res['steps_requested']} {html.escape(unit)}s"
            + (" (**stopped by the user**)" if res.get("stopped") else "")
            + f", {res['threads']} thread{'s' * (res['threads'] != 1)}, style "
            f"{html.escape(str(style))}. " + machine_line(res))
    verdict = (pn.pane.Alert(f"**{n_fail} check{'s' * (n_fail != 1)} failed.** Each failure "
                             "is shown against the tolerance registered before any run; "
                             "no tolerance is changed after the fact.", alert_type="danger",
                             sizing_mode="stretch_width") if n_fail else
               pn.pane.Alert("Every check that could be measured passed its registered "
                             "tolerance." + (f" {n_na} {'was' if n_na == 1 else 'were'} not "
                                             "measured in this run; the table says why for "
                                             "each." if n_na else ""),
                             alert_type="success", sizing_mode="stretch_width"))
    notes = [f"- {n}" for n in res.get("notes", [])]
    pou = res.get("problem", {}).get("partition_of_unity")
    if pou:
        notes.append(f"- Partition of unity certified by `GridPartitionOfUnity`: identity "
                     f"residual {fmt(pou['identity_residual'])} (tolerance "
                     f"{fmt(pou['tolerance'])}), smallest weight {fmt(pou['chi_min'])}.")
    notes.append(f"- {res.get('timing_note', '')}.")
    rec = res.get("record_path")
    if rec:
        notes.append(f"- Record: `{html.escape(os.path.relpath(rec, wb_root()))}` "
                     f"(the case as it marched, every per-step time, the machine's state).")
    return pn.Column(
        pn.pane.Markdown(head, sizing_mode="stretch_width", margin=(0, 10)),
        verdict,
        pn.pane.Markdown("**Speed and accuracy, each arm against the full domain** "
                         "(t_full / t_arm above 1 is faster than the full domain)",
                         margin=(4, 10, 0, 10)),
        table_pane(table),
        pn.pane.Markdown("**Sanity checks, each against the tolerance registered before any "
                         "run**", margin=(8, 10, 0, 10)),
        table_pane(checks, wrap=("detail", "registered")),
        pn.pane.Markdown("\n".join(notes), sizing_mode="stretch_width", margin=(6, 10)),
        sizing_mode="stretch_width")


def wb_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


__all__ = ["RunPanel", "results_view", "results_frames", "timing_frame", "machine_line",
           "METRIC_COLUMNS", "DIVERGING"]
