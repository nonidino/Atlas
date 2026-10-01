"""The Run & compare and Results steps: what a run draws while it marches, and after.

Everything here is drawn from a `runner.CaseRun` -- its live `Progress` while it
marches and its record when it is done -- so the page cannot show a number the
record does not hold.  Rebuilt whenever the step is shown; while the step is
showing, `RunPanel.update` is called by the page's periodic callback.
"""

from __future__ import annotations

import html
import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import (ColorBar, ColumnDataSource, LinearColorMapper, LogColorMapper,
                          Range1d)
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

from . import registry
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
        #: a family whose field spans decades (a plume: the outfall's cell against
        #: the mixed river) is drawn on a log scale over FIELD_DECADES below its peak;
        #: display only -- the difference panel and every number stay linear
        self.log_scale = bool(self.module is not None
                              and getattr(self.module, "FIELD_SCALE", "linear") == "log")
        self.decades = float(getattr(self.module, "FIELD_DECADES", 4.0)) if self.module else 4.0
        self.field_mapper = (LogColorMapper(palette=Viridis256, low=1e-4, high=1.0)
                             if self.log_scale else
                             LinearColorMapper(palette=Viridis256, low=0.0, high=1.0))
        self.diff_mapper = LinearColorMapper(palette=DIVERGING, low=-1.0, high=1.0)
        self.labels = run.arm_labels if run is not None else dict(ARM_LABELS)
        label = getattr(self.module, "FIELD_LABEL", "field") if self.module else "field"
        if self.log_scale:                        # short: the figure is 300 px wide
            label = label[:-1] + ", log)" if label.endswith(")") else label + " (log)"
        #: what the three panels are called: a family may name its own per arm (a
        #: split by physics has no "decomposed" arm: its panels are whichever split
        #: is shown and the unsplit solver); the first panel shows the parallel arm
        #: when it runs, else the serial one
        custom = dict(getattr(self.module, "PANEL_NAMES", None) or {}) if self.module else {}
        dec_arm = "parallel" if (run is not None and "parallel" in run.arms) else "serial"
        dec_name = custom.get(dec_arm, "Decomposed")
        full_name = custom.get("full", "Full domain")
        self.difference_name = (custom.get(f"difference_{dec_arm}",
                                           f"{dec_name} minus {full_name.lower()}")
                                if custom else "Decomposed minus full")
        titles = {"decomposed": f"{dec_name}: {label}", "full": f"{full_name}: {label}",
                  "difference": self.difference_name}
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
                                 line_width=2, legend_label=self.labels[a])
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
        explicit = False
        if run is not None:
            try:
                explicit = registry.family(run.spec.physics.family).explicit_coupling
            except KeyError:
                explicit = False
        if run is not None and style in ("B", "C", "D") and not explicit:
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
                                   line_width=2, legend_label=self.labels[a])
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
        arm = (f" &middot; now: {html.escape(run.arm_labels.get(p.arm, p.arm))}"
               if run.active else "")
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
            if self.log_scale:
                hi = max(hi, 1e-300)
                lo = hi * 10.0 ** (-self.decades)
                #: the image is clipped to the scale's floor, so a zero (clean water)
                #: is drawn as the floor's colour rather than as nothing
                shown = {k: (np.clip(f, lo, hi) if (f is not None and k != "difference")
                             else f) for k, f in shown.items()}
            elif hi - lo < 1e-12:
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
            self.difference_name + (
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

    def live_view(self, progress: bool = True):
        return pn.Column(
            *([pn.Row(self.progress, self.status, sizing_mode="stretch_width")]
              if progress else []),
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
    #: the work inside one step: the wind farm's sub-steps, an iterated style's
    #: iterations, or nothing (an explicit step does not iterate)
    work = next((k for k in ("substeps", "iterations")
                 if any(k in p.series.get(a, {}) for a in run.arms)), None)
    for a in run.arms:
        s = p.seconds.get(a) or []
        mean = float(np.mean(s)) if s else None
        row = {"arm": run.arm_labels[a], f"{unit}s": len(s),
               f"mean s per {unit}": fmt(mean, ".4g"),
               "median": fmt(float(np.median(s)) if s else None, ".4g"),
               "t_full / t_arm": fmt(t_full / mean if (t_full and mean) else None, ".3g")}
        if work is not None:
            w = (p.series.get(a, {}).get(work) or [None])[-1]
            row["sub-steps (last)" if work == "substeps" else "iterations (last)"] = \
                fmt(w, ".0f")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


def results_frames(res: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The results table and the checks table, straight from a run's record."""
    timing, metrics = res["timing"], res.get("metrics", {})
    labels = {**ARM_LABELS, **(res.get("arm_labels") or {})}
    rows = []
    for a in res["arms"]:
        t = timing.get(a, {})
        m = metrics.get(a, {})
        row = {"arm": labels[a],
               f"seconds per {res.get('step_label', 'macro-step')}": fmt(t.get("mean"),
                                                                          ".4g"),
               "median": fmt(t.get("median"), ".4g"),
               "t_full / t_arm (measured now)": fmt(t.get("speedup_vs_full"), ".3g")}
        for key, label, spec in METRIC_COLUMNS.get(res["family"], ()):
            row[label] = fmt(m.get(key), spec)
        #: every family: the displayed field's rms difference from the full domain
        fd = (res.get("field_difference") or {}).get(a)
        col = f"rms difference vs full, {res.get('field_label', 'field')}"
        row[col] = ("-" if a == "full" else fmt(fd["rms"], ".3g") if fd
                    else "no full domain" if "full" not in res["arms"] else "-")
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
                          ("rms_velocity_difference", "rms velocity difference (u and v)",
                           ".3g")),
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
    "conjugate-heat-2d": (("carried_out_W", "heat the coolant carries out (W/m)", ".7g"),
                          ("bulk_rise_K", "coolant's bulk rise (K)", ".5g"),
                          ("max_temperature_K", "hottest point (K)", ".6g"),
                          ("max_temperature_difference_K", "max |T - T_full| (K)", ".3g"),
                          ("iterations_last", "iterations", ".0f")),
    "thermoelastic-2d": (("max_stress_MPa", "largest von Mises stress (MPa)", ".6g"),
                         ("mean_temperature_K", "mean temperature (K)", ".6g"),
                         ("lag_stress_error", "lag error in stress", ".3g"),
                         ("balance_max", "heat balance", ".2e")),
    "elasticity-2d": (("max_displacement_mm", "largest displacement (mm)", ".6g"),
                      ("displacement_vs_full", "displacement vs full", ".2e"),
                      ("max_stress_MPa", "largest von Mises stress (MPa)", ".5g"),
                      ("iterations_last", "iterations", ".0f")),
    "acoustics-2d": (("reflection", "reflection R (measured)", ".9f"),
                     ("reflection_vs_textbook", "R - textbook", "+.2e"),
                     ("energy_second", "energy share in the second medium", ".4g"),
                     ("energy_drift", "energy drift", ".2e")),
    "transport-2d": (("mass_g", "pollutant in the river (g)", ".7g"),
                     ("outflow_g_per_s", "leaving at the outlet (g/s)", ".4g"),
                     ("mass_vs_full", "mass vs full", "+.2e"),
                     ("max_concentration_difference", "max |c - c_full| (mg/L)", ".3g")),
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


CARD_STYLE = {"background": "#ffffff", "border": "1px solid #e2e8f0", "border-radius": "12px",
              "padding": "4px 6px"}


def _card(title: str, value: str, detail: str, verdict: str | None = None,
          good: bool | None = None) -> pn.pane.HTML:
    """One result card: what it is, the measured value, and what that means."""
    colour = {True: "#15803d", False: "#b91c1c", None: "#5b6b7c"}[good]
    tag = (f"<span style='float:right;font-size:12px;font-weight:700;color:{colour}'>"
           f"{html.escape(verdict)}</span>" if verdict else "")
    return pn.pane.HTML(
        f"<div style='font-size:12px;font-weight:600;letter-spacing:0.4px;text-transform:"
        f"uppercase;color:#5b6b7c'>{html.escape(title)}{tag}</div>"
        f"<div style='font-size:24px;font-weight:700;color:#0f172a;margin:6px 0 4px 0;"
        f"font-family:ui-monospace,Consolas,monospace'>{html.escape(value)}</div>"
        f"<div style='font-size:13px;color:#334155;line-height:1.4'>{html.escape(detail)}</div>",
        styles=CARD_STYLE, sizing_mode="stretch_width", margin=(0, 6), min_width=220)


def summary_cards(res: dict[str, Any]) -> list:
    """Speed, agreement, the balance and the checks, from a run's record: the four
    numbers a person reads first (the tables are folded under Details)."""
    checks = res.get("checks", [])
    timing = res.get("timing", {})
    labels = {**ARM_LABELS, **(res.get("arm_labels") or {})}
    cards = []
    ratios = {a: t.get("speedup_vs_full") for a, t in timing.items()
              if a != "full" and t.get("speedup_vs_full")}
    if ratios:
        best = max(ratios, key=lambda a: ratios[a])
        cards.append(_card("Speed", f"{ratios[best]:.3g}×",
                           f"the whole domain's time divided by the {labels[best].lower()} "
                           f"arm's; above 1 is faster than the whole domain"))
    else:
        cards.append(_card("Speed", "-", "needs a decomposed arm and the whole domain"))
    by_kind = {k: [c for c in checks if c.get("kind") == k] for k in ("reference", "balance")}
    for kind, title in (("reference", "Agreement"), ("balance", "Balance")):
        cs = [c for c in by_kind[kind] if c["passed"] is not None] or by_kind[kind]
        if cs:
            c = cs[0]
            cards.append(_card(title, fmt(c["value"], ".2e"),
                               f"{c['title']}; limit {fmt(c['tolerance'], '.0e')}",
                               c["verdict"], c["passed"]))
    n_pass = sum(1 for c in checks if c["passed"] is True)
    n_fail = sum(1 for c in checks if c["passed"] is False)
    n_na = sum(1 for c in checks if c["passed"] is None)
    detail = (f"{n_fail} failed against the tolerance registered before any run"
              if n_fail else "every check that could be measured passed")
    if n_na:
        detail += f"; {n_na} not measured (Details says why)"
    cards.append(_card("Checks", f"{n_pass} of {len(checks)} passed", detail,
                       "FAIL" if n_fail else "pass", not n_fail))
    return cards


def results_view(wb: "Workbench", res: dict[str, Any] | None):
    if res is None:
        return pn.pane.Alert("No run has finished in this session yet. Press **Run**: the "
                             "results appear here and are saved beside the case file.",
                             alert_type="info", sizing_mode="stretch_width")
    table, checks = results_frames(res)
    unit = res.get("step_label", "macro-step")
    style = (res.get("case") or {}).get("coupling", {}).get("style") or res.get("style", "?")
    #: which case: its kind, not a name (a case has none since case@0.6)
    kind = html.escape(registry.short_label(res.get("family", "")))
    head = (f"**The {kind} case**, as it was at {res['committed_at'][11:19]}: "
            f"finished at {res['finished'][11:19]} after **{res['steps_done']}** of "
            f"{res['steps_requested']} {html.escape(unit)}s"
            + (" (**stopped**)" if res.get("stopped") else "")
            + f", {res['threads']} thread{'s' * (res['threads'] != 1)}, style "
            f"{html.escape(str(style))}. " + machine_line(res))
    #: the last run is of the case as it committed it; once the open case differs
    #: (an edit, another example), these are not the open case's results
    stale = res.get("case") is not None and res["case"] != json.loads(wb.spec.to_json())
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
    changed = ([pn.pane.Alert(f"These are the results of **the {kind} case** as the run took "
                              "it; the case open now differs. Run it again for its own "
                              "results.",
                              alert_type="warning", sizing_mode="stretch_width")]
               if stale else [])
    #: a river that leaves by several outlets: each one's share, where it is read first
    #: (the owner's forked river, 2026-09-30)
    metrics = res.get("metrics") or {}
    outlets = (metrics.get("full") or next(iter(metrics.values()), {}) or {}).get("outlets")
    if outlets and len(outlets) > 1:
        parts = []
        for bid, s in outlets.items():
            w = s.get("water_share")
            p = s.get("pollutant_share")
            parts.append(f"**{html.escape(str(bid))}** takes "
                         f"{'-' if w is None else f'{100.0 * w:.1f}%'} of the water and "
                         f"{'none yet' if p is None else f'{100.0 * p:.1f}%'} of the pollutant")
        changed.append(pn.pane.Markdown(f"The river leaves by {len(outlets)} outlets: "
                                        + "; ".join(parts) + ".", sizing_mode="stretch_width",
                                        margin=(0, 10, 4, 10)))
    details = pn.Card(
        pn.pane.Markdown("**Each arm against the whole domain** (t_full / t_arm above 1 is "
                         "faster than the whole domain)", margin=(4, 10, 0, 10)),
        table_pane(table),
        pn.pane.Markdown("**Every check, against the tolerance registered before any run**",
                         margin=(8, 10, 0, 10)),
        table_pane(checks, wrap=("detail", "registered")),
        pn.pane.Markdown("\n".join(notes), sizing_mode="stretch_width", margin=(6, 10)),
        title="Details: every arm, every check, the notes and the record", collapsed=True,
        sizing_mode="stretch_width", margin=(10, 6))
    return pn.Column(
        pn.pane.Markdown(head, sizing_mode="stretch_width", margin=(12, 10, 4, 10)),
        *changed,
        #: a Row shares the width among the cards; in a wrapping FlexBox each
        #: stretch_width card took a whole line (seen)
        pn.Row(*summary_cards(res), sizing_mode="stretch_width"),
        details,
        sizing_mode="stretch_width")


def run_controls(wb: "Workbench", run_btn, stop_btn, steps, threads, arms, label: str,
                 arm_notes: list[str], why: list[str]):
    """The top of the Run tab: Run and Stop, what runs, and the settings folded away."""
    run = wb.run
    if run is not None and run.active:
        # no step count here: this line is built once, and read "0 of 6000" while
        # the run was at 842 (seen); the live count is in the progress row below
        state = (f"<b>Marching</b> {run.steps:,} {html.escape(run.step_label)}s: the "
                 f"progress is below, and Stop ends the run at a whole step")
    elif run is not None:
        state = f"<b>Last run</b>: {html.escape(run.label())}"
    else:
        state = ("Runs the windows one after another, on threads, and the whole domain at "
                 "once, in turns, and compares them.")
    settings = pn.Card(pn.Row(steps, threads, sizing_mode="stretch_width"),
                       pn.pane.Markdown("**Arms**", margin=(4, 10, 0, 10)), arms,
                       *[pn.pane.HTML(f"<small>{html.escape(n)}</small>", margin=(0, 10))
                         for n in arm_notes],
                       title=f"Settings: {wb.spec.run.steps} {label}s, "
                             f"{wb.spec.run.threads} threads", collapsed=True,
                       sizing_mode="stretch_width", margin=(6, 6))
    out = [pn.Row(run_btn, stop_btn,
                  pn.pane.HTML(state, sizing_mode="stretch_width", margin=(8, 12),
                               styles={"font-size": "14px"}),
                  sizing_mode="stretch_width")]
    if why:
        out.append(pn.pane.Alert(" ".join(why), alert_type="warning",
                                 sizing_mode="stretch_width"))
    out.append(settings)
    return pn.Column(*out, sizing_mode="stretch_width", styles=CARD_STYLE, margin=(0, 0, 10, 0))


def compile_view(wb: "Workbench"):
    """The Atlas compiler's verdict on the case, per seam, with a button to compile."""
    job = wb.compile_job
    busy = (job is not None and job.active) or (wb.run is not None and wb.run.active)
    n_err = sum(1 for i in wb.issues() if i.severity == "error")
    btn = pn.widgets.Button(name="Compile", button_type="primary", button_style="outline",
                            width=110, disabled=bool(n_err) or busy)
    btn.on_click(lambda e: wb.dispatch("run:compile"))
    #: a compile of the case open before this one is not this case's verdict (the
    #: gallery pass found farm-21's shown beside farm-12)
    other = (job is not None and not job.active and job.status != "created"
             and getattr(wb, "compiled_serial", None) != getattr(wb, "case_serial", None))
    if job is None or other:
        text = ("<b>Atlas compiler</b>: "
                + ("not compiled in this session" if job is None else
                   f"not compiled since this case was opened (the last compile was of "
                   f"the {html.escape(registry.short_label(job.spec.physics.family))} "
                   f"case open before it)")
                + ". Compiling builds the case's graph from its family (an agent per "
                "window or piece, a seam wherever two meet) and gives each seam the "
                "compiler's verdict.")
        body: list = []
    elif job.active or job.status == "created":
        text = ("<b>Atlas compiler</b>: compiling the case (it probes every seam through "
                "its agents' own solves).")
        body = []
    elif job.status == "failed":
        text = f"<b>Atlas compiler</b>: the compile failed: {html.escape(job.error or '')}"
        body = []
    else:
        s = job.summary
        changed = job.committed_json != wb.spec.to_json()
        colour = {"admit": "#15803d", "admit-uncertified": "#b45309", "refuse": "#b91c1c"}
        text = (f"<b>Atlas compiler</b>: <b style='color:{colour.get(s.verdict, '#0f172a')}'>"
                f"{html.escape(s.verdict)}</b>, {len(s.seams)} "
                f"seam{'s' * (len(s.seams) != 1)} in {s.seconds:.2f} s"
                + (" (the case has changed since; compile again)" if changed else ""))
        body = [_compile_details(wb, job)]
    head = pn.Row(btn, pn.pane.HTML(text, sizing_mode="stretch_width", margin=(8, 12),
                                    styles={"font-size": "14px"}),
                  sizing_mode="stretch_width")
    return pn.Column(head, *body, sizing_mode="stretch_width", styles=CARD_STYLE,
                     margin=(10, 0, 20, 0))


def _compile_details(wb: "Workbench", job) -> pn.Card:
    s = job.summary

    def cut(msg: str, n: int) -> str:                  # the record keeps it whole
        return msg if len(msg) <= n else msg[:n].rstrip() + "..."
    body: list = []
    if s.refused_before:
        body.append(pn.pane.Markdown(f"**Refused before the compiler**, by the package's own "
                                     f"vocabulary: {html.escape(s.refused_before)}",
                                     sizing_mode="stretch_width", margin=(0, 10)))
    else:
        cps = ("none (declared, W162)" if not s.cross_points
               else ", ".join(s.cross_points) + " (declared, W162)")
        body.append(pn.pane.Markdown(f"{s.agents} agents, {len(s.seams)} seams; cross-points: "
                                     f"{html.escape(cps)}.", margin=(0, 10)))
        rows = [{"seam": sv.seam, "between": " and ".join(sv.between), "port": sv.port_type,
                 "verdict": sv.verdict,
                 "rules": "; ".join(f"{r['layer']}/{r['rule']} ({r['verdict']}): "
                                    f"{cut(r['message'], 220)}" for r in sv.rules)
                 or "every decision about it admits"} for sv in s.seams]
        body.append(table_pane(pd.DataFrame(rows), wrap=("rules",)))
        if s.other:
            other = [{"layer": r["layer"], "rule": r["rule"], "verdict": r["verdict"],
                      "about": r["subject"], "message": cut(r["message"], 300)}
                     for r in s.other]
            body.append(pn.pane.Markdown("**About the graph, its agents and the run**",
                                         margin=(4, 10, 0, 10)))
            body.append(table_pane(pd.DataFrame(other), wrap=("message",)))
        if any(r["rule"] == "R10" and r["layer"] == "L5" and r["verdict"] ==
               "admit-uncertified" and "direct-schur" in r["message"] for r in s.other):
            body.append(pn.pane.Markdown(
                "<small>R10 decertifies instead of refusing (W348): each window solves its "
                "piece of one physics directly, with its neighbours' values on its cut, "
                "and the compiled scheme solves the interface directly, so together the "
                "pieces are the whole-domain solution. Keeping each solve inside its "
                "piece costs sweeps, not accuracy; the run's agreement check measures how "
                "close the pieces come.</small>", sizing_mode="stretch_width",
                margin=(0, 10)))
    if job.record_path:
        body.append(pn.pane.Markdown(
            f"<small>Record: `{html.escape(os.path.relpath(job.record_path, wb_root()))}` "
            f"(every decision, the compiler's report, the case as compiled).</small>",
            margin=(0, 10)))
    return pn.Card(*body, title="Every seam and rule", collapsed=True,
                   sizing_mode="stretch_width", margin=(6, 6))


def wb_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


__all__ = ["RunPanel", "results_view", "results_frames", "timing_frame", "machine_line",
           "METRIC_COLUMNS", "DIVERGING", "summary_cards", "run_controls", "compile_view"]
