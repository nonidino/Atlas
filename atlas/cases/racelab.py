"""Tier 51 -- PoC 3 RaceLab, phase 1: the car's geometry, its windows, its graph.

`POC3-RACELAB-REQUIREMENTS.md` section 9 splits RaceLab into five phases.  This
is phase 1 and it is the whole of phase 1: **the car's bodies as immersed body
forces, a window decomposition DERIVED from those bodies, the assembled
`CaseGraph`, and a headless march at the declared vehicle scale.**  There is no
server, no dashboard, no expert switch and no third dimension here; those are
phases 2 to 5.

What is reused rather than re-implemented, and where
----------------------------------------------------

  ================================  ==================================
  `ground_effect.GroundTiling`      `RaceTiling` SUBCLASSES it, so the
                                    partition of unity, the halo, the
                                    cut, the assembly and the
                                    contamination mask are that class's
                                    and not a second copy.  What the
                                    subclass overrides is the LAYOUT:
                                    the offsets are handed in rather
                                    than derived from a row and column
                                    count, which is what lets them come
                                    from the car instead of from a grid.
  `wing_fsi.FlexWing`               every body in the car is one of
                                    these -- the porous inclined plate,
                                    its stations, its normal traction
                                    and its exactly-conservative
                                    stamping kernel, at a different
                                    ``x_le``, ``y_mount``, ``chord`` and
                                    ``alpha``.  `PlateBody` is a name
                                    for a `FlexWing` and a rigid zero
                                    deflection, nothing more.
  `wing_fsi.FSIRollout`             `RaceRollout` subclasses it for cut,
                                    blend, project and band.
  `vehicle_march`                   the vehicle scale, the host-sized
                                    rotor, the machine sized to it, the
                                    devices' body force and the coolant
                                    sub-cycling.
  `integration_union`               J1, J2 and J3 -- the core, the
                                    mount and the rotor -- unchanged.
  ================================  ==================================

The three things phase 1 adds
-----------------------------

1. **A car** (`CAR`, `Body`, `PlateBody`, `WheelBody`).  Eleven bodies on a
   moving ground, each a porous plate segment or a ring of them.
2. **A decomposition derived from it** (`windows_from_geometry`).  The cuts are
   placed where the car is not, by a dynamic program over the offsets, subject
   to two device planes having to land ON a cut because that is where
   `integration_union` declares the core's and the rotor's ports.  This is
   W124's discipline -- no seam through a body -- turned from a check into an
   objective.
3. **The union re-sited** (`build`, `RaceRollout`, `march`).  The same eighteen
   agents CS-18 marched, on the car's lattice instead of the front wing's.

What this file does NOT do, so it is not read as doing it
---------------------------------------------------------

The Reynolds number is the tiling's 250 and not a vehicle's 1.6e6
(`vehicle-scale-and-sizing` section 1.5), the ground is a rolling road with no
boundary layer, and the car is a two-dimensional centreline slice.  The car is
also SHORT: at the inherited ``L0 = 0.50`` m it is **3.35 m long and 0.78 m
tall** against an F1 car's 5.6 m and 0.95 m, so it is compressed in length by a
factor of 1.7 and in height by 1.2 -- a recognisable car rather than a
dimensionally faithful one.  A full-length car with a wake wants about 1024
cells, which stage `size` MEASURED at 0.219 s a macro-step against the chosen
box's 0.156 -- affordable, and rejected because it is 8.0 m of domain to hold
5.6 m of car and puts the window count at 18 with nothing gained.  `CAR_NOTES`
carries every such departure in one place and the case study repeats them.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from ..capability import MotionClass
from ..graph import (Agent, CaseGraph, Connection, Decomposition, FluxMatching,
                     GlobalField)
from ..ports import PortType
from . import cooling_loop as CL
from . import front_wing as FW
from . import powertrain as PT
from . import wake_array as WA
from . import ground_effect as GE
from . import integration_union as IU
from . import vehicle_march as VM
from . import wing_fsi as W
from .ground_effect import DX, GroundTiling

__all__ = [
    "RNX", "RNY", "WX", "WY", "HALO_MIN", "RAMP",
    "Body", "PlateBody", "WheelBody", "DeviceWindowSpec",
    "CAR", "CAR_NOTES", "car_bodies", "geometry_fingerprint", "body_extents",
    "CarParams",
    "RaceTiling", "windows_from_geometry", "LAYOUT", "layout", "layout_report",
    "layout_frontier", "body_profile", "force_profile", "PROFILE_KINDS",
    "DUCT_CORE_RANGE", "TURBINE_RANGE", "DUCT_Y0", "WING_BODY_ID",
    "HOST_ROTOR_WIDTH", "U_MAX_BAND", "DEVICE_CELLS",
    "RaceDeviceSpec", "RaceDeviceForcing", "device_sites", "ring_velocity",
    "RaceWindow", "make_experts", "connections", "build",
    "RaceRollout", "march", "GATE", "PREDICTION",
    "machine_for_host", "U_HOST_REF", "EnvelopeDeclined", "ENVELOPE_STATES",
    "settled_field", "N_SPIN",
]


# ===========================================================================
# the box, measured rather than guessed
# ===========================================================================

#: The domain, in cells at ``DX = 1/64``.  **Chosen from a measurement**, not
#: from the requirements' ``384x192`` guess: `scripts/tier51_racelab_graph.py`
#: stage ``size`` times a macro-step at seven candidate boxes in one process
#: with the front-wing case as the control in the same process, so what is
#: compared is a RATIO and not a wall-clock number that rots overnight.
#:
#: ``672 x 240`` is ``10.5 x 3.75`` length units and, at the inherited
#: ``L0 = 0.50`` m, ``5.25 x 1.88`` m.  The width is 672 and not 608 because the
#: radiator duct has to be long enough to hold TWO device planes a full stride
#: apart -- the core and the recovery turbine each need a cut of their own, and
#: a cut narrow enough that the ring a device reads its inflow on is close to
#: the plane its force sits in.  CS-18 measured that gap costing 4.4% of J3's
#: receiving balance at SEVEN cells; the first layout this tier searched put it
#: at forty, and the box grew rather than the error.
RNX, RNY = 672, 240

#: The window, in cells.  **128 and not the front wing's 80, and the reason is
#: phase 2.**  `wake_array`'s line 24: "The checkpoint is fixed at 128x128".
#: Poseidon-T is the learned expert the switch switches to, so a window that is
#: not 128 cells has to be resized before the checkpoint can see it, and a
#: resize is an error that is not the expert's -- it would be charged to the
#: learned column and the whole point of phase 2 is to price that column
#: honestly.  Measured, the price of 128 over 80 is 1.7x in the macro-step and
#: both are far inside the 0.5 s ceiling.
WX, WY = 128, 128

#: The smallest overlap between neighbouring windows, so the largest stride is
#: ``WX - HALO_MIN``.  16 cells is `ground_effect.HALO` unchanged -- the front
#: wing's own halo, and the one `wing_fsi`'s measured constants were taken at.
HALO_MIN = 16

#: The partition of unity's ramp, `ground_effect.RAMP` unchanged.
RAMP = GE.RAMP

#: **The wheel's normal-traction coefficient, DERIVED rather than inherited.**
#:
#: A wheel is a closed ring of porous plate segments, and giving each of them
#: the plate's own ``C_N = 20`` is not "the same model applied to a circle": it
#: is twelve independent screens in series, and measured at the release state
#: it produced 11.52 of drag per wheel against the 1.0 the whole rest of the
#: car produced, with a peak force density 16x the next largest body's.  The
#: march blew up at macro-step 2.
#:
#: So the coefficient is chosen to make the body's TOTAL drag a circular
#: cylinder's, which is the one number about a wheel this model can be asked to
#: get right.  For a segment whose outward normal is at angle ``theta``, the
#: relative normal velocity is ``U cos(theta)`` and the traction's streamwise
#: component is ``0.5 c_n U^2 |cos| cos^2``, so
#:
#:     drag = 0.5 c_n U^2 . integral |cos| cos^2 R dtheta = (4/3) c_n U^2 R
#:
#: against a cylinder's ``0.5 C_D U^2 (2R) = C_D U^2 R``.  Equating,
#: ``c_n = 3 C_D / 4``.  At ``C_D = 1.2`` -- a two-dimensional circular cylinder
#: in the subcritical range -- that is ``0.9``, and the discrete twelve-segment
#: ring reproduces the continuous integral to 1.3% (measured, and asserted in
#: the tests).
#:
#: **This is a calibration and it is named as one.**  It makes the wheel's drag
#: right and says nothing about its wake, its lift, or the rotation that
#: `WheelBody` records as invisible to a normal-only closure.
WHEEL_CD = 1.2
WHEEL_CN = 3.0 * WHEEL_CD / 4.0

#: The body whose wetted and mount seams pair with STRUCT and SUSP.  The car's
#: front wing IS CS-12's plate, at CS-12's chord and station count, re-sited.
WING_BODY_ID = "FW_MAIN"

#: The rotor's swept width, `vehicle_march.HOST_ROTOR_WIDTH` unchanged: the
#: disk is sized to its host face, which is decision 2 (W199) and not re-taken.
HOST_ROTOR_WIDTH = VM.HOST_ROTOR_WIDTH

#: **A stalled march is a blow-up.**  `WindowNS` sets its sub-step count from
#: ``u_max``, so a diverging field does not raise -- it takes more and more
#: sub-steps and reads as a hang.  `march` bails past this band with the
#: macro-step number.  Four free-stream speeds is far above anything a porous
#: body in this flow produces (the front-wing case settles near 1.5) and far
#: below the runaway, so it separates the two.
U_MAX_BAND = 4.0


# ===========================================================================
# the car
# ===========================================================================


@dataclass(frozen=True)
class Body:
    """One immersed body, declared in CELLS on the race lattice.

    Everything is a porous plate segment: a straight run of length ``chord``
    from ``(x_le, y_le)`` at angle ``alpha`` to the free stream, carrying the
    normal traction ``0.5 c_n |w| w`` at each of ``n_station`` stations and
    stamping it back onto the fluid with `wing_fsi.FlexWing`'s exactly
    conservative kernel.  A wheel is a closed ring of them (`WheelBody`).

    The units are the tiling's: ``x_le`` and ``y_le`` are CELLS and ``chord`` is
    CELLS, converted at construction, because a car is easier to draw on a
    lattice than in length units and the conversion in one place is safer than
    the conversion in thirty.
    """

    body_id: str
    x_le: float                 # cells
    y_le: float                 # cells
    chord: float                # cells
    alpha_deg: float
    c_n: float = GE.C_N
    n_station: int = 16
    group: str = "aero"
    label: str = ""
    #: bodies the layout is allowed to cut.  The duct walls are, because a
    #: device plane has to land on a cut and the core's plane is inside them.
    cuttable: bool = False

    @property
    def alpha(self) -> float:
        return math.radians(self.alpha_deg)

    @property
    def x_te(self) -> float:
        return self.x_le + self.chord * math.cos(self.alpha)

    @property
    def y_te(self) -> float:
        return self.y_le + self.chord * math.sin(self.alpha)

    def x_span(self) -> tuple[float, float]:
        return (min(self.x_le, self.x_te), max(self.x_le, self.x_te))

    def y_span(self) -> tuple[float, float]:
        return (min(self.y_le, self.y_te), max(self.y_le, self.y_te))


class PlateBody:
    """A `Body` and the `wing_fsi.FlexWing` that carries its physics.

    **No new force model.**  `FlexWing.forcing` is called with a zero deflection
    and a zero surface velocity, which is the rigid case of the machinery CS-12
    already marches, so the force on the fluid integrates to minus the force on
    the body to floating point on this lattice exactly as it does on the front
    wing's -- `test_tier51_racelab_graph` asserts it.
    """

    def __init__(self, body: Body, device: str = "cpu") -> None:
        self.body = body
        self.wing = W.FlexWing(
            chord=body.chord * DX, alpha=body.alpha, c_n=body.c_n,
            x_le=body.x_le * DX, y_mount=body.y_le * DX,
            n_station=body.n_station, device=device)
        import torch
        self._zero = torch.zeros(body.n_station, dtype=W.TORCH_DTYPE, device=device)

    @property
    def body_id(self) -> str:
        return self.body.body_id

    def forcing(self, u, v, ny: int, nx: int, w_plate=None):
        """``(fx, fy, w, load, drag)`` on the lattice.  Rigid: delta = 0.

        ``w_plate`` is the SURFACE's own normal velocity, the port's FLOW
        half, which `FlexWing.station_normal` subtracts from the external
        flow's normal component.  It is zero for every fixed body and is
        where a rolling wheel's surface velocity enters (`WheelBody`).

        ``load`` is downforce per unit span, positive DOWN, and ``drag`` is the
        streamwise component of the same station tractions -- one evaluation,
        two projections, so the two can never disagree about the force.
        """
        wp = self._zero if w_plate is None else w_plate
        fx, fy, w, load = self.wing.forcing(u, v, self._zero, wp, ny, nx)
        fn = self.wing.normal_traction(w)
        drag = (fn * float(self.wing.n_hat[0]) * self.wing.ds).sum()
        return fx, fy, w, load, drag


class WheelBody:
    """A wheel: a closed ring of plate segments, on a ground that moves.

    **The simplest defensible form, and the simplification is named.**  The
    wheel is a circle of ``n_seg`` porous plate segments, each carrying the same
    normal traction every other body carries.  It is therefore a bluff circular
    body with a wake and no lift, and:

      * **it barely rotates in any way the fluid can see, and that is
        measured rather than asserted.**  The surface velocity is COMPUTED --
        ``u_s = omega z x (r - r_c)`` at every station, projected onto that
        station's outward normal and handed to `FlexWing.station_normal` as
        the port's FLOW half -- so what the measurement reports is what the
        model does with it.  In the CONTINUUM the answer is exactly zero:
        ``u_s`` is tangential to the circle, the closure carries a normal
        traction only, and ``u_s . n = 0``.  On a twelve-sided polygon a
        chord's normal is radial only at its midpoint, so the projection is
        zero there and ``omega s`` at a station ``s`` along the chord, and what
        survives is a DISCRETISATION artifact of the polygon rather than
        physics.  Its size, and what it moves in the field, are in stage
        ``geometry``.  A real rotating-surface condition needs a TANGENTIAL
        closure this project does not have, and adding one would be a new
        constant with no measurement behind it.
      * it has no contact patch and no tyre deformation -- the requirements'
        section 11 puts a tyre contact model out of scope.
      * it is a two-dimensional slice, so it is a disc and not a torus, and
        there is no flow around its sides, which is most of a real wheel's drag.
    """

    def __init__(self, body_id: str, xc: float, yc: float, r: float,
                 n_seg: int = 12, c_n: float = None, group: str = "wheel",
                 label: str = "", omega: float = 0.0, n_station: int = 3,
                 device: str = "cpu") -> None:
        self.body_id = body_id
        self.xc, self.yc, self.r, self.n_seg = xc, yc, r, n_seg
        self.group, self.label = group, label
        c_n = WHEEL_CN if c_n is None else float(c_n)
        self.c_n = c_n
        #: the shaft speed, in the tiling's units.  Rolling without slip on a
        #: ground that moves at the free stream is ``omega = U_inf / R``.
        self.omega = float(omega)
        self.n_station = int(n_station)
        self.segments: list[PlateBody] = []
        self._bodies: list[Body] = []
        side = 2.0 * r * math.sin(math.pi / n_seg)
        for k in range(n_seg):
            th0 = 2.0 * math.pi * k / n_seg
            th1 = 2.0 * math.pi * (k + 1) / n_seg
            x0, y0 = xc + r * math.cos(th0), yc + r * math.sin(th0)
            x1, y1 = xc + r * math.cos(th1), yc + r * math.sin(th1)
            b = Body(body_id=f"{body_id}_s{k:02d}", x_le=x0, y_le=y0,
                     chord=side, alpha_deg=math.degrees(math.atan2(y1 - y0, x1 - x0)),
                     c_n=c_n, n_station=n_station, group=group, label=label)
            self._bodies.append(b)
            self.segments.append(PlateBody(b, device=device))

    @property
    def bodies(self) -> list[Body]:
        return list(self._bodies)

    def x_span(self) -> tuple[float, float]:
        return (self.xc - self.r, self.xc + self.r)

    def y_span(self) -> tuple[float, float]:
        return (self.yc - self.r, self.yc + self.r)

    def cylinder_drag(self, u_inf: float = GE.U_INF) -> float:
        """The drag `WHEEL_CN` was derived to reproduce: ``0.5 C_D U^2 D``."""
        return 0.5 * WHEEL_CD * u_inf ** 2 * (2.0 * self.r * DX)

    def rolling_omega(self, u_inf: float = GE.U_INF) -> float:
        """Rolling without slip on a ground moving at ``u_inf``."""
        return u_inf / (self.r * DX)

    def surface_normal_velocity(self, seg: PlateBody):
        """``u_s . n`` per station: what the rotation actually contributes.

        ``u_s = omega z x (r - r_c)`` is tangential to the CIRCLE, and a
        chord's normal is radial only at the chord's midpoint, so this is
        zero at the midpoint and ``omega s`` at a station ``s`` along the
        chord.  It is returned rather than assumed zero, which is what
        makes the rotation measurement a measurement.
        """
        cx, cy = seg.wing.stations(seg._zero)
        rx = cx - self.xc * DX
        ry = cy - self.yc * DX
        usx, usy = -self.omega * ry, self.omega * rx
        return usx * seg.wing.n_hat[0] + usy * seg.wing.n_hat[1]

    def max_surface_normal_velocity(self) -> float:
        """The polygon's artifact, as one number."""
        import torch
        return max(float(torch.abs(self.surface_normal_velocity(s)).max())
                   for s in self.segments)

    def forcing(self, u, v, ny: int, nx: int, w_plate=None):
        fx = fy = None
        load = drag = 0.0
        for s in self.segments:
            wp = (self.surface_normal_velocity(s) if self.omega else None)
            a, b, _w, l, d = s.forcing(u, v, ny, nx, w_plate=wp)
            fx = a if fx is None else fx + a
            fy = b if fy is None else fy + b
            load, drag = load + l, drag + d
        return fx, fy, None, load, drag


# ---------------------------------------------------------------------------
# the declared car
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarParams:
    """The geometry knobs section 3.3 exposes that reach the GEOMETRY.

    Phase 1 wires the three that move bodies and nothing else; the cooling,
    powertrain and brake knobs reach subsystems phase 1 marches but does not
    re-site, and the requirements' rule is that a knob that cannot reach its
    subsystem is omitted and recorded rather than shown.  `CAR_NOTES` records
    which of section 3.3's thirteen this phase moves.
    """

    ride_height: float = 0.22        # cells above the road, the floor's leading edge
    rake: float = 0.10               # rear ride height minus front, in cells per unit
    #: **The two knob-driven angles are the DRAWN ones (2026-09-13).**  `DIFF`
    #: and `FW_FLAP` take their incidence from these fields rather than from
    #: `car_geometry.json`, so a default that disagreed with the drawing meant
    #: the model built a flap at 30 degrees that had been drawn at 21.8 and a
    #: diffuser at 11 that had been drawn flat -- the editor showed one car and
    #: the march ran another.  The defaults now ARE the drawing, so the knob
    #: still sweeps its declared range and the nominal car is the drawn car.
    diffuser_deg: float = 0.0
    front_flap_deg: float = 21.8
    #: RW_FLAP is `rear_wing_deg + 16`, and it was drawn at 43.6, so the
    #: default is 27.6.  RW_MAIN was deleted from the car, so nothing else
    #: reads this field -- which is why the mismatch went unnoticed until
    #: the built angles were listed against the drawn ones.
    rear_wing_deg: float = 27.6


#: The car, front to back, in cells on the ``608 x 240`` lattice.
#:
#: Read it as a side view with the road at ``y = 0`` and the flow running left
#: to right.  Angles are to the free stream and POSITIVE means the trailing edge
#: is higher than the leading edge, which for a wing in this project's sign
#: convention is the downforce-making direction (`ground_effect` line 211).
#: Where the car's shape lives.  **It is data, not code.**  A geometry that
#: only exists as literals in a function can only be changed by someone who
#: reads Python, and the person who knows what a Formula One car looks like is
#: not necessarily that person.  `scripts/car_editor.py` edits this file with
#: the mouse and `scripts/car_check.py` tells you whether what you drew is
#: something this model can actually march.
GEOMETRY_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "car_geometry.json")

_GEOM_CACHE: dict[str, Any] = {}


def load_geometry(path: str = None, reload: bool = False) -> dict:
    """Read `car_geometry.json`.  Cached, because `car_bodies` is hot."""
    path = GEOMETRY_JSON if path is None else path
    key = os.path.abspath(path)
    if reload or key not in _GEOM_CACHE:
        with open(path, encoding="utf-8") as fh:
            _GEOM_CACHE[key] = json.load(fh)
    return _GEOM_CACHE[key]


def car_bodies(p: CarParams = None, device: str = "cpu",
               geometry: dict = None) -> tuple[list[Any], list[Body]]:
    """The car's bodies, and the flat list of `Body` records behind them.

    **The shape comes from `car_geometry.json`.**  Every plate is a
    leading-edge point, a chord in cells and an angle in degrees, with POSITIVE
    alpha meaning the trailing edge is HIGHER; every wheel is a centre and a
    radius.  Three kinds of entry are not free numbers, because a knob owns
    them and would overwrite anything stored here:

      ``y_plus_ride``   the height is this plus `CarParams.ride_height`
      ``y_plus_duct``   the height is `DUCT_Y0` plus this, so the radiator
                        duct stays exactly `DEVICE_CELLS` tall whatever the
                        band is moved to
      ``alpha_from``    the angle is `CarParams.<that name>` plus an optional
                        ``alpha_offset`` -- the diffuser ramp, the front flap
                        and the rear wing are section 3.3's own knobs

    The editor shows those locked and says so, rather than letting a number be
    typed that the build would silently discard.
    """
    p = CarParams() if p is None else p
    doc = load_geometry() if geometry is None else geometry
    out: list[Any] = []

    for e in doc["plates"]:
        y = float(e["y"])
        if "y_plus_ride" in e:
            y = float(e["y_plus_ride"]) + p.ride_height
        if "y_plus_duct" in e:
            y = DUCT_Y0 + float(e["y_plus_duct"])
        a = float(e["alpha"])
        if "alpha_from" in e:
            a = (float(getattr(p, e["alpha_from"]))
                 + float(e.get("alpha_offset", 0.0)))
        kw: dict[str, Any] = {}
        if e.get("n_station"):
            kw["n_station"] = int(e["n_station"])
        b = Body(body_id=str(e["id"]), x_le=float(e["x"]), y_le=y,
                 chord=float(e["chord"]), alpha_deg=a,
                 group=str(e.get("group", "body")),
                 cuttable=bool(e.get("cuttable", False)),
                 label=str(e.get("label", "")), **kw)
        out.append(PlateBody(b, device=device))

    for w in doc["wheels"]:
        out.append(WheelBody(str(w["id"]), float(w["x"]), float(w["y"]),
                             float(w["r"]),
                             label=str(w.get("label", "wheel"))))

    flat: list[Body] = []
    for o in out:
        if isinstance(o, PlateBody):
            flat.append(o.body)
        else:
            flat.extend(o.bodies)
    return out, flat


def geometry_fingerprint(p: CarParams = None, geometry: dict = None) -> str:
    """A hash of the car AS BUILT, so a field can say which car it belongs to.

    **A settled field belongs to a geometry, and nothing recorded which.**  The
    demo released the traced car from the hand-drawn car's settled field in
    Tier 54, and then released the drawn car from the traced car's in Tier 55
    -- the same defect twice, each time found only because the envelope stamp
    turned red.  A directory name cannot carry the answer, because the car is
    now edited with a mouse and the directory does not change when it does.

    The hash is over the BUILT bodies rather than the file's bytes: the file's
    bytes move with line endings and key order, which change nothing, and the
    knob-owned entries (`y_plus_ride`, `y_plus_duct`, `alpha_from`) are only
    resolved at build time, which changes everything.  ``label`` is left out
    because it is a caption.
    """
    import hashlib
    _objs, flat = car_bodies(p, geometry=geometry)
    rows = ["RNX=%d RNY=%d DUCT_Y0=%r DEVICE_CELLS=%d"
            % (RNX, RNY, float(DUCT_Y0), int(IU.DEVICE_CELLS))]
    for b in flat:
        rows.append("|".join((
            str(b.body_id), repr(float(b.x_le)), repr(float(b.y_le)),
            repr(float(b.chord)), repr(float(b.alpha_deg)),
            repr(float(b.c_n)), str(int(b.n_station)), str(b.group),
            str(bool(b.cuttable)))))
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


#: The duct's lower wall, in cells.  The device planes span the 32 cells above
#: it, which is `integration_union.DEVICE_CELLS` exactly: the duct is declared
#: 32 cells tall so that a device fills it rather than being declared to.
#:
#: **Lowered from 44 to 18 on 2026-09-13, when the car became a traced
#: silhouette rather than a hand-drawn one.** A real Formula One body TAPERS
#: behind the cockpit, and the old band put the duct's roof at y = 76 where a
#: traced engine cover sits at y = 60.6 at its lowest -- the duct poked out
#: through the bodywork by 16 cells and the turbine plane floated OUTSIDE the
#: car, at the rear wheel.  The device planes' x positions are untouched in
#: kind: they still sit on tiling seams inside the duct, so the graph's
#: topology is the same and only the rows the devices occupy have moved.  The
#: traced body spans y = 9.4 to 60.6 at the duct's tightest station, so a
#: 32-cell duct at 18..50 clears the floor by 8.6 cells and the shell by 10.6.
DUCT_Y0 = 18

#: The x-window the radiator core's plane is allowed to sit in: inside the duct,
#: clear of both its ends.  `windows_from_geometry` puts a CUT here, because
#: `integration_union.DeviceSite` declares a device on the overlap of a tiling
#: x-seam and nowhere else.
DUCT_CORE_RANGE = (268.0, 296.0)

#: The x-window the recovery turbine's plane is allowed to sit in: the same
#: duct, DOWNSTREAM of the core.  The two devices have to be at least one
#: stride apart, because each needs a cut of its own, which is why the duct is
#: 150 cells long and not the 92 a radiator alone would need.  Its downstream
#: end stops at x = 378, clear of the rear tyre, which at the duct's heights
#: spans x = 443.4 to 486.2.
TURBINE_RANGE = (380.0, 406.0)


#: Every departure from a real car, in one place, so no page has to hunt for
#: them and none of them is discovered on screen.
CAR_NOTES: tuple[str, ...] = (
    "The car is a two-dimensional centreline slice.  There is no flow around "
    "the sides of anything, which is where most of a real car's wheel drag and "
    "all of its front wing's tip vortex live.",
    "The car is SHORT, and by a MEASURED factor.  At the inherited "
    "L0 = 0.50 m the body runs x = 72 to 501 cells and y = 2 to 102, which is "
    "3.35 m long and 0.78 m tall against an F1 car's 5.6 m and 0.95 m: "
    "compressed 1.7x in length and 1.2x in height.  So the silhouette is a "
    "recognisable car and not a dimensionally faithful one, and the front "
    "wing's chord is at its own real size (0.25 m) inside a body that is not.  "
    "A 1024-cell box would hold a full-length car and stage size measured it "
    "at 0.219 s a macro-step against the chosen box's 0.156 -- affordable, and "
    "rejected because it is 8.0 m of domain for 5.6 m of car.",
    "The Reynolds number is the tiling's 250, not a vehicle's 1.6e6.  "
    "vehicle-scale-and-sizing section 1.5: fixing lengths, times and speeds "
    "does not fix Re.",
    "The ground is a ROLLING ROAD held at the free stream with no boundary "
    "layer -- ground_effect's band condition, inherited unchanged.  It is the "
    "moving ground the requirements' section 3.1 asks for, and it was already "
    "there: nothing new was written for it.",
    "The wheels are declared NOT to rotate, and that is a measured decision "
    "rather than an omission.  In the continuum a rolling surface's velocity "
    "is tangential and the project's closure carries a normal traction only, "
    "so the rotation is exactly invisible.  On the twelve-sided polygon that "
    "represents the wheel here, a chord's normal is radial only at its "
    "midpoint, so a rigid rotation projects onto it as omega*s -- 17% of the "
    "free stream at the declared geometry, which moved the field by 13% over "
    "eight macro-steps.  Declaring the wheel to roll would show a "
    "discretisation artifact and call it a rotating wall, so it is not "
    "declared and the arm that measures it is kept.",
    "Every body is POROUS.  There is no no-slip surface anywhere, no boundary "
    "layer on any body and no Kutta condition on any wing -- poc2-frontwing-"
    "results' own list, unchanged, and now eleven times over instead of once.",
    "The rotor is an energy-recovery turbine in the radiator duct's exit flow.  "
    "A real car has no such device; it is wake_array's actuator disk re-sited "
    "so that J3's topology -- a device in the flow driving the machine the "
    "block is bolted to -- is the one CS-18 measured.  The claim is about the "
    "coupling, not about F1 hardware.",
    "The sidepod is two plate surfaces, not a closed body.  Flow passes "
    "between them where a real pod is solid, and the duct inside it is a "
    "second pair of surfaces rather than a bounded passage.",
)


def body_extents(bodies: Sequence[Body], include_cuttable: bool = False
                 ) -> list[tuple[float, float, str]]:
    """``(x_lo, x_hi, body_id)`` for every body a cut should avoid."""
    out = []
    for b in bodies:
        if b.cuttable and not include_cuttable:
            continue
        lo, hi = b.x_span()
        out.append((lo, hi, b.body_id))
    return out


# ===========================================================================
# the tiling -- `GroundTiling`'s machinery with the LAYOUT handed in
# ===========================================================================


@dataclass(frozen=True)
class RaceTiling(GroundTiling):
    """`GroundTiling` with the offsets given rather than derived from a grid.

    **The parent's machinery is untouched.**  `weights`, `cut`, `assemble`,
    `contaminated`, `artificial_faces` and `partition_of_unity` are all written
    against ``self.offsets``, ``self.wx``, ``self.wy``, ``self.nx`` and
    ``self.ny``, and every one of those is a property.  Overriding the five
    properties therefore moves the layout without touching one line of the
    partition of unity, which is the requirement: reuse `wing_fsi`'s overlapping
    tiling, halo and partition of unity, do not write a second copy.

    ``halo`` is the SMALLEST overlap in the layout, because that is the number
    the graph declares and the number a bound is evaluated at; `overlaps`
    reports all of them.
    """

    cols: tuple[int, ...] = ()
    rows: tuple[int, ...] = ()
    nx_cells: int = RNX
    ny_cells: int = RNY
    wx_cells: int = WX
    wy_cells: int = WY

    @classmethod
    def of(cls, cols: Sequence[int], rows: Sequence[int], nx: int = RNX,
           ny: int = RNY, wx: int = WX, wy: int = WY,
           ramp: int = RAMP) -> "RaceTiling":
        """Build from an explicit layout.

        The parent's ``n_col``, ``n_row``, ``nw`` and ``stride`` are real
        dataclass fields, so they are FILLED here rather than shadowed: a
        non-uniform layout has no single stride, and the one recorded is the
        smallest, which is the only one a bound could be evaluated at.  Nothing
        in the parent's machinery reads ``stride`` on this class -- `offsets`
        and `halo` are both overridden -- and a test asserts that the recorded
        stride and the actual offsets agree wherever the layout IS uniform.
        """
        cols = tuple(int(c) for c in cols)
        rows = tuple(int(r) for r in rows)
        stride = min([cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
                     + [rows[j + 1] - rows[j] for j in range(len(rows) - 1)]
                     or [0])
        return cls(n_col=len(cols), n_row=len(rows), nw=wx, stride=stride,
                   ramp=ramp, cols=cols, rows=rows, nx_cells=nx, ny_cells=ny,
                   wx_cells=wx, wy_cells=wy)

    # -- the properties the parent's machinery reads ------------------------

    @property
    def is_single(self) -> bool:
        return len(self.cols) == 1 and len(self.rows) == 1 and \
            self.wx_cells == self.nx_cells and self.wy_cells == self.ny_cells

    @property
    def nx(self) -> int:
        return self.nx_cells

    @property
    def ny(self) -> int:
        return self.ny_cells

    @property
    def wx(self) -> int:
        return self.wx_cells

    @property
    def wy(self) -> int:
        return self.wy_cells

    @property
    def n_windows(self) -> int:
        return len(self.cols) * len(self.rows)

    @property
    def offsets(self) -> list[tuple[int, int]]:
        return [(int(a), int(b)) for b in self.rows for a in self.cols]

    @property
    def names(self) -> list[str]:
        return [f"F{i}{j}" for j in range(len(self.rows))
                for i in range(len(self.cols))]

    @property
    def halo(self) -> int:
        return min(self.overlaps()) if self.overlaps() else 0

    # -- what the layout is, reported ---------------------------------------

    def overlaps(self) -> list[int]:
        xs = [self.wx_cells - (self.cols[i + 1] - self.cols[i])
              for i in range(len(self.cols) - 1)]
        ys = [self.wy_cells - (self.rows[j + 1] - self.rows[j])
              for j in range(len(self.rows) - 1)]
        return xs + ys

    def x_bands(self) -> list[tuple[int, int]]:
        """The x-overlap bands, ``[lo, hi)`` on the global lattice: the cuts."""
        return [(self.cols[i + 1], self.cols[i] + self.wx_cells)
                for i in range(len(self.cols) - 1)]

    def y_bands(self) -> list[tuple[int, int]]:
        return [(self.rows[j + 1], self.rows[j] + self.wy_cells)
                for j in range(len(self.rows) - 1)]

    def covers(self) -> bool:
        """Every cell inside at least one window, with no gap at any join."""
        if self.cols[0] != 0 or self.rows[0] != 0:
            return False
        if self.cols[-1] + self.wx_cells != self.nx_cells:
            return False
        if self.rows[-1] + self.wy_cells != self.ny_cells:
            return False
        return all(o >= 1 for o in self.overlaps())

    def clearance(self, extents: Sequence[tuple[float, float, str]]
                  ) -> tuple[float, dict[str, float]]:
        """Cells between each x-cut and the nearest body it does not cut.

        Negative means a cut passes through that body, which is exactly what
        W124 found `wake_array`'s tiling rule could not avoid.  Here it is an
        objective rather than a check.
        """
        per: dict[str, float] = {}
        worst = float(self.nx_cells)
        for lo, hi in self.x_bands():
            for blo, bhi, bid in extents:
                if blo >= hi:                      # body entirely downstream
                    d = blo - hi
                elif bhi <= lo:                    # body entirely upstream
                    d = lo - bhi
                else:                              # the cut passes through it
                    d = -1.0
                per[bid] = min(per.get(bid, float(self.nx_cells)), d)
                worst = min(worst, d)
        return worst, per

    def y_clearance(self, bodies: Sequence[Body]) -> float:
        """Cells between each y-cut and the nearest body, `wing_fsi`'s check."""
        worst = float(self.ny_cells)
        for lo, hi in self.y_bands():
            for b in bodies:
                blo, bhi = b.y_span()
                d = blo - hi if blo >= hi else (lo - bhi if bhi <= lo else -1.0)
                worst = min(worst, d)
        return worst

    def band_centre(self, k: int) -> float:
        """The device plane a cut can carry: the centre of x-band ``k``."""
        lo, hi = self.x_bands()[k]
        return 0.5 * (lo + hi)

    def window_of(self, x: float, y: float) -> str:
        """The window whose box holds ``(x, y)`` furthest from its own edges."""
        best, best_m = None, -1.0
        for k, (ox, oy) in enumerate(self.offsets):
            if not (ox <= x < ox + self.wx_cells and oy <= y < oy + self.wy_cells):
                continue
            m = min(x - ox, ox + self.wx_cells - x, y - oy, oy + self.wy_cells - y)
            if m > best_m:
                best, best_m = self.names[k], m
        if best is None:
            raise ValueError(f"({x}, {y}) is in no window of this layout")
        return best

    def owns(self, b: Body) -> str | None:
        """The single window that owns a whole body in x AND y, or None."""
        xlo, xhi = b.x_span()
        ylo, yhi = b.y_span()
        for k, (ox, oy) in enumerate(self.offsets):
            if ox <= xlo and xhi <= ox + self.wx_cells and \
               oy <= ylo and yhi <= oy + self.wy_cells:
                return self.names[k]
        return None


# ---------------------------------------------------------------------------
# the layout, derived from the car
# ---------------------------------------------------------------------------


#: The two ways to ask "how much car is at this ``x``", and the default.
PROFILE_KINDS = ("occupancy", "release-force")


def body_profile(bodies: Sequence[Body] = None, nx: int = RNX,
                 kind: str = "occupancy", objects: Sequence[Any] = None,
                 ny: int = RNY, device: str = "cpu") -> np.ndarray:
    """``Phi(x)``: how much car each lattice COLUMN carries.

    The layout minimises the ``Phi`` that lands inside a cut band, so what
    ``Phi`` measures is the whole content of "derived from the geometry".  Two
    answers, and the second is the control:

    ``occupancy`` (the default, and purely GEOMETRIC).  Every station of every
      body contributes its own arc length ``ds``, spread in ``x`` over the same
      ``sigma_n`` the stamping kernel uses, and nothing depends on the flow.

    ``release-force`` (the control).  The actual ``|f_x| + |f_y|`` the bodies
      stamp at the release state -- uniform flow at ``U_inf``, ``v = 0``.

    **Why the geometric one is the default, measured rather than asserted.**
    At the release state the normal traction is ``0.5 c_n |w| w`` with
    ``w = -U sin(alpha)``, so a body at zero incidence carries EXACTLY zero
    force.  The car's floor is at zero incidence.  So under ``release-force``
    a cut straight through the floor is free, which is false the moment the
    flow develops -- and a layout is computed once, at startup, and lives for
    the whole run (requirements section 4.1).  An objective that is blind to
    the floor is the wrong objective, and the two layouts are reported side by
    side so the difference is a number rather than an argument.

    The state-independence is the point.  A profile read off a developed field
    would make the decomposition a function of the flow, the flow a function of
    the decomposition, and the layout a fixed point of its own answer.
    """
    if kind not in PROFILE_KINDS:
        raise ValueError(f"kind is one of {PROFILE_KINDS}, not {kind!r}")
    if kind == "release-force":
        import torch
        if objects is None:
            objects, _flat = car_bodies(device=device)
        u = torch.full((ny, nx), GE.U_INF, dtype=W.TORCH_DTYPE, device=device)
        v = torch.zeros((ny, nx), dtype=W.TORCH_DTYPE, device=device)
        tot = torch.zeros((ny, nx), dtype=W.TORCH_DTYPE, device=device)
        for b in objects:
            fx, fy, _w, _l, _d = b.forcing(u, v, ny, nx)
            tot = tot + torch.abs(fx) + torch.abs(fy)
        return tot.sum(dim=0).detach().cpu().numpy()

    if bodies is None:
        _objects, bodies = car_bodies(device=device)
    xc = np.arange(nx) + 0.5
    out = np.zeros(nx)
    sig = GE.SIGMA_N                                   # cells, the kernel's own
    for b in bodies:
        ds = b.chord / b.n_station
        s = (np.arange(b.n_station) + 0.5) * ds
        sx = b.x_le + s * math.cos(b.alpha)
        k = np.exp(-0.5 * ((xc[None, :] - sx[:, None]) / sig) ** 2)
        k = k / (k.sum(axis=1, keepdims=True) + 1e-300)
        out += (ds * k).sum(axis=0)
    return out


#: Kept under its old name because the case study and the tests name it.
def force_profile(objects: Sequence[Any] = None, nx: int = RNX, ny: int = RNY,
                  device: str = "cpu") -> np.ndarray:
    """`body_profile` with ``kind='release-force'``: the control."""
    return body_profile(None, nx, "release-force", objects, ny, device)


def windows_from_geometry(bodies: Sequence[Body],
                          device_ranges: Sequence[tuple[float, float]]
                          = (DUCT_CORE_RANGE, TURBINE_RANGE),
                          nx: int = RNX, ny: int = RNY,
                          wx: int = WX, wy: int = WY,
                          halo_min: int = HALO_MIN,
                          n_col: int | None = None,
                          n_col_cap: int = 12,
                          stride_min: int = 48,
                          device_overlap_max: int = 16,
                          profile: str = "occupancy",
                          clear_bodies: Sequence[str] = ("FW_MAIN", "FW_FLAP"),
                          step: int = 2,
                          phi: np.ndarray | None = None,
                          objects: Sequence[Any] | None = None
                          ) -> tuple[RaceTiling, dict[str, Any]]:
    """Lay the windows out along the car: cuts where the car is thinnest.

    **The obvious objective is unachievable here, and finding that out is the
    first real result of the decomposition.**  W124's discipline, which CS-10
    set and `wing_fsi` inherited, is that no seam should pass through a body:
    a body cut by a seam has its force split between two experts that exchange
    only a ring.  `WingTiling.cuts_clear_of_wing` checks it and the front wing
    satisfies it with eight cells to spare, because there is ONE body in 208
    cells.  Here there are eleven bodies whose x-spans cover 72 to 467 of 608
    cells almost without a gap, and the largest stride the halo allows is 112,
    so **every layout that covers the domain puts at least three cuts through
    the car.**  There is no clear-of-the-bodies layout to find.

    So the objective is not "avoid the bodies" but **"cut where the car is
    thinnest"**: minimise the total body force, at the release state, that falls
    inside a cut band.

        minimise   sum over bands B of  sum over x in B of Phi(x)

    with ``Phi`` from `force_profile`.  That is exact and additive, so it is a
    shortest-path problem over the offsets and is solved by dynamic programming
    rather than searched -- no layout is missed and none is sampled.

    **One hard constraint, from the union's own declarations.**
    `integration_union.DeviceSite` declares a device on the OVERLAP of a tiling
    x-seam and nowhere else: the inflow ring is the downstream window's ``xlo``
    ring and the outflow ring the upstream window's ``xhi``.  So the radiator
    core and the recovery turbine each need a cut to sit ON, inside the x-range
    their geometry allows.  Those two cuts are not free choices and the cut
    through the sidepod that the core's cut implies is REQUIRED rather than
    accidental -- which is why the pod and duct surfaces are declared
    ``cuttable``.

    **The rows are not a choice at all.**  ``ny = 240`` with ``wy = 128`` admits
    exactly one row offset above zero, ``112``: lower leaves the top uncovered,
    higher is not a row.  So the y-cut is ``[112, 128)`` whatever the car looks
    like and what the car has to do is stay below it.  `y_clearance` measures by
    how much, and there the clear-of-the-bodies discipline IS met.

    ``n_col`` fixes the window count; ``None`` returns the best over every count
    the stride admits.  `layout_frontier` returns the whole curve, which is the
    control that says what the count costs.
    """
    if phi is None:
        phi = body_profile(bodies, nx, profile, objects, ny)
    extents = body_extents(bodies)
    want = set(clear_bodies)
    must_clear = [b.x_span() for b in bodies if b.body_id in want]
    missing = want - {b.body_id for b in bodies}
    if missing:                                              # pragma: no cover
        raise ValueError(f"clear_bodies names {sorted(missing)}, which the car "
                         "does not have")
    rows = (0, ny - wy)
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(phi, dtype=float))])

    last = nx - wx
    stride_max = wx - halo_min
    if stride_min > stride_max:                              # pragma: no cover
        raise ValueError(f"stride_min {stride_min} exceeds the largest stride "
                         f"the halo allows, {stride_max}")
    nodes = list(range(0, last + 1, step))
    if nodes[-1] != last:
        nodes.append(last)
    idx = {o: i for i, o in enumerate(nodes)}
    n_mask = 1 << len(device_ranges)
    #: The largest column count the DP carries as state.  The stride admits at
    #: most `len(nodes)` columns in principle and nothing here wants more than
    #: ten windows a row, so the state is capped and the cap is declared: a
    #: layout needing more columns than this is not searched and `frontier`
    #: says where it stops.
    n_slot = int(n_col_cap)

    INF = float("inf")
    # best[i][mask][c] -- least banded force reaching node i with `mask` device
    # ranges served by exactly `c` columns so far.
    best = [[[INF] * (n_slot + 1) for _ in range(n_mask)] for _ in nodes]
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    best[0][0][1] = 0.0
    for i, o in enumerate(nodes):
        for mask in range(n_mask):
            for c in range(1, n_slot):
                base = best[i][mask][c]
                if base == INF:
                    continue
                for p in nodes:
                    if not (o + stride_min <= p <= o + stride_max):
                        continue
                    band_lo, band_hi = p, o + wx
                    if band_hi <= band_lo:
                        continue
                    centre = 0.5 * (band_lo + band_hi)
                    bit = 0
                    for d, (lo, hi) in enumerate(device_ranges):
                        if lo <= centre <= hi:
                            bit |= (1 << d)
                    # a band may serve at most one device range, and a range
                    # already served may not be served twice
                    if bit & mask:
                        continue
                    # **A device's band has to be narrow.**  `DeviceSite` reads
                    # the inflow on the downstream window's `xlo` ring and the
                    # outflow on the upstream window's `xhi`, so the band's
                    # width IS the distance between the two rings the device
                    # sees, and the plane sits half of it from each.  CS-18
                    # section 7.1 measured that gap at 7.5 cells accounting for
                    # 4.4% of J3's receiving balance; a band wide enough to be
                    # convenient would put it at forty and the balance would be
                    # reporting the layout rather than the join.
                    if bit and (band_hi - band_lo) > device_overlap_max:
                        continue
                    # and a band that is NOT a device's must not accidentally
                    # land inside a device range, or the device has no seam
                    if (not bit) and any(lo <= centre <= hi
                                         for lo, hi in device_ranges):
                        continue
                    # **The front wing must be clear of every cut, and that
                    # is not an aesthetic preference.**  The `wet` and `mount`
                    # seams pair the STRUCTURE and the SUSPENSION with ONE
                    # fluid window, so the plate has to be owned by one window
                    # -- `wing_fsi.WingTiling.wing_window` raises if it is not.
                    # A plate inside an overlap would have two experts carrying
                    # its force and one of them declaring the seam, which is a
                    # declaration that does not describe the march.  What it
                    # costs is measured: `clear_bodies=()` is the control.
                    if any(blo < band_hi and band_lo < bhi
                           for blo, bhi in must_clear):
                        continue
                    cost = base + float(cum[min(band_hi, nx)] - cum[band_lo])
                    j, m2, c2 = idx[p], mask | bit, c + 1
                    if c2 > n_slot:
                        continue
                    if cost < best[j][m2][c2]:
                        best[j][m2][c2] = cost
                        prev[(j, m2, c2)] = (i, mask, c)

    full = n_mask - 1
    j_end = idx[last]
    cands = [(best[j_end][full][c], c) for c in range(2, n_slot + 1)
             if best[j_end][full][c] < INF]
    if not cands:
        raise ValueError(
            "no column layout covers the domain at this stride AND puts one cut "
            "in each device range: the car, the box and the halo are "
            "inconsistent, and that is a finding rather than a bug")
    frontier = {c: v for v, c in sorted(cands, key=lambda t: t[1])}
    if n_col is None:
        cost, c_best = min(cands)
    else:
        pick = [(v, c) for v, c in cands if c == n_col]
        if not pick:
            raise ValueError(f"no {n_col}-column layout meets the constraints; "
                             f"the counts that do are {sorted(frontier)}")
        cost, c_best = pick[0]

    cols: list[int] = []
    state = (j_end, full, c_best)
    while state in prev:
        cols.append(nodes[state[0]])
        state = prev[state]
    cols.append(nodes[state[0]])
    cols = tuple(sorted(cols))
    t = RaceTiling.of(cols, rows, nx=nx, ny=ny, wx=wx, wy=wy)
    if not t.covers():                                       # pragma: no cover
        raise AssertionError(f"the dynamic program returned {cols}, which does "
                             "not cover: the transition rule and `covers` "
                             "disagree")
    info = layout_report(t, bodies, device_ranges)
    info.update(
        objective=f"minimise the {profile} profile inside the cut bands, "
                  "subject to one narrow cut in each device range",
        profile=profile,
        banded_force=cost,
        total_force=float(cum[-1]),
        banded_force_fraction=cost / float(cum[-1]) if cum[-1] else None,
        frontier={int(k): float(v) for k, v in frontier.items()},
        nodes_searched=len(nodes),
        stride_max=stride_max,
        stride_min=stride_min,
        device_overlap_max=device_overlap_max,
        clear_bodies=list(clear_bodies),
        step=step,
    )
    return t, info


def layout_frontier(bodies: Sequence[Body], **kw) -> dict[int, dict[str, Any]]:
    """The best layout at each window count: what more windows actually cost.

    A control for the count.  The requirements ask for 12 to 20 fluid windows
    and a single number in that range is a choice; the curve says what the
    choice is between.
    """
    _t, info = windows_from_geometry(bodies, **kw)
    out: dict[int, dict[str, Any]] = {}
    for c in sorted(info["frontier"]):
        try:
            tt, ii = windows_from_geometry(bodies, n_col=c, **kw)
        except ValueError:                                   # pragma: no cover
            continue
        out[tt.n_windows] = {
            "n_col": c, "n_windows": tt.n_windows, "cols": list(tt.cols),
            "banded_force": ii["banded_force"],
            "banded_force_fraction": ii["banded_force_fraction"],
            "min_overlap": min(tt.overlaps()),
            "overlap_ratio": ii["overlap_ratio"],
            "bodies_cut": len({b["body"] for b in ii["bodies_cut"]}),
            "device_planes": ii["device_planes"],
        }
    return out


def layout_report(t: RaceTiling, bodies: Sequence[Body],
                  device_ranges: Sequence[tuple[float, float]]
                  = (DUCT_CORE_RANGE, TURBINE_RANGE)) -> dict[str, Any]:
    """Everything about a layout that a page or a test would want to quote."""
    extents = body_extents(bodies)
    worst, per = t.clearance(extents)
    centres = [t.band_centre(k) for k in range(len(t.x_bands()))]
    device_bands = []
    for lo, hi in device_ranges:
        device_bands.append(next((k for k, c in enumerate(centres)
                                  if lo <= c <= hi), None))
    cut_bodies = []
    for lo, hi in t.x_bands():
        for b in bodies:
            blo, bhi = b.x_span()
            if blo < hi and lo < bhi:
                cut_bodies.append({"band": [lo, hi], "body": b.body_id,
                                   "cuttable": b.cuttable})
    return {
        "cols": list(t.cols), "rows": list(t.rows),
        "n_windows": t.n_windows, "n_col": t.n_col, "n_row": t.n_row,
        "nx": t.nx, "ny": t.ny, "wx": t.wx, "wy": t.wy,
        "overlaps": t.overlaps(), "halo": t.halo,
        "x_bands": [list(b) for b in t.x_bands()],
        "y_bands": [list(b) for b in t.y_bands()],
        "band_centres": centres,
        "device_band_index": device_bands,
        "device_planes": [None if k is None else centres[k] for k in device_bands],
        "min_cut_to_body_clearance_cells": worst,
        "per_body_clearance_cells": per,
        "y_cut_to_body_clearance_cells": t.y_clearance(bodies),
        "bodies_cut": cut_bodies,
        "covers": t.covers(),
        "overlap_ratio": t.n_windows * t.wx * t.wy / (t.nx * t.ny),
        "domain_metres": [t.nx * DX * 0.50, t.ny * DX * 0.50],
    }


#: The layout this tier marches, derived once from `car_bodies` at the default
#: parameters.  It is a module constant so that a test, a script and the graph
#: all get the same one without re-running the search, and so that a change to
#: the car that moves it is visible as a diff rather than as a drift.
def _default_layout() -> tuple[RaceTiling, dict[str, Any]]:
    _objs, flat = car_bodies()
    return windows_from_geometry(flat)


LAYOUT: tuple[RaceTiling, dict[str, Any]] | None = None


def layout() -> tuple[RaceTiling, dict[str, Any]]:
    """The derived layout, computed once and cached."""
    global LAYOUT
    if LAYOUT is None:
        LAYOUT = _default_layout()
    return LAYOUT


# ===========================================================================
# the devices: where J1's core and J3's turbine sit on THIS lattice
# ===========================================================================

#: The window rows a device's plane spans -- `wake_array.ROTOR_CELLS`, which is
#: `integration_union.DEVICE_CELLS`, unchanged at 32 cells.  The radiator duct
#: is declared 32 cells tall for exactly this reason: the device fills it.
DEVICE_CELLS = IU.DEVICE_CELLS


@dataclass(frozen=True)
class RaceDeviceSpec:
    """`vehicle_march.DeviceSpec` against the race tiling instead of the front
    wing's.

    Identical in every formula -- the plane is midway between the two rings, the
    inflow is read on the UPSTREAM one, `disk.disk_average`'s rule -- and
    different only in which tiling the offsets come from.  It is a separate
    class rather than a parameter on `vehicle_march.DeviceSpec` because that
    class is frozen into CS-18's recorded numbers and this tier does not touch
    them (`case-study-vehicle-march-atlas-0.1` section 10's standing rule).
    """

    join: str
    site: IU.DeviceSite
    label: str
    tiling: "RaceTiling"

    @property
    def rows(self) -> np.ndarray:
        _ox, oy = self.tiling.offsets[self.tiling.names.index(self.site.right)]
        return oy + self.site.cells

    @property
    def i_up(self) -> int:
        """Column index of the ring the inflow is read on."""
        ox, _oy = self.tiling.offsets[self.tiling.names.index(self.site.right)]
        return int(ox)

    @property
    def i_down(self) -> int:
        ox, _oy = self.tiling.offsets[self.tiling.names.index(self.site.left)]
        return int(ox + self.tiling.wx - 1)

    @property
    def x_plane(self) -> float:
        return 0.5 * ((self.i_up + 0.5) + (self.i_down + 0.5)) * DX

    @property
    def y0(self) -> float:
        return float(self.rows[0]) * DX

    @property
    def width(self) -> float:
        return float(self.rows.size) * DX

    @property
    def ring_to_plane_cells(self) -> float:
        """How far upstream of its own force a device reads its inflow.

        CS-18 section 7.1 diagnosed J3's whole 4% receiving residual as this
        number: the disk's claim is ``T`` times the velocity on a ring the
        force has not yet slowed.  It is 7.5 cells on the front wing's tiling;
        `windows_from_geometry` constrains the device bands so that it is
        comparable here, and the layout report carries it.
        """
        return self.x_plane / DX - (self.i_up + 0.5)


def device_sites(t: "RaceTiling") -> dict[str, IU.DeviceSite]:
    """The two device sites the derived layout puts on the duct's two cuts.

    The seam ids, the two windows and the first ring cell are all READ OFF the
    layout rather than written down, which is the difference between a
    decomposition derived from geometry and one that has been hand-fitted to
    agree with it.
    """
    bands = t.x_bands()
    centres = [t.band_centre(k) for k in range(len(bands))]
    out: dict[str, IU.DeviceSite] = {}
    for key, (dev, seg, rng) in (
            ("RAD", ("RAD", "core", DUCT_CORE_RANGE)),
            ("ROTOR", ("ROTOR", "rotor", TURBINE_RANGE))):
        k = next((i for i, c in enumerate(centres) if rng[0] <= c <= rng[1]), None)
        if k is None:                                        # pragma: no cover
            raise ValueError(f"the layout has no cut in {dev}'s range {rng}")
        i_left = k                       # column k and column k+1 share band k
        left, right = f"F{i_left}0", f"F{i_left + 1}0"
        oy = t.offsets[t.names.index(right)][1]
        first = int(DUCT_Y0 - oy)
        out[dev] = IU.DeviceSite(dev, seg, f"x{i_left}0", left, right, first)
    return out


class RaceDeviceForcing(VM.DeviceForcing):
    """`vehicle_march.DeviceForcing` on the race lattice.

    The parent reads `wing_fsi.DEFAULT_TILING` for the lattice in four lines of
    its constructor and nothing else; those four are replaced here and every
    formula -- the donor's exact-overlap weighting, the discrete-thrust
    identity asserted on EVERY call, the smearing thickness and its declared
    one-cell floor -- is the parent's.
    """

    def __init__(self, devices, tiling: "RaceTiling", dt_apply: float = None,
                 cell_floor: float = 1.0) -> None:
        super().__init__(devices=devices, dt_apply=dt_apply, cell_floor=cell_floor)
        self.ny, self.nx = tiling.ny, tiling.nx
        self.x_c = (np.arange(self.nx) + 0.5) * DX
        self.y_c = (np.arange(self.ny) + 0.5) * DX


def ring_velocity(u_full: np.ndarray, site: IU.DeviceSite,
                  t: "RaceTiling") -> np.ndarray:
    """The streamwise velocity on the ring upstream of a device's plane."""
    k = t.names.index(site.right)
    ox, oy = t.offsets[k]
    return np.asarray(u_full, dtype=float)[oy + site.cells, ox].copy()



# ===========================================================================
# DECISION 8 -- a machine sized for the inflow its HOST delivers (W222)
# ===========================================================================

#: The inflow `powertrain`'s machine was sized at, which is the ring CS-18's
#: rotor sat in: open flow in `front_wing`'s tiling, downstream of nothing.
#: `machine_for_host` is a similarity ABOUT this point, so it is the one number
#: the whole re-sizing is referred to and it is declared rather than inferred.
U_HOST_REF = 0.9225


def machine_for_host(u_host: float, u_ref: float = U_HOST_REF,
                     scale: float = None) -> dict[str, Any]:
    """`vehicle_march.machine_for_rotor`'s similarity, applied to the INFLOW.

    **Why this exists (W222).**  Tier 51 re-sited the device from open flow into
    a radiator duct downstream of the core, and its inflow fell from ``u_ref``
    to ``0.669``.  The machine sits just above the battery's open-circuit
    voltage, so `powertrain.MachineAgent.validity` -- which declares *"a
    generator can only push current into the battery while its back-EMF exceeds
    the open-circuit voltage; below that speed the loop current reverses and the
    machine MOTORS, which is a legitimate mode and a different one from the mode
    this graph declares"* -- returns False, and every arm of that tier marched
    with it False and nothing asked.

    **This is not a new decision; it is W199's own similarity applied to the
    thing that changed.**  That row re-sized the machine when the disk's WIDTH
    moved.  Here the width is unchanged and the INFLOW moved, and the same two
    conditions decide the scaling uniquely.  Write ``x = u_host / u_ref``:

      * the back-EMF must not move.  ``omega = lambda u / r`` so
        ``omega ~ x``, and ``k_e omega`` invariant gives ``k_e -> k_e / x``;
        ``k_e = k_t`` is the same air-gap flux linkage, so ``k_t`` follows.
      * the torque must follow the disk's.  ``T ~ u^2`` at fixed induction and
        ``tau_disk = T r / lambda``, so ``tau ~ x^2``; with ``k_t ~ 1/x`` that
        needs ``I ~ x^3``, and the numerator ``k_e omega - V_oc`` is invariant,
        so ``R -> R / x^3`` for every element.

    ``x = 1`` must therefore reproduce `machine_for_rotor` exactly, and that is
    the control `scripts/tier52_racelab_switch.py` stage ``envelope`` asserts
    rather than this docstring claiming it.

    **What it costs is a smaller machine on a slower shaft**, and the induction,
    the demand-over-supply ratio and `rotor_valid` all come back to what they
    were at ``u_ref`` -- which is what makes it a similarity rather than a fit.
    """
    x = float(u_host) / float(u_ref)
    if not (x > 0.0):
        raise ValueError(f"the host inflow ratio must be positive, got {x}")
    els = VM.machine_for_rotor(HOST_ROTOR_WIDTH if scale is None else scale)
    x3 = x ** 3
    for e in els.values():
        e.resistance = e.resistance / x3
    mgu = els["MGU"]
    mgu.k_e = mgu.k_e / x
    mgu.k_t = mgu.k_t / x
    mgu.r_total = mgu.r_total / x3
    return els


# ===========================================================================
# the envelope, CONSULTED (W222)
# ===========================================================================

#: What `check_envelopes` reports per expert.  Three states and not two:
#: ``True`` inside, ``False`` declined, and ``None`` **not consultable** -- an
#: expert whose record declares no predicate, or one whose predicate could not
#: be evaluated at this state.  A missing check that defaults to ``False`` is a
#: false alarm and one that defaults to ``True`` is the hole this row exists to
#: close, so neither is allowed.
ENVELOPE_STATES = (True, False, None)


class EnvelopeDeclined(RuntimeError):
    """One or more declared envelopes declined the state the march is in.

    Carries `report` so a caller can say WHICH expert declined and at what
    value, rather than only that something did.
    """

    def __init__(self, message: str, report: dict) -> None:
        super().__init__(message)
        self.report = report

# ===========================================================================
# the graph
# ===========================================================================


@dataclass
class RaceWindow(IU.SegmentedWindow):
    """One fluid window of the car, with a body's wetted seam where there is one.

    `integration_union.SegmentedWindow` unchanged except for WHERE the plate is:
    the parent hard-codes `wing_fsi.X_LE` and the front wing's mount, and the
    car's front wing is somewhere else.  ``wing_body`` re-sites it, and the
    window's own corner is subtracted from both coordinates, which is W130 --
    getting the x one wrong put CS-10's plate outside the window's array, the
    border clamp returned a constant and the probed fluid block came back
    exactly zero with every other diagnostic healthy.
    """

    wing_body: Body = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.wing_body is not None:
            b = self.wing_body
            self.has_wing = True
            self._wing = W.FlexWing(
                chord=b.chord * DX, alpha=b.alpha, c_n=b.c_n,
                x_le=b.x_le * DX - self.ox * DX,
                y_mount=b.y_le * DX - self.oy * DX,
                n_station=W.N_STATION, device=self.device)


def make_experts(u_full: np.ndarray, v_full: np.ndarray, t: "RaceTiling",
                 sites: dict[str, IU.DeviceSite], wing_body: Body,
                 nu: float = GE.NU, flux_mode: str = "reaction",
                 design: dict | None = None) -> dict[str, Any]:
    """The fluid windows, the front wing's structure and its suspension."""
    u_full = np.asarray(u_full, dtype=float)
    v_full = np.asarray(v_full, dtype=float)
    if u_full.shape != (t.ny, t.nx):
        raise ValueError(f"field is {u_full.shape}, expected {(t.ny, t.nx)}")
    by_window: dict[str, dict[str, IU.DeviceSite]] = {}
    for s in sites.values():
        by_window.setdefault(s.left, {})["xhi"] = s
        by_window.setdefault(s.right, {})["xlo"] = s
    owner = t.owns(wing_body)
    if owner is None:                                        # pragma: no cover
        raise ValueError(f"no single window owns {wing_body.body_id}; the wetted "
                         "seam has no well-defined fluid side")
    delta = np.zeros(W.N_STATION)
    d = dict(FW.DESIGN_REF, **(design or {}))
    out: dict[str, Any] = {}
    for k, name in enumerate(t.names):
        ox, oy = t.offsets[k]
        out[name] = RaceWindow(
            agent_id=name,
            u0=u_full[oy:oy + t.wy, ox:ox + t.wx],
            v0=v_full[oy:oy + t.wy, ox:ox + t.wx],
            shared_faces=t.artificial_faces(ox, oy),
            has_wing=False, ox=ox, oy=oy, delta=delta, nu=nu,
            flux_mode=flux_mode, ride_height=wing_body.y_le * DX,
            sites=by_window.get(name, {}),
            wing_body=(wing_body if name == owner else None))
    out["STRUCT"] = W.WingStructure(e_star=d["e_star"], thick=d["tc"] * GE.CHORD,
                                    delta=delta)
    out["SUSP"] = GE.Suspension(k=d["k"], h0=d["h0"], h=wing_body.y_le * DX)
    return out


def connections(t: "RaceTiling", wing_window: str,
                sites: dict[str, IU.DeviceSite],
                joins: Sequence[str] = ("J1", "J2", "J3")) -> list:
    """Every seam: fluid-fluid from the LAYOUT, then the car's and the joins'.

    `wing_fsi.connections` walks an ``n_col x n_row`` grid and names the seams
    from the indices.  The layout here is not a grid in x, but it is still a
    product -- one column list and one row list -- so the same walk names the
    same seams, and a test asserts the seam count is the one the layout implies.
    """
    conns: list = []
    for j in range(t.n_row):
        for i in range(t.n_col - 1):
            conns.append(Connection(
                seam_id=f"x{i}{j}", a=(f"F{i}{j}", "xhi:MECH"),
                b=(f"F{i+1}{j}", "xlo:MECH"),
                port_type=PortType.MECH, derive_space=True,
                geometrically_coincident=False, expected_null_dim=0,
                note="fluid-fluid, an ordinary artificial boundary"))
    for j in range(t.n_row - 1):
        for i in range(t.n_col):
            conns.append(Connection(
                seam_id=f"y{i}{j}", a=(f"F{i}{j}", "yhi:MECH"),
                b=(f"F{i}{j+1}", "ylo:MECH"),
                port_type=PortType.MECH, derive_space=True,
                geometrically_coincident=False, expected_null_dim=0,
                note="fluid-fluid, an ordinary artificial boundary"))
    conns.append(Connection(
        seam_id="wet", a=(wing_window, "wet:MECH"), b=("STRUCT", "wet:MECH"),
        port_type=PortType.MECH, derive_space=True,
        geometrically_coincident=True, expected_null_dim=0,
        effort_normal="STRUCT",
        note="CS-12's seam, on the car's front wing: fluid-structure MECH "
             "across two governing families"))
    conns.append(Connection(
        seam_id="mount", a=(wing_window, "mount:MECH"), b=("SUSP", "mount:MECH"),
        port_type=PortType.MECH, derive_space=True,
        geometrically_coincident=True, expected_null_dim=0,
        effort_normal="SUSP",
        note="CS-10's seam, on the car's front wing: field-to-lumped MECH "
             "carrying the wing's POSITION"))
    for s in sites.values():
        conns = IU._split(conns, s)
    if "J1" in joins:
        conns += IU._device_seams(sites["RAD"], "J1", "the radiator core")
    if "J2" in joins:
        conns.append(Connection(
            seam_id="J2_heat", a=("MGU", "case:THERM"), b=("BLOCK", "outer:THERM"),
            port_type=PortType.THERM, derive_space=True,
            geometrically_coincident=True, expected_null_dim=0,
            effort_normal="BLOCK",
            note="J2: the machine's I^2 R into the block's dry face through the "
                 "mount conductance"))
    if "J3" in joins:
        conns += IU._device_seams(sites["ROTOR"], "J3", "the recovery turbine")
    return conns


def build(u_full: np.ndarray, v_full: np.ndarray,
          t: "RaceTiling" = None, bodies: Sequence[Body] = None,
          joins: Sequence[str] = ("J1", "J2", "J3"),
          clocks: str = "native",
          rotor_width: float = None, machine_scale: float = None,
          flux_matching: FluxMatching = FluxMatching.POINTWISE,
          name: str | None = None) -> tuple[CaseGraph, dict[str, Any]]:
    """The car's union, assembled: the same five families CS-18 marched.

    `integration_union.build` does this on the front wing's tiling and is
    frozen into [[joining-seam-cost]]'s recorded numbers, so it is not touched
    and not parameterised.  This is its shape on the car's lattice, and the
    differences are exactly three: the tiling, the wetted body, and the rotor
    re-sited from `wake_array`'s own lattice to the radiator duct.

    ``joins=()`` is the DISJOINT union, the control every join's cost is a
    difference against.
    """
    joins = tuple(j for j in ("J1", "J2", "J3") if j in tuple(joins))
    if clocks not in ("native", "reconciled"):
        raise ValueError(f"clocks is 'native' or 'reconciled', not {clocks!r}")
    if t is None or bodies is None:
        t, _info = layout()
        _objs, bodies = car_bodies()
    rotor_width = HOST_ROTOR_WIDTH if rotor_width is None else float(rotor_width)
    machine_scale = rotor_width if machine_scale is None else float(machine_scale)
    sites_all = device_sites(t)
    sites = {d: s for d, s in sites_all.items()
             if (d == "RAD" and "J1" in joins) or (d == "ROTOR" and "J3" in joins)}
    wing_body = next(b for b in bodies if b.body_id == WING_BODY_ID)
    info: dict[str, Any] = {"joins": joins, "clocks": clocks,
                            "flux_matching": flux_matching.value,
                            "rotor_width": rotor_width,
                            "machine_scale": machine_scale,
                            "layout": layout_report(t, bodies)}

    experts = make_experts(u_full, v_full, t, sites, wing_body)
    wing_window = t.owns(wing_body)
    #: `front_wing`'s capabilities and not `wing_fsi`'s: the window that owns
    #: the front wing has to declare ``mount:MECH`` beside ``wet:MECH``, which
    #: is the SUSPENSION's seam, and only `front_wing.flow_ports` adds it.
    agents = [Agent(n, FW.flow_capabilities(experts[n], MotionClass.STATIC),
                    domain=f"fluid window {n}") for n in t.names]
    for n in t.names:
        if experts[n].sites:
            a = next(x for x in agents if x.agent_id == n)
            agents[agents.index(a)] = replace(a, capabilities=replace(
                a.capabilities,
                ports=IU.segmented_flow_ports(experts[n], MotionClass.STATIC)))
    agents.append(Agent("STRUCT",
                        W.structure_capabilities(experts["STRUCT"],
                                                 MotionClass.STATIC),
                        domain="Omega_solid: the front wing's section",
                        role="structure"))
    agents.append(Agent("SUSP",
                        FW.suspension_capabilities(experts["SUSP"],
                                                   MotionClass.STATIC),
                        domain="the front wing's mounting face: a lumped "
                               "suspension", role="suspension"))

    # -- the powertrain, settled before either circuit is built --------------
    pt = dict(PT.make_elements())
    pt.update(VM.machine_for_rotor(machine_scale))
    q_machine = None
    if "J3" in joins:
        u_rotor = ring_velocity(u_full, sites["ROTOR"], t)
        pt["ROTOR"] = WA.RotorDisk(agent_id="ROTOR", u_ref=u_rotor)
        op = VM.SizedCircuitSolve(width=rotor_width, rotor=pt["ROTOR"],
                                  elements=pt,
                                  u_ref=float(np.mean(u_rotor))).solve()
        info["operating_point"] = op.as_dict()
    else:
        pt["ROTOR"] = PT._rotor()
    info["mgu_omega"] = float(pt["MGU"].omega)
    info["rotor_u_ref_mean"] = float(np.mean(pt["ROTOR"].u_ref))
    if "J2" in joins:
        p_ref, _i = IU.calibrated_p_ref(PT.MachineAgent())
        current = pt["MGU"].current_at(pt["MGU"].omega)
        q_machine = current * current * pt["MGU"].resistance * p_ref
        info.update(p_ref=p_ref, current_op=current, q_machine=q_machine)

    # -- the coolant circuit ------------------------------------------------
    dt_c = GE.MACRO_DT if clocks == "reconciled" else CL.MACRO_DT
    cl = dict(CL.make_legs(4))
    if "J1" in joins:
        cl["RAD"] = IU.CoreRadiator(u_air=ring_velocity(u_full, sites["RAD"], t))
    for leg in cl.values():
        leg.dt = dt_c
    block = (IU.MountedBlock(dt=dt_c, q_machine=q_machine) if "J2" in joins
             else CL.BlockAgent(dt=dt_c))
    cl["BLOCK"] = block
    info["rad_ua"] = -math.log(cl["RAD"].decay) * CL.MDOT * CL.CP_COOLANT
    g_cl, _ = CL.build(experts=cl, measured=None)
    for a in g_cl.agents:
        if a.agent_id == "RAD" and "J1" in joins:
            a = replace(a, capabilities=IU.core_capabilities(cl["RAD"],
                                                             a.capabilities))
        if a.agent_id == "BLOCK" and "J2" in joins:
            a = replace(a, capabilities=IU.mounted_block_capabilities(
                block, a.capabilities))
        agents.append(a)

    # -- the powertrain's agents --------------------------------------------
    dt_p = GE.MACRO_DT if clocks == "reconciled" else PT.MACRO_DT
    if "J2" in joins:
        base = pt["MGU"]
        pt["MGU"] = IU.CooledMachine(current_op=info["current_op"],
                                     p_ref=info["p_ref"],
                                     t_case_ref=float(block.t_case),
                                     omega=base.omega)
        pt["MGU"].k_t = base.k_t
        pt["MGU"].k_e = base.k_e
        pt["MGU"].resistance = base.resistance
        info["t_case_ref"] = float(block.t_case)
    for e in pt.values():
        e.dt = dt_p
    g_pt, _ = PT.build(experts=pt, measured=None)
    for a in g_pt.agents:
        if a.agent_id == "ROTOR" and "J3" in joins:
            a = replace(a, capabilities=IU.host_rotor_capabilities(
                pt["ROTOR"], a.capabilities))
        if a.agent_id == "MGU" and "J2" in joins:
            a = replace(a, capabilities=IU.cooled_machine_capabilities(
                pt["MGU"], a.capabilities))
        agents.append(a)

    experts.update(cl)
    experts.update(pt)

    conns = connections(t, wing_window, sites, joins)
    conns += [c for c in g_cl.connections] + [c for c in g_pt.connections]
    families = {a.agent_id: a.capabilities.governing_family for a in agents}
    graph_axis = Decomposition.OVERLAPPING

    def axis(c):
        fa, fb = families[c.a[0]], families[c.b[0]]
        if fa != fb:
            return None
        own = (Decomposition.OVERLAPPING if fa == IU.FLUID
               else Decomposition.NON_OVERLAPPING)
        return None if own is graph_axis else own

    conns = [replace(c, cut_axis=axis(c)) for c in conns]
    graph = CaseGraph(
        name=name or ("racelab-" + f"{t.n_windows}w-"
                      + ("+".join(joins) if joins else "disjoint")
                      + ("" if clocks == "native" else f"-clocks-{clocks}")),
        agents=agents,
        connections=conns,
        decomposition=graph_axis,
        overlap={IU.FLUID: t.halo * DX},
        overlap_cells={IU.FLUID: t.halo},
        partition_of_unity={IU.FLUID: GE.projected_assembly(t)},
        global_fields=[GlobalField(
            "pressure", produced_by=tuple(t.names), applies_to=tuple(t.names),
            note="the elliptic part, in the composition layer where R10 puts "
                 "it: one global spectral Leray projection on the assembled "
                 "fluid field, once per exchange")],
        cross_points=() if t.is_single else ("mid",),
        loop_gains=tuple(lg for g in (g_cl, g_pt) for lg in g.loop_gains),
        macro_dt=GE.MACRO_DT,
        flux_matching=flux_matching,
        measured=None,
        note=(f"RaceLab phase 1: {t.n_windows} reference.WindowNS windows of "
              f"{t.wx}x{t.wy} cells over {t.nx}x{t.ny} at h = 1/64, halo "
              f"{t.halo}, laid out from the car's geometry; {len(bodies)} "
              f"immersed porous bodies on a rolling road; joined by "
              + (", ".join(joins) if joins else "nothing")))
    info["agent_count"] = len(agents)
    info["seam_count"] = len(conns)
    info["sites"] = {d: {"seam": s.seam, "left": s.left, "right": s.right,
                         "first_cell": s.first_cell}
                     for d, s in sites.items()}
    return graph, {"experts": experts, "info": info, "tiling": t,
                   "bodies": bodies, "sites": sites}


# ===========================================================================
# the march
# ===========================================================================


class RaceRollout(W.FSIRollout):
    """The car's fluid column: `FSIRollout`'s four operators, eleven bodies.

    **It subclasses and overrides `exchange` and `macro_step` and nothing else
    in the fluid.**  The cut, the partition of unity, the one global spectral
    Leray projection per exchange and the boundary band are CS-12's, which are
    CS-10's.  Every body is applied as a body force to the ASSEMBLED field
    before it is cut, exactly as `FlexWing.forcing` already applied the front
    wing's plate and as `vehicle_march.UnionRollout` already applied the two
    devices -- that is decision 5 (W94), inherited rather than re-taken.

    **The structure is rigid**, `motion=False`, which is what CS-18 marched and
    what `integration_union` compiles.  So the wetted and mount seams are
    DECLARED and not solved, the car's shape does not move, and the only
    couplings under test are the three joins and the tiling's own window seams.
    That is a limitation of this phase and is named on the page.
    """

    def __init__(self, tiling: "RaceTiling" = None,
                 objects: Sequence[Any] = None,
                 params: CarParams = None,
                 *, joins=("J1", "J2", "J3"), join_coupling: str | dict = "tight",
                 rotor_width: float = None, machine_scale: float | None = None,
                 n_join_inner: int = 3, p_ref: float | None = None,
                 null: str | None = None, wheel_omega: float | None = 0.0,
                 enforce: bool = True, host_inflow: float | None = None,
                 **kw) -> None:
        if tiling is None:
            tiling, _i = layout()
        kw.setdefault("coupling", "lagged")
        kw.setdefault("motion", False)
        super().__init__(tiling=tiling, **kw)
        self.tiling = tiling
        if objects is None:
            objects, _flat = car_bodies(params, device=self.device)
        self.objects = list(objects)
        self.joins = tuple(j for j in ("J1", "J2", "J3") if j in tuple(joins))
        if isinstance(join_coupling, str):
            join_coupling = {j: join_coupling for j in ("J1", "J2", "J3")}
        bad = [v for v in join_coupling.values() if v not in ("tight", "lagged")]
        if bad:
            raise ValueError(f"a join's coupling is 'tight' or 'lagged', not {bad}")
        self.join_coupling = dict(join_coupling)
        self.rotor_width = (HOST_ROTOR_WIDTH if rotor_width is None
                            else float(rotor_width))
        self.machine_scale = (self.rotor_width if machine_scale is None
                              else float(machine_scale))
        self.n_join_inner = int(n_join_inner)
        #: **Decision 8 (W222).**  ``host_inflow`` re-sizes the machine for
        #: the inflow its host actually delivers, by the same similarity
        #: W199 used when the disk's WIDTH moved.  ``None`` is the machine
        #: `powertrain` declares, which is what Tier 51 marched and what
        #: `MGU.validity` declines in this duct.
        self.host_inflow = host_inflow
        self.elements = (VM.machine_for_rotor(self.machine_scale)
                         if host_inflow is None else
                         machine_for_host(host_inflow, scale=self.machine_scale))
        self.p_ref = (IU.calibrated_p_ref(PT.MachineAgent())[0] if p_ref is None
                      else float(p_ref))
        self.null = null if null in ("J1", "J3") else None
        #: **The wheels do not rotate, and that is a MEASURED decision.**
        #: ``0.0`` (the default) is the car this tier declares; ``None`` is
        #: rolling without slip on the moving ground, which is the arm that
        #: says why the default is what it is.  `WheelBody`'s docstring has
        #: the mechanism and stage ``geometry`` has the numbers: on a
        #: twelve-sided polygon a rigid rotation projects onto the chord
        #: normals as ``omega s``, which is 17% of the free stream here and
        #: moved the field by 13% over eight macro-steps.  That is spurious
        #: blowing and suction on the wheel's surface, not a rotating wall,
        #: and a demo that showed it would be showing a discretisation.
        self.wheel_omega = wheel_omega
        for b in self.objects:
            if isinstance(b, WheelBody):
                b.omega = (b.rolling_omega() if wheel_omega is None
                           else float(wheel_omega))
        sites = device_sites(tiling)
        self.specs = tuple(
            RaceDeviceSpec(j, sites[d], lab, tiling)
            for j, d, lab in (("J1", "RAD", "the radiator core"),
                              ("J3", "ROTOR", "the recovery turbine"))
            if j in self.joins)
        self.forcing_ = RaceDeviceForcing(self.specs, tiling)
        self.state = VM.JoinState()
        self._fx_dev = np.zeros((self.ny, self.nx))
        self._fx_by_dev: dict[str, np.ndarray] = {}
        self._opt_t = dict(dtype=W.TORCH_DTYPE, device=self.device)
        self._core: IU.CoreRadiator | None = None
        self.join_residual: list = []
        self.work_trace: list[dict] = []
        self._exchange_i = 0
        self.body_force_calls = 0
        #: **W222.**  ``enforce`` ON is the default, because a check that
        #: is off by default is not a check.  OFF still RUNS the check and
        #: records what it would have said into `outside`.
        self.enforce = bool(enforce)
        self.outside: dict | None = None
        self.outside_first: dict | None = None
        self.outside_steps = 0
        self.envelope: dict | None = None

    # -- the car's body force ----------------------------------------------

    def car_forcing(self, u, v):
        """``(fx, fy, load, drag)``: every body's force on the fluid.

        Summed on the ASSEMBLED field before the cut, which is the one place a
        body force can go if a body that straddles a window boundary is to be
        seen consistently by both windows.  19% of the car's occupancy does
        straddle one, measured, which is why this matters here and did not on
        the front wing.
        """
        import torch
        fx = torch.zeros((self.ny, self.nx), **self._opt_t)
        fy = torch.zeros((self.ny, self.nx), **self._opt_t)
        load = torch.zeros((), **self._opt_t)
        drag = torch.zeros((), **self._opt_t)
        for b in self.objects:
            a, c, _w, l, d = b.forcing(u, v, self.ny, self.nx)
            fx, fy, load, drag = fx + a, fy + c, load + l, drag + d
        self.body_force_calls += 1
        return fx, fy, load, drag

    # -- reading the rings -------------------------------------------------

    def ring(self, u, dev: RaceDeviceSpec) -> np.ndarray:
        import torch
        w = u.detach().cpu().numpy() if torch.is_tensor(u) else np.asarray(u)
        return np.asarray(w, dtype=float)[dev.rows, dev.i_up].copy()

    # -- the joins ---------------------------------------------------------

    def refresh(self, u, step: int, which=("J1", "J3")) -> None:
        """`vehicle_march.UnionRollout.refresh`, on the car's devices.

        Decision 6 (W196) unchanged: the operating point is solved ACROSS THE
        UNION at the inflow the host ring actually carries, every time this is
        called, and the thrust that comes back is what the fluid is forced with.
        """
        s = self.state
        fx = np.zeros((self.ny, self.nx))
        for dev in self.specs:
            if dev.join == "J3" and "J3" in which:
                ring = self.ring(u, dev)
                res, _els, disk = VM.operating_point(
                    ring, width=self.rotor_width, scale=self.machine_scale,
                    elements=self.elements)
                s.u_rotor = float(np.mean(ring))
                s.induction = float(res.induction)
                s.omega = float(res.omega)
                s.current = float(res.current)
                s.thrust_rotor = float(disk.thrust)
                s.shaft_power = float(disk.power)
                s.rotor_valid = bool(res.rotor_valid)
                if "J2" in self.joins:
                    s.q_machine = float(s.current ** 2
                                        * self.elements["MGU"].resistance
                                        * self.p_ref)
                self.forcing_.last[dev.site.device] = {"u_disk": s.u_rotor}
            elif dev.join == "J1" and "J1" in which:
                ring = self.ring(u, dev)
                s.u_core = float(np.mean(ring))
                if self._core is None:
                    self._core = IU.CoreRadiator(u_air=ring)
                core = self._core
                core.u_air = ring
                s.ua = (CL.UA_RAD if self.null == "J1" else float(core.ua))
                tau = core.traction(ring)
                s.thrust_core = float(np.sum(tau) * DX)
                s.core_flow_power = float(np.sum(tau * ring) * DX)
                self.forcing_.last[dev.site.device] = {"u_disk": s.u_core}
        by_dev: dict[str, np.ndarray] = {}
        for dev in self.specs:
            if dev.join == "J3" and self.null == "J3":
                continue
            if dev.join == "J3" and s.thrust_rotor:
                by_dev[dev.site.device] = self.forcing_.field(
                    s.thrust_rotor, dev, s.u_rotor)
            elif dev.join == "J1" and s.thrust_core:
                by_dev[dev.site.device] = self.forcing_.field(
                    s.thrust_core, dev, s.u_core)
        for f in by_dev.values():
            fx = fx + f
        self._fx_by_dev = by_dev
        self._fx_dev = fx
        s.step = int(step)


    # -- the envelope, consulted (W222) ------------------------------------

    def validity_report(self, u=None, v=None) -> dict:
        """Every DECLARED validity predicate on this graph, at this state.

        **The predicates are not new and that is the point.**
        `powertrain.MachineAgent.validity` already declares the condition Tier
        51 walked past -- *"a generator can only push current into the battery
        while its back-EMF exceeds the open-circuit voltage; below that speed
        the loop current reverses and the machine MOTORS, which is a legitimate
        mode and a different one from the mode this graph declares, so the
        record declines rather than reporting a negative generated power as if
        it were generation."*  The disk declares its induction clamp and the
        fluid window declares its cell-Reynolds bound.  **All three existed and
        none was consulted.**

        Three states per expert, never two: ``True`` inside, ``False``
        declined, ``None`` **not consultable**.  A predicate that cannot be
        evaluated is not a pass.
        """
        import torch
        out: dict[str, Any] = {}
        s = self.state

        # -- the machine: its own declared predicate, at the shaft speed the
        # -- union's operating point actually solved
        mgu = self.elements.get("MGU")
        if mgu is None or "J3" not in self.joins:
            out["MGU"] = {"valid": None,
                          "why": "no J3 in this union, so no shaft speed"}
        else:
            try:
                ok = bool(mgu.validity(np.full(1, s.omega)))
                out["MGU"] = {
                    "valid": ok, "omega": float(s.omega),
                    "back_emf": float(mgu.k_e * s.omega),
                    "V_oc": float(PT.V_OC),
                    "current": float(s.current),
                    "why": ("the back-EMF exceeds the open-circuit voltage"
                            if ok else
                            "the back-EMF %.6g is at or below the battery's "
                            "open-circuit voltage %.6g, so the loop current "
                            "reverses and the machine MOTORS -- a legitimate "
                            "mode, and not the one this graph declares"
                            % (mgu.k_e * s.omega, PT.V_OC))}
            except Exception as exc:                         # pragma: no cover
                out["MGU"] = {"valid": None, "why": "not consultable: %s" % exc}

        # -- the disk: `clamp_induction` is its own declared envelope
        if "J3" not in self.joins:
            out["ROTOR"] = {"valid": None, "why": "no J3 in this union"}
        elif s.step < 0:
            out["ROTOR"] = {"valid": None, "why": "no operating point solved yet"}
        else:
            out["ROTOR"] = {
                "valid": bool(s.rotor_valid), "induction": float(s.induction),
                "why": ("the induction is inside the disk's clamp" if s.rotor_valid
                        else "the induction %.6g is AT the clamp `clamp_induction` "
                             "declares, so the disk has no operating point here"
                             % s.induction)}

        # -- the fluid windows: the cell-Reynolds bound `FSIFlowWindow.validity`
        # -- declares, evaluated on the assembled field
        if u is None or v is None:
            out["FLUID"] = {"valid": None, "why": "no field handed in"}
        else:
            with torch.no_grad():
                umax = float(torch.max(torch.hypot(u, v)))
            finite = bool(torch.isfinite(u).all() and torch.isfinite(v).all())
            re_h = self.solver.h * umax / self.nu
            ok = bool(finite and re_h <= 8.0 and umax <= U_MAX_BAND)
            out["FLUID"] = {
                "valid": ok, "u_max": umax, "cell_reynolds": re_h,
                "finite": finite, "band": U_MAX_BAND,
                "why": ("inside the window expert's cell-Reynolds bound"
                        if ok else
                        "u_max = %.6g gives a cell Reynolds number of %.4g "
                        "against the window expert's declared bound of 8, or "
                        "went past the declared band of %.4g -- and WindowNS sizes "
                        "its sub-step count from u_max, so a march past this "
                        "reads as a hang rather than as a blow-up"
                        % (umax, re_h, U_MAX_BAND))}

        # -- the radiator core, if J1 is in
        if "J1" not in self.joins or self._core is None:
            out["RAD"] = {"valid": None, "why": "no J1 in this union"}
        else:
            try:
                out["RAD"] = {"valid": bool(self._core.validity()),
                              "ua": float(s.ua)}
            except Exception as exc:                         # pragma: no cover
                out["RAD"] = {"valid": None, "why": "not consultable: %s" % exc}

        out["_declined"] = sorted(k for k, r_ in out.items()
                                  if not k.startswith("_")
                                  and r_.get("valid") is False)
        out["_unconsultable"] = sorted(k for k, r_ in out.items()
                                       if not k.startswith("_")
                                       and r_.get("valid") is None)
        return out

    def check_envelopes(self, step: int, u=None, v=None) -> dict:
        """Raise `EnvelopeDeclined` if any declared predicate declines.

        **W222.**  Tier 51 marched 600 macro-steps with `MGU.validity` False and
        nothing said so, which is PoC 2's W145 on a second subsystem: the flag
        existed, was computed, was recorded into `JoinState` -- and was read by
        nothing.
        """
        rep = self.validity_report(u, v)
        bad = rep["_declined"]
        if bad:
            why = "; ".join("%s: %s" % (k, rep[k].get("why", "declined"))
                            for k in bad)
            raise EnvelopeDeclined(
                "the march left a declared envelope at macro-step %d -- %s"
                % (step, why), rep)
        return rep

    def _absorb(self, step: int, u, v) -> None:
        """The ONE place every path funnels through, PoC 2's `Engine._absorb`.

        ``enforce`` does not remove the check.  With it off, the check still
        runs and what it WOULD have said is recorded into `outside`, so a caller
        can report the number beside the reason the model declines to stand
        behind it -- which is the only way to publish a measurement of a graph
        that is out of envelope without publishing it as if it were in.
        `outside_first` is sticky: a march that left the envelope at step 4 and
        came back at step 40 still left it.
        """
        try:
            rep = self.check_envelopes(step, u, v)
            self.outside = None
            self.envelope = rep
        except EnvelopeDeclined as exc:
            if self.enforce:
                raise
            self.outside = {"step": int(step), "why": str(exc)[:400],
                            "declined": list(exc.report["_declined"])}
            self.envelope = exc.report
            self.outside_steps += 1
            if self.outside_first is None:
                self.outside_first = dict(self.outside)

    # -- the exchange ------------------------------------------------------

    def _advance(self, u, v):
        import torch
        fx, fy, load, drag = self.car_forcing(u, v)
        fdev = torch.as_tensor(self._fx_dev, **self._opt_t)
        fx = fx + fdev
        us, vs = self.cut(u), self.cut(v)
        fxs, fys = self.cut(fx), self.cut(fy)
        u1, v1 = self.solver.step_batch(us, vs, self.dt_ex, bc0=None,
                                        force=(fxs, fys))
        self.substep_log.append(int(self.solver.last_substeps))
        au, av = self.blend(u1, v1)
        au, av = self.project(au, av)
        bu, bv = self.band(au, av)
        return bu, bv, load, drag, fdev

    def exchange(self, u, v):
        """`UnionRollout.exchange` on the car: tight iterates, lagged holds."""
        import torch
        tight = [j for j in ("J1", "J3")
                 if j in self.joins and self.join_coupling.get(j) == "tight"]
        u0 = u
        if tight:
            prev, r_first, r_last = None, None, 0.0
            for _ in range(max(1, self.n_join_inner) + 1):
                bu, bv, load, drag, fdev = self._advance(u, v)
                self.refresh(bu, self.state.step, which=tight)
                cur = np.array([self.state.u_rotor, self.state.u_core])
                if prev is not None:
                    r_last = float(np.linalg.norm(cur - prev))
                    r_first = r_last if r_first is None else r_first
                prev = cur
            bu, bv, load, drag, fdev = self._advance(u, v)
            self.join_residual.append((float(r_first or 0.0), r_last))
        else:
            bu, bv, load, drag, fdev = self._advance(u, v)
            self.join_residual.append((0.0, 0.0))
        with torch.no_grad():
            a_ = float((fdev * u0).sum() * DX * DX)
            b_ = float((fdev * bu).sum() * DX * DX)
            per = {}
            for nm, f in self._fx_by_dev.items():
                ft = torch.as_tensor(f, **self._opt_t)
                pa = float((ft * u0).sum() * DX * DX)
                pb = float((ft * bu).sum() * DX * DX)
                per[nm] = 0.5 * (pa + pb)
        self.work_trace.append({
            "exchange": self._exchange_i, "step": self.state.step,
            "power_on_the_fluid_start": a_, "power_on_the_fluid_end": b_,
            "power_on_the_fluid_mid": 0.5 * (a_ + b_),
            "shaft_power": self.state.shaft_power,
            "core_flow_power": self.state.core_flow_power,
            "per_device": per})
        self._exchange_i += 1
        return bu, bv, load, drag

    def macro_step(self, u, v, step: int = 0):
        lagged = [j for j in ("J1", "J3")
                  if j in self.joins and self.join_coupling.get(j) == "lagged"]
        self.state.step = int(step)
        if lagged or self.state.step <= 0:
            self.refresh(u, step, which=tuple(lagged) if lagged else ())
        load = drag = None
        for _ in range(GE.EXCHANGES):
            u, v, load, drag = self.exchange(u, v)
        #: **W222: the one funnel.**  Every path into this rollout goes
        #: through `macro_step`, so the check goes here and nowhere else.
        self._absorb(step, u, v)
        return u, v, load, drag


def march(u0: np.ndarray = None, v0: np.ndarray = None, steps: int = 200,
          tiling: "RaceTiling" = None, objects: Sequence[Any] = None,
          joins=("J1", "J2", "J3"), join_coupling="tight",
          null: str | None = None, enforce: bool = True,
          host_inflow: float | None = None,
          n_per_coolant: int = VM.N_FLUID_PER_COOLANT,
          progress=None, **kw) -> VM.UnionMarch:
    """March the car: the fluid on its clock, the coolant circuit sub-cycled.

    **The result is a `vehicle_march.UnionMarch`**, deliberately: CS-18's
    `receiver_balances`, `rotor_power_paths`, `crossing_vector` and
    `composition_error` then read this march with no change at all, so what
    phase 1 reports about the joins is the same function of the same trace keys
    that produced [[case-study-vehicle-march-atlas-0.1]]'s numbers.  Nothing is
    re-derived and nothing can drift.

    ``null`` removes ONE join's term and leaves everything that reads it in
    place -- CS-18 section 3.4's rule, because a null arm that cannot fail is
    not a control.

    **Every declared envelope is CONSULTED, once, at the end of each
    macro-step** (W222).  `RaceRollout._absorb` is the single funnel, as
    PoC 2's `Engine._absorb` is: `MachineAgent.validity`, the disk's
    induction clamp and the fluid window's cell-Reynolds bound all
    existed before this tier and none of them was read, which is how Tier
    51 marched 600 macro-steps with the machine motoring.  ``enforce``
    ON is the default; OFF still runs the check and records what it would
    have said, so a measurement of an out-of-envelope graph can be
    published WITH the stamp rather than as if it were in.

    ``host_inflow`` is decision 8: re-size the machine for the inflow its
    host actually delivers (`machine_for_host`).  ``None`` leaves the
    machine `powertrain` declares, which is the Tier 51 configuration.

    **A stalled march is a blow-up.**  `WindowNS` sets its sub-step count
    from ``u_max``, so a diverging field reads as a hang rather than as an
    error; that bound is one of the predicates `validity_report` consults.
    """
    import time
    import torch
    if tiling is None:
        tiling, _i = layout()
    r = RaceRollout(tiling=tiling, objects=objects, joins=joins,
                    join_coupling=join_coupling, null=null,
                    enforce=enforce, host_inflow=host_inflow, **kw)
    opt = dict(dtype=W.TORCH_DTYPE, device=r.device)
    u = (torch.full((r.ny, r.nx), GE.U_INF, **opt) if u0 is None
         else torch.as_tensor(np.asarray(u0), **opt))
    v = (torch.zeros((r.ny, r.nx), **opt) if v0 is None
         else torch.as_tensor(np.asarray(v0), **opt))
    r.substep_log = []

    block = (IU.MountedBlock(q_machine=0.0)
             if ("J2" in r.joins and null != "J2") else CL.BlockAgent())
    loop = CL.LoopSolve(block=block)
    t_in = CL.T_COOLANT_0

    keys = ("u_rotor", "u_core", "ua", "induction", "omega", "current",
            "thrust_rotor", "shaft_power", "thrust_core", "core_flow_power",
            "q_machine", "load", "drag", "u_max", "power_on_the_fluid")
    trace: dict[str, list] = {kk: [] for kk in keys}
    for extra in ("t_wall", "power_rotor", "power_core",
                  "u_at_the_rotor_plane"):
        trace[extra] = []
    coolant: list[dict] = []
    t0 = time.perf_counter()
    r.refresh(u, 0, which=tuple(j for j in ("J1", "J3") if j in r.joins))
    if "J1" in r.joins and null != "J1" and r._core is not None:
        loop.legs["RAD"] = r._core
    rotor_spec = next((d for d in r.specs if d.join == "J3"), None)
    i_pl = (int(round(rotor_spec.x_plane / DX - 0.5)) if rotor_spec else 0)
    for s in range(steps):
        n_before = len(r.work_trace)
        u, v, load, drag = r.macro_step(u, v, s)
        st = r.state
        wk = r.work_trace[n_before:]
        for kk in ("u_rotor", "u_core", "ua", "induction", "omega", "current",
                   "thrust_rotor", "shaft_power", "thrust_core",
                   "core_flow_power", "q_machine"):
            trace[kk].append(float(getattr(st, kk)))
        trace["load"].append(float(load.detach()))
        trace["drag"].append(float(drag.detach()))
        umax = float(torch.max(torch.hypot(u, v)).detach())
        trace["u_max"].append(umax)
        trace["power_on_the_fluid"].append(
            float(np.mean([x["power_on_the_fluid_mid"] for x in wk])) if wk else 0.0)
        trace["power_rotor"].append(float(np.mean(
            [x["per_device"].get("ROTOR", 0.0) for x in wk])) if wk else 0.0)
        trace["power_core"].append(float(np.mean(
            [x["per_device"].get("RAD", 0.0) for x in wk])) if wk else 0.0)
        if rotor_spec is not None:
            trace["u_at_the_rotor_plane"].append(float(
                u.detach().cpu().numpy()[rotor_spec.rows, i_pl].mean()))
        else:
            trace["u_at_the_rotor_plane"].append(float("nan"))
        #: finiteness and the speed band are `validity_report`'s FLUID
        #: predicate, consulted by `_absorb` at the end of every
        #: `macro_step` -- so this loop no longer carries its own copy.
        if (s + 1) % n_per_coolant == 0:
            if isinstance(block, IU.MountedBlock):
                block.q_machine = float(st.q_machine)
                tw = float(np.mean(block._face_T(block._T, "outer")))
                block.t_case = tw + block.q_machine / (block.h_mount * block.area)
            row = VM._coolant_step(loop, block, t_in)
            t_in = row["t_in"]
            row["fluid_step"] = s
            coolant.append(row)
        trace["t_wall"].append(float(np.mean(block._face_T(block._T, "inner"))))
        if progress is not None:
            progress(s, time.perf_counter() - t0)
    return VM.UnionMarch(
        steps=steps, join_coupling=dict(r.join_coupling), joins=tuple(r.joins),
        trace={kk: np.asarray(vv) for kk, vv in trace.items()},
        coolant=coolant, u=u.detach().cpu().numpy().copy(),
        v=v.detach().cpu().numpy().copy(), block_T=block._T.copy(), t_in=t_in,
        wall_s=time.perf_counter() - t0,
        thickness=r.forcing_.thickness_report(),
        join_residual=np.asarray(r.join_residual),
        notes={"null": null, "n_per_coolant": n_per_coolant,
               "rotor_width": r.rotor_width, "machine_scale": r.machine_scale,
               "p_ref": r.p_ref, "n_join_inner": r.n_join_inner,
               "coolant_steps_taken": len(coolant),
               "n_windows": tiling.n_windows,
               "body_force_calls": r.body_force_calls,
               "substeps_max": max(r.substep_log) if r.substep_log else None,
               "device_ring_to_plane_cells": {
                   d.site.device: d.ring_to_plane_cells for d in r.specs},
               "enforce": bool(enforce),
               "host_inflow": host_inflow,
               "outside_the_envelope": r.outside_first,
               "outside_the_envelope_steps": r.outside_steps,
               "envelope_at_the_end": r.envelope})



# ===========================================================================
# the spin-up: a release state the experts stand behind (W222, W223)
# ===========================================================================

#: How many lagged macro-steps the car takes to get from a uniform freestream
#: to a state every declared envelope admits.  Measured, not guessed:
#: `scripts/tier52_racelab_switch.py` stage ``spinup`` reports the last
#: macro-step at which any predicate declined, and this is that plus a margin.
N_SPIN = 240


def settled_field(steps: int = N_SPIN, host_inflow: float | None = None,
                  tiling: "RaceTiling" = None, objects: Sequence[Any] = None,
                  joins=("J1", "J2", "J3"), progress=None, **kw):
    """March from the freestream to a state the experts admit, and return it.

    **Why this exists, and it is inherited practice rather than a new idea.**
    CS-18 releases its arms from ``out/w141/settled.npz``; `atlas/demo_frontwing`
    says in its README that without the settled cache *"the demo releases from
    the freestream, says so on screen, and shows a transient that no number on
    the results page was measured at."*  **Tier 51 released every arm from a
    uniform freestream and did not say so**, and the envelope check added in
    Tier 52 found it immediately: from ``u = U_inf`` everywhere, the flow
    through the duct is still at the free stream, the machine over-generates and
    the disk's induction hits the UPPER edge of `clamp_induction`.

    So the spin-up runs with ``enforce=False`` -- the check still runs, and what
    it would have said is recorded and returned -- and the measured arms release
    from the state it ends at, with ``enforce=True``.  **The transient is
    declared rather than marched through in silence.**

    Returns ``(u, v, report)`` with the report carrying the last macro-step at
    which any predicate declined and which ones, so the caller can assert the
    release state is admitted rather than assume it.
    """
    import torch
    if tiling is None:
        tiling, _i = layout()
    r = RaceRollout(tiling=tiling, objects=objects, joins=joins,
                    join_coupling="lagged", enforce=False,
                    host_inflow=host_inflow, **kw)
    opt = dict(dtype=W.TORCH_DTYPE, device=r.device)
    u = torch.full((r.ny, r.nx), GE.U_INF, **opt)
    v = torch.zeros((r.ny, r.nx), **opt)
    r.refresh(u, 0, which=tuple(j for j in ("J1", "J3") if j in r.joins))
    declined_at: list[int] = []
    who: dict[str, int] = {}
    for s in range(steps):
        u, v, _load, _drag = r.macro_step(u, v, s)
        if r.outside is not None:
            declined_at.append(s)
            for k in r.outside["declined"]:
                who[k] = who.get(k, 0) + 1
        if progress is not None:
            progress(s, 0.0)
    rep = {
        "steps": steps,
        "macro_steps_outside": len(declined_at),
        "last_macro_step_outside": (max(declined_at) if declined_at else None),
        "first_macro_step_outside": (min(declined_at) if declined_at else None),
        "which_experts_declined": who,
        "release_state_is_admitted": r.outside is None,
        "envelope_at_the_release_state": r.envelope,
        "host_inflow": host_inflow,
        "u_rotor_at_the_release_state": float(r.state.u_rotor),
        "note": "the spin-up runs with enforce=False BY DESIGN -- a march that "
                "stops at macro-step 3 of a transient reports nothing -- and "
                "the measured arms release from what it ends at with "
                "enforce=True",
    }
    return (u.detach().cpu().numpy().copy(),
            v.detach().cpu().numpy().copy(), rep)

# ===========================================================================
# the gate, PRE-REGISTERED
# ===========================================================================

#: **Phase 1's gate, as numbers, fixed before any arm ran.**
#:
#: Written after the instrument run of `scripts/tier51_racelab_graph.py` stage
#: ``calib`` -- 40 macro-steps tight, its bitwise repeat, the settling report
#: and the cost -- and BEFORE the arms at the horizon.  That is
#: `defect-correction-learned-operator` section 7's discipline.
#:
#: **Every threshold is CS-18's own**, inherited rather than chosen here, so
#: that none of them is this tier's answer read backwards.  The one clause that
#: is new is P6, and its number is the requirements' section 10 ceiling written
#: before this project existed.
#:
#: **Its weakness, named.**  The car is one geometry and the horizon is one
#: horizon; "out of sample" here means a different graph from CS-18's, not a
#: different rung.  What makes that worth something is that the graph really is
#: different -- 26 agents against 18, 14 fluid windows against 6, a tiling
#: derived from geometry rather than a 3x2 grid -- so a threshold met on both
#: is a statement about the joins rather than about the front wing's layout.
GATE: dict[str, Any] = {
    "P1_compile": {
        "measured_as": "the joined union's refusals, and the disjoint union's",
        "passes_if": "the joined union refuses at L7/R9 and nothing else, and "
                     "the disjoint union refuses nothing",
        "expected_refusals_joined": ["L7/R9"],
        "expected_refusals_disjoint": [],
        "why_these": "CS-18 section 4.5's table, on the front wing's graph.  "
                     "Reproducing it on a different graph says the reason is "
                     "the clocks and not the layout",
    },
    "P2_J3_receiving_balance": {
        "measured_as": "the work the turbine's body force does on the fluid "
                       "over the settle window, against the shaft power the "
                       "union's operating point claims",
        "tol_with": 0.075,
        "tol_without": 0.5,
        "passes_if": "relative residual <= tol_with with the term and "
                     ">= tol_without with the term withheld (the null arm)",
        "why_this": "CS-18 section 5's G1, unchanged: 0.075 is the inflow "
                    "non-uniformity rung9-gate-restated section 4.2 measured "
                    "and declared a property of the model",
    },
    "P3_J1_parametric": {
        "measured_as": "UA against the declared exponent's prediction from the "
                       "measured air; the null pins UA at UA_RAD",
        "tol_with": 1e-06,
        "tol_without": 0.0,
        "passes_if": "tracking residual <= tol_with with the term, and UA "
                     "exactly constant without it while the air moves",
        "why_this": "CS-18 section 5's G3: an algebraic identity's tolerance",
    },
    "P4_J2_receiving_balance": {
        "measured_as": "the block's first law over the coolant circuit's own "
                       "march, with the mount term and without it",
        "tol_with": 1e-06,
        "tol_without": 0.01,
        "passes_if": "<= tol_with with, >= tol_without without",
        "why_this": "CS-18 section 5's G2, unchanged",
    },
    "P5_repeat_floor": {
        "measured_as": "the referent marched twice as two independent "
                       "constructions in one process",
        "passes_if": "bitwise identical in every crossing quantity",
        "why_this": "CS-18 section 5's G6.  A difference below the "
                    "instrument's floor is not a signal",
    },
    "P6_macro_step_cost": {
        "measured_as": "the wall time of one macro-step of the WHOLE march -- "
                       "the car's thirteen bodies, the two devices and the "
                       "composition layer -- at the chosen box",
        "ceiling_s": 0.5,
        "passes_if": "the lagged column's macro-step is under the ceiling",
        "why_this": "POC3-RACELAB-REQUIREMENTS section 10's first named risk, "
                    "written before this tier",
        "note": "the TIGHT column is five solver calls an exchange and is not "
                "expected to clear it; which column a dashboard would march is "
                "the question this clause is really about",
    },
    "P7_body_force_conserves": {
        "measured_as": "sum(f dA) against minus the force on each body, over "
                       "every one of the car's bodies at the release state",
        "tol": 1e-12,
        "passes_if": "the worst relative residual is under tol",
        "why_this": "CS-18 decision 5 (W94): a discretization losing 3% of the "
                    "force would read as a 3% interface residual at a join and "
                    "be blamed on the join",
    },
}

#: **The prediction, recorded with the gate and before the arms.**
PREDICTION: tuple[str, ...] = (
    "P1 passes.  The car's graph has the same three clocks in the same three "
    "unit systems as CS-18's, and L7/R9 compares bare numbers, so the refusal "
    "is structural; the disjoint union has no multirate seam and W194's "
    "narrowing should leave it alone.",
    "P2 passes, at a residual LARGER than the 0.0221 the 40-step instrument "
    "showed, and for CS-18's reason: the turbine's own wake deepens over a "
    "longer horizon and the whole residual is the gap between the ring the "
    "disk reads and the cell its force sits in.  That ring-to-plane distance "
    "is 7.5 cells here, which is the front wing's to the digit, so the "
    "residual should land near CS-18's 0.0399 rather than far from it.",
    "P3 and P4 pass, and P3's null arm leaves the fluid BITWISE identical, "
    "because J1 carries no power and its term is a dependence.",
    "P5 passes.  The 40-step instrument was already bitwise.",
    "P6 passes on the lagged column and FAILS on the tight one.  The lagged "
    "macro-step was 0.333 s at the instrument and the tight one 1.687 s, and "
    "neither number moves with the horizon.  So the honest reading is that "
    "the interactive column is the lagged one and phase 3 has to say so.",
    "P7 passes at machine precision; the geometry stage already measured "
    "2.1e-16 at the release state and the arms do not change the kernel.",
    "The composition error between the tight and lagged columns will be "
    "DOMINATED by q_machine and the loop current rather than by anything in "
    "the fluid, because the machine sits just above the battery's open-circuit "
    "voltage and CS-18 section 7.2 measured that elasticity at 63.3.  The "
    "40-step instrument already shows a 0.52% band in u_rotor becoming a 36% "
    "band in q_machine, a ratio of 69.  That is a REPRODUCTION of CS-18's "
    "finding on a different graph, not a new one.",
)
