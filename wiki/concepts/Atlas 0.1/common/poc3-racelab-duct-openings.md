# The duct opened: the car's cooling air on solid walls

*Case study, PoC 3. New 2026-09-15. Tier 63.*

---

# 0. The result, in one paragraph

**Tier 62's solid car sealed its own radiator duct, and the duct flowed backwards (W278). The user decided the solids rule should cut the openings — an inlet in the body shell ahead of the duct and an outlet behind it, sized by editable numbers — and that the duct's flow be measured before the radiator core and the turbine go in. With the openings the duct flows forward: $+0.090$ at both device planes over $t\in[12, 16]$, against Tier 62's $-0.109$ and the porous column's $+0.457$ on the same drawing.** That is a fifth of the porous column's flow and just under the floor registered for it ($0.10$), so D2 failed; the five other predictions held. **The flow had not settled**: it read $0.121$ from $t = 2$ to $6$ and has fallen steadily since, to $0.088$ over the last two units (W281). The openings split the body shell into three solids, left the march as sound as the sealed pod's, and raised the car's lift from $+1.83$ to $+2.94$, most of the rise on the duct's own plates.

---

# 1. What was decided

After Tier 62, the user was shown the sealed pod's reversed duct and chose among four repairs: openings cut by the rule, openings drawn in the editor, moving the devices to where the solid car already has forward flow, and accepting the reversed flow. **They chose the rule** — recorded in `POC3-RACELAB-REQUIREMENTS.md` §13.2 — **and that the duct be measured before the devices are added.** This tier is that measurement, and nothing else changes: the front wing is still Tier 62's filled wedge (W275, the other decision, a tier of its own).

---

# 2. The openings

Each opening removes a stretch of a panel's centreline before the panel is thickened, so both of its edges are round caps — a test pins that the area lost is the stretch's rectangle *less* two half-disc caps, which a square cut would not give back:

| role | plate | stretch of the centreline | length | where |
|---|---|---|---|---|
| inlet | `CHASSIS` | $0.25$ to $0.75$ | $19.2$ cells | the forward-facing slope rising at $32°$ ahead of the duct's entry |
| outlet | `POD_UP` | $0.80$ to $1.00$ | $20.1$ cells | the end of the pod's top, just behind the duct's exit at $x = 416.5$ |

Both are entries in `car_solids.SOLIDS_DEFAULT["openings"]` and can be overridden from `car_geometry.json`. **These positions and sizes are this tier's choice**, made before anything was marched on them; the rule and the decision to measure first are the user's.

**Tier 62's record still reproduces.** Its script now pins `openings` to none, so a re-run builds the sealed pod the record describes, and a test holds it there.

**What the rule made.** Tier 62's one body shell became three solids — the nose with the front of `CHASSIS` (no fillet needed), the middle of the body from the back of `CHASSIS` to the front of `POD_UP` (the ladder chose $4.5$ cells, $8.2$ cells² added), and the tail with the rear wing and the diffuser ($6$ cells, $67.6$ added). The composite has $377{,}267$ unknowns on twelve grids; a manufactured pressure on it errs by at most $3.0\times10^{-3}$, and the uniform stream stays uniform to $1.6\times10^{-14}$.

---

# 3. The march

Tier 62's march unchanged — the walls ramped to rest from the stream, $\Delta t = 0.0125$, to $t = 16$, statistics over $[12, 16]$ — on battery, beside the user's two phone-remote servers. **The laptop entered standby from 15:32 to 17:10 in the middle of it**, so its $8112$ s of wall time are not a cost; the median step was $1.25$ s.

## 3.1 The duct

| | sealed (Tier 62) | **opened** | porous column |
|---|---|---|---|
| mean streamwise velocity at the turbine plane, $[12, 16]$ | $-0.109$ | $\mathbf{+0.090}$ | $+0.457$ |
| at the radiator core's plane | $-0.109$ | $+0.090$ | — |
| core-to-turbine flux mismatch | $2\times10^{-5}$ | $9.5\times10^{-5}$ | — |

**Forward, and weak.** Over successive two-unit windows the opened duct reads $0.121$, $0.121$, $0.113$, $0.102$, $0.095$, $0.091$, $0.088$: it rose to $0.12$ as the start's uniform stream drained from the pod, held there to $t = 6$, and has fallen since without levelling. A sixteen-unit march did not reach the pod's own settled state, and whether it settles above zero is not known (W281).

> **[AI Inference]:** the pod is a large cavity with four openings — the new inlet and outlet, and the clearances round both wheels — and the duct is a narrow channel inside it that the cavity's flow can bypass above and below. A cavity that slow to settle suggests a recirculation growing inside it rather than a through-flow finding its level. The pressures at the four openings, measured over time, would say which openings feed the duct and which drain it; that is the first measurement W281 needs.

## 3.2 Soundness

Largest speed $3.37$ during the ramp; largest divergence over the window $0.031$, on the rear road patch beside the road; net flux through the contour round the car $0.16\%$ of its inflow; at most $7$ BiCGSTAB iterations a component after $t = 1$; drag spread $1.0\%$ of its mean. **D5 held** — the opened car marches as soundly as the sealed one.

## 3.3 The forces

| | sealed (Tier 62) | **opened** |
|---|---|---|
| drag $F_x$ | $2.736$ | $\mathbf{2.913}$ |
| vertical $F_y$ (positive up) | $+1.834$ | $\mathbf{+2.940}$ |
| front wing and nose tip, $F_y$ | $+1.419$ | $+1.426$ |
| duct plates, $F_y$ | $+0.271$ | $+0.711$ |
| upper duct plate alone, $F_y$ | $+0.133$ | $+0.570$ |
| front wheel, $F_x$ | $1.869$ | $1.670$ |

**The front wing's lift is unchanged** ($+1.43$), as D6 expected. **The rise in lift is the pod's**: the upper duct plate goes from $+0.13$ to $+0.57$, and the middle of the body, now on its own, carries $-0.23$ where Tier 62's whole body shell carried $-0.90$. An open pod with air moving through it loads its internal plates.

---

# 4. The predictions

Registered at 15:17:55, before anything was marched on the opened car.

| id | claim | outcome |
|---|---|---|
| D1 | the duct's mean streamwise velocity at the turbine plane over $[12, 16]$ is positive | **held** — $+0.090$ |
| D2 | that mean is at least $0.10$, a fifth of the porous column's $0.457$ | **failed** — $0.0895$, and still falling |
| D3 | and at most $0.457$ | **held** |
| D4 | core-plane and turbine-plane fluxes agree to $2\%$ | **held** — $9.5\times10^{-5}$ |
| D5 | speed $\le 4$, divergence $\le 0.2$, contour flux $\le 1\%$, drag spread $\le 10\%$ | **held** — $3.37$, $0.031$, $0.16\%$, $1.0\%$ |
| D6 | the car's mean vertical force is still positive | **held** — $+2.940$ |

---

# 5. What this tier did NOT do, named

- **The duct's flow did not settle** (W281), and the pressures at the pod's four openings were not measured.
- **No other opening size or position was tried**; the two in the rule are the only ones marched.
- **No radiator core, turbine, join or agent is in the duct** — the next tier's, now that the duct flows forward.
- **The front wing is still Tier 62's filled wedge** (W275); the grid generator the user chose to keep it as drawn is not built. Two quick attempts at it — a thickness cap where the smoothed normals converge, and a grid serving its own outer ring from its far wall — still folded the front wing and the tail and left orphans in the middle of the body, and are not in this commit.
- **No cost was taken on mains, and the march's wall time includes 98 minutes of standby.**
- **RaceLab's column, its records, the gate, the demo, the bundle and `car_geometry.json` are unchanged.**
- **Nothing was downloaded or installed**, no machine was rented, the unlicensed structural checkpoint was not loaded, and nothing was pushed.

```
python scripts/tier63_duct_openings.py --out out/racelab12 --stages car,march,compare,summary
```

The record is `out/racelab12/racelab12.json`; the march's trajectory, `out/racelab12/march.json`, is not carried upstream.

---

## See Also

- [[poc3-racelab-car-solids]] — Tier 62, the sealed pod whose duct flowed backwards, and the machinery this tier reuses.
- [[poc3-racelab-outlet-and-start]] — Tier 59, the porous column's settled rotor inflow.
- [[poc3-racelab-overset-flow]] — Tier 61, the flow solver.
- [[gap-worklist]] — W278 and W281.
