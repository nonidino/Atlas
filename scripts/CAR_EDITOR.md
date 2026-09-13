# Editing the car yourself

The car's shape is no longer buried in Python. It lives in
**`atlas/cases/car_geometry.json`**, and `atlas/cases/racelab.py` reads that
file. Change the file, the car changes — no code editing, no re-tracing.

---

## Run it

```bash
python scripts/car_editor.py --open
```

That opens `http://127.0.0.1:8017/`. Leave the terminal running; `Ctrl-C` stops
it. Nothing else in the project imports this — it is a tool, and it writes
exactly one file.

---

## What you are looking at

The car in **cells**, the same units the model uses. `x` runs downstream from
the inlet at 0 to the outlet at 672; `y` runs **up** from the road at 0.

| on screen | what it is |
|---|---|
| white / grey lines | the body shell |
| orange | floor and diffuser |
| blue | front and rear wings |
| green | the radiator duct |
| red circles | the wheels |
| dotted line at y=112 | **the tiling seam — the body must stay below it** |
| faint blue band at x≈384–400 | where the turbine cut lands; the rear tyre must not reach it |

Every plate is a **leading-edge point**, a **chord** in cells, and an **angle**
in degrees, with positive angle meaning the trailing edge is higher. You never
have to think in those terms though — drag the endpoints and the numbers follow.

---

## Editing

- **Drag a round handle** to move that end of a plate.
- **`weld`** (on by default) means that when two plates share a point, dragging
  one drags the other, so the outline stays joined instead of tearing.
- **`snap ½`** rounds to half a cell. Turn it off for fine work.
- **Click a plate** to select it; the right-hand panel shows its numbers and you
  can type them directly.
- **`split`** cuts the selected plate in half and gives you a new point to pull —
  this is how you add detail where the shape is too coarse.
- **`+ plate`** adds a fresh one; **`delete`** removes the selection.
- **Arrow keys** nudge by half a cell, **shift+arrow** by five.
- **`ctrl+Z`** undoes. **`Revert`** throws away everything unsaved.
- **Middle-drag or space+drag** pans, **scroll** zooms, **`fit`** reframes.

### Tracing over a picture

Drop any `.svg`, `.png` or `.jpg` into **`out/backdrops/`** and it appears in the
dropdown. Your `f1_side_profile.svg` is already there. Use the opacity, scale,
x and y sliders to line it up under the geometry, then drag the points onto it.

### Things you cannot type

A few plates have their height or angle owned by a knob in `CarParams`, and the
panel says so in amber when you select them:

- `FW_MAIN`, `FW_FLAP`, `DIFF` — height is *(a number) + ride height*
- `DUCT_UP`, `DUCT_LO` — height is `DUCT_Y0` plus an offset, so the duct stays
  exactly `DEVICE_CELLS` (32 cells) tall wherever the band is moved
- `DIFF`, `RW_MAIN`, `RW_FLAP` — angle comes from the diffuser / rear-wing knobs

You can still move them in `x` and reshape everything around them. Typing over a
bound number would simply be discarded at build time, which is why it is shown
locked rather than silently ignored.

---

## Check before you save

**`Check`** runs the real model — the same `car_bodies` and
`windows_from_geometry` the march itself calls — and reports:

| it checks | because |
|---|---|
| the body clears **y=112** | two 128-tall window rows in a 240-tall box force the seam exactly there. It cannot move. A body crossing it has its force split between two experts that only exchange a ring. |
| **no plate inside a wheel** | a plate inside a wheel double-counts that wheel's drag |
| the **duct sits inside the body** | the 32 cells between the duct walls are where the radiator and turbine planes live |
| the **front tyre clears the front wing** | same double-counting problem |
| the **rear tyre clears x=400** | `wx=128` and `device_overlap_max=16` force the two device cuts exactly 112 cells apart, which puts the turbine cut at 384–400 at the earliest |
| **a window layout exists** | if no covering layout can be found the car cannot be marched at all |

Gaps in the outline are reported as warnings, not failures — the wings are meant
to be detached.

**`Save`** writes `car_geometry.json` and keeps the previous version beside it as
`car_geometry.json.bak`, then re-checks automatically.

---

## From the command line

```bash
python scripts/car_check.py
```

Runs the identical checks (it calls the same function, so the two cannot drift
apart), prints them, and writes a picture to `out/car_geometry.png`. Exit code is
1 if anything failed, so you can put it in front of a long run.

---

## After you change the car

The geometry is an input to everything downstream, so:

```bash
python -m pytest tests/test_tier51_racelab_graph.py tests/test_tier55_car_geometry.py -q
```

is the quick sanity pass. **Any marched result — the arms, the gate, the settled
field cache — describes the car that was in place when it ran.** Changing the
shape does not invalidate the code, but it does invalidate those numbers, and
they have to be re-measured before they mean anything again:

```bash
python scripts/tier54_traced_car.py --stages spinup,size,verify,arms,compare
```

which is about a hundred minutes. The demo will also want a fresh settled field
(`--stages spinup` alone, a couple of minutes) or it releases this car from the
previous one's flow.
