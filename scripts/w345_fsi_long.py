"""W345: march CS-12 2000 macro-steps and record the one-step residual every 25.

Written after the floor stage of `w345_coarse_competitor.py` found the residual
falling where CS-12's case page had implied a settled unsteadiness.  Writes
``out/w345/fsi_long.json``; about 7 minutes on the dev laptop.  Run from the repo root.
"""
import os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.getcwd())
import numpy as np
from atlas.cases import wing_fsi as wf
N = 2000
keep = sorted({s for k in range(0, N, 25) for s in (k, k + 1)})
t0 = time.time()
r = wf.FSIRollout().run(steps=N, keep_fields=tuple(keep))
rows = []
for k in range(0, N, 25):
    if k + 1 in r.fields and k in r.fields:
        u0, v0, _ = r.fields[k]; u1, v1, _ = r.fields[k + 1]
        rows.append((k, float(np.sqrt(np.mean((u1-u0)**2+(v1-v0)**2)) / np.sqrt(np.mean(u1**2+v1**2)))))
load = r.load
out = {"steps": N, "residual_every_25": rows, "wall_s": time.time() - t0,
       "load_pp_by_quarter": [float((q.max()-q.min())/abs(q.mean())) for q in np.array_split(load, 4)],
       "tip_last": float(r.tip[-1])}
json.dump(out, open("out/w345/fsi_long.json", "w"), indent=1)
for k, x in rows[::8]: print(k, "%.3e" % x)
print("load p-p by quarter", out["load_pp_by_quarter"], "wall", out["wall_s"])
