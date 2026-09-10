"""CS-S2 -- NeuberNet wearing the expert interface, and nothing more.

**Provenance, stated because the licence is not what the donor survey hoped.**

  * weights and model code from https://github.com/grossIt/neubernet at commit
    ``cce93244dfc27db9dbe351760760a8142cee0bb0`` (2025-07-09): the four files in
    `FILES`, with their sizes and SHA-256 checked on every load;
  * the repository carries **no licence** -- GitHub reports none -- so no right
    beyond reading it is granted; the paper's code-availability statement points
    readers to it; the Zenodo dataset (doi:10.5281/zenodo.14880154, 29.8 GB) is
    CC BY 4.0 and its preview does not show these files; the article itself is
    CC BY-NC-ND 4.0;
  * the files live OUTSIDE this repository, in ``~/.cache/neubernet`` (override
    with ``NEUBERNET_DIR``), and nothing derived from them except measured
    numbers is committed.

**Loading.**  `neubernet.pt` is a whole pickled module (the training script calls
``torch.save(model, ...)``).  It is loaded with ``torch.load(weights_only=True)``
under an allowlist that maps the checkpoint's ``utils.definitions.*`` names onto
the classes of the reviewed `definitions.py`, imported under a private module
name, so no pickle opcode can construct anything outside that list.  An opcode
walk of all three checkpoints finds nothing else to allow.

**What the expert is**, read from `database/s3_import_database.py` and
`fem/s2_ansys_database_macro.mac`:

  * input row = [108 boundary values, 6 parameters, x, y] (116 columns);
  * boundary values = [u_x (36), u_y (36), ROTY (36)] at -180..170 deg on
    |p| = 5 R_n, each times E / sigma_y, displacements over R_n; ROTY is the
    rotation about the shaft axis, u_theta / rho in magnitude -- and its SIGN
    against the classical patch's hoop axis is measured rather than read off
    the pipeline: `NeuberNetPatch.HOOP_SIGN`, the third port defect;
  * parameters = [R / R_n, alpha (deg), beta (deg), sigma_y / E, E_t / E, nu];
  * output = 14 full fields over sigma_y (no elastic solution subtracted):
    [omega_el, omega_pl, s_xx, s_yy, s_zz, s_xy, s_yz, s_xz, eps_pl x 6], in
    the notch frame (the macro writes stresses under `RSYS,12`);
  * `production_mode` is True in the shipped file: a SignNet sets the signs of
    tension and torsion and a YieldNet rescales an elastic input up to yield and
    the output back down.  That is the operator the authors publish, so it is
    the one measured.  Its forward MUTATES its input tensor, so every call here
    passes a copy.

**The port.**  The seam is the material part of the ring.  The trace is the
displacement at the material sensors, u E / (sigma_y R_n) with the hoop
component as u_theta, read as the hat interpolant between sensors; the flux is
the traction sigma . n / sigma_y weighted by the same hats, int t phi_j rho ds /
int phi_j rho ds -- the Galerkin dual of that trace, and the restriction that
converges (pointwise sampling at the knots does not; see `cs_s2_elastic_patch`).  Sensors inside the notch opening are not boundary data and are filled
exactly as the pipeline fills them, by linear interpolation across the gap, so
a poke at a flank-end sensor reaches NeuberNet through the filled entries too.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import cs_s2_elastic_patch as EP  # noqa: E402

CACHE = os.environ.get("NEUBERNET_DIR",
                       os.path.join(os.path.expanduser("~"), ".cache", "neubernet"))

SOURCE = {
    "repository": "https://github.com/grossIt/neubernet",
    "commit": "cce93244dfc27db9dbe351760760a8142cee0bb0",
    "commit_date": "2025-07-09",
    "licence": None,
    "licence_note": ("no licence on the repository (GitHub API: license null); the "
                     "Zenodo dataset doi:10.5281/zenodo.14880154 is CC BY 4.0 and its "
                     "preview does not show the weight files; the article is "
                     "CC BY-NC-ND 4.0"),
    "paper": ("Grossi, Beghini & Benedetti, NeuberNet, Communications Engineering "
              "(2025), doi:10.1038/s44172-025-00549-5"),
    "stored_outside_repository": "~/.cache/neubernet (NEUBERNET_DIR)",
}

FILES = {
    "neubernet.pt": (89998330,
                     "a89de3f48c36be8c576f2a2cc5a32d5ef6313d2925426e7c33cdd142b804a0a6"),
    "yieldnet.pt": (2165818,
                    "f495f28644ebbef67cc25a287978ff69dd8c19f498e7242e1721163db4871adc"),
    "signnet.pt": (2165004,
                   "4a44bb5386fdbe607f324495f71617bbc2f628b3e611819b4067f06c05040506"),
    "definitions.py": (17874,
                       "873a60a30866db42bc71e7be813429263eb548bbb8675432f79f9991c1fcdf93"),
}

#: **The checkpoint carries 2,198,346 subnormal weights -- 9.8% of its
#: 22,450,322 parameters -- counted 2026-09-10 (Tier 41).**  On an x86 CPU every
#: product that touches one takes a microcode assist: one forward of the port's
#: 120 quadrature points was measured at 61.5 s with torch's flush-denormal off and
#: 0.28 s with it on, on the same state, with the flux output BITWISE equal -- a
#: subnormal term cannot move a float32 sum of ordinary size.  So the flush is a
#: speed setting and not a change to the model, and `load_model` sets it.  The
#: second CS-S2 run and the torsion control ran without it; the third run and the
#: sign-jump measurement ran with it, and the third run is compared against the
#: second where the hoop fix cannot reach.
FLUSH_DENORMAL = True

#: output columns [s_xx, s_yy, s_zz, s_xy, s_yz, s_xz] / sigma_y
STRESS = slice(2, 8)
COMP = {"ux": 0, "uy": 1, "ut": 2}


def available():
    return all(os.path.exists(os.path.join(CACHE, f)) for f in FILES)


def verify():
    """Size and SHA-256 of every file, against the values recorded at download."""
    out = {}
    for name, (size, sha) in FILES.items():
        path = os.path.join(CACHE, name)
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        got = os.path.getsize(path)
        out[name] = {"size": got, "sha256": digest,
                     "matches": bool(got == size and digest == sha)}
        if not out[name]["matches"]:
            raise RuntimeError(f"{name}: size/sha256 differ from the recorded download")
    return out


def load_model():
    spec = importlib.util.spec_from_file_location(
        "_neubernet_definitions", os.path.join(CACHE, "definitions.py"))
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    allow = [(D.BranchNet, "utils.definitions.BranchNet"),
             (D.NOMAD, "utils.definitions.NOMAD"),
             (D.NeuberNetComponent, "utils.definitions.NeuberNetComponent"),
             (D.NeuberNet, "utils.definitions.NeuberNet"),
             (D.YieldNet, "utils.definitions.YieldNet"),
             (D.SignNet, "utils.definitions.SignNet"),
             torch.nn.Linear, torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.ModuleList]
    with torch.serialization.safe_globals(allow):
        model = torch.load(os.path.join(CACHE, "neubernet.pt"), weights_only=True,
                           map_location="cpu")
    model.eval()
    if FLUSH_DENORMAL:
        #: per calling thread -- every CS-S2 script runs torch on one thread
        if not torch.set_flush_denormal(True):
            raise RuntimeError("this CPU does not support flushing denormals")
    return model


class NeuberNetPatch:
    """The adapter.  ``respond(port, trace)`` on ports ``ring:ux|uy|ut|all``."""

    def __init__(self, model, alpha_deg=30.0, R=50.0, sy_over_e=3e-3,
                 et_over_e=1e-2, nu=0.3, base36=None, agent_id="NEUBERNET"):
        self.model = model
        self.alpha = float(alpha_deg)
        self.R = float(R)
        self.nu = float(nu)
        self.sy_over_e = float(sy_over_e)
        self.theta_s, self.phi_s = EP.material_sensors(self.alpha)
        self.M = self.n = int(self.theta_s.size)
        self.mat = np.array([int(np.flatnonzero(np.isclose(EP.SENSOR_ANGLES, t))[0])
                             for t in self.theta_s])
        self.void = np.setdiff1d(np.arange(EP.SENSOR_ANGLES.size), self.mat)
        self.allphi = np.mod(EP.SENSOR_ANGLES, 360.0)
        th = np.radians(self.theta_s)
        self.points = np.stack([EP.RL * np.cos(th), EP.RL * np.sin(th)], axis=1)
        self.normals = np.stack([np.cos(th), np.sin(th)], axis=1)
        self.rho36 = self.R + EP.RL * np.cos(np.radians(EP.SENSOR_ANGLES))
        self._build_ring_quadrature()
        self.params = np.array([self.R, self.alpha, 0.0, sy_over_e, et_over_e, nu])
        self.base36 = (np.zeros((EP.SENSOR_ANGLES.size, 3)) if base36 is None
                       else np.array(base36, dtype=float))
        self.agent_id = agent_id
        self.calls = 0

    # -- the pipeline's input ----------------------------------------------

    #: ``ROTY = HOOP_SIGN * u_theta / rho``, with u_theta along the classical
    #: patch's hoop axis -- the axis its s_yz and s_xz are taken on.  **The third
    #: port defect, measured 2026-09-10 (Tier 41).**  The first adapter wrote +1,
    #: and `cs_s2_torsion_control.py` found it the wrong way round: at a base with
    #: nonzero torsion on the elastic branch -- where SignNet's call is firm and
    #: production mode is exactly odd in torsion -- NeuberNet's s_yz and s_xz came
    #: back NEGATED against linear elasticity (cosine -1.000 and -0.999, hoop flux
    #: -0.996, s_yz off by 198%) while its in-plane stresses agreed; with -1 the
    #: same row reads +1.000, +0.999, +0.996 and 2.3%.  The convention check the
    #: first adapter passed was run at a tension-only base, where both subjects'
    #: hoop shears are identically zero and no sign could show.  The run this
    #: corrects is kept locally as `out/cs_s2/cs_s2_run2_hoop_plus1.json`.
    HOOP_SIGN = -1.0

    def fill_opening(self, s36):
        v = np.array(s36, dtype=float, copy=True)
        for k in range(3):
            v[self.void, k] = np.interp(self.allphi[self.void], self.phi_s,
                                        v[self.mat, k], period=360.0)
        return v

    def branch(self, s36):
        v = self.fill_opening(s36)
        return np.concatenate([v[:, 0], v[:, 1], self.HOOP_SIGN * v[:, 2] / self.rho36])

    # -- forward ------------------------------------------------------------

    def fields(self, s36, points):
        X = np.empty((points.shape[0], 116))
        X[:, :108] = self.branch(s36)
        X[:, 108:114] = self.params
        X[:, 114:116] = points
        xt = torch.as_tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y = self.model(xt.clone())
        self.calls += 1
        return y.numpy().astype(np.float64)

    def stress(self, s36, points):
        return self.fields(s36, points)[:, STRESS]

    @staticmethod
    def traction(sig, normals):
        nx, ny = normals[:, 0], normals[:, 1]
        return np.stack([sig[:, 0] * nx + sig[:, 3] * ny,
                         sig[:, 3] * nx + sig[:, 1] * ny,
                         sig[:, 5] * nx + sig[:, 4] * ny], axis=1)

    def _build_ring_quadrature(self, nq=4):
        """Gauss points on the material arc, segment by segment between the hats' knots."""
        ts = EP.void_half_angle(self.alpha)
        knots = np.concatenate([[ts], self.phi_s, [360.0 - ts]])
        g, w = np.polynomial.legendre.leggauss(nq)
        phi, wt = [], []
        for a, b in zip(knots[:-1], knots[1:]):
            phi.append(0.5 * (b - a) * g + 0.5 * (b + a))
            wt.append(0.5 * (b - a) * w)
        phi = np.concatenate(phi)
        th = np.radians(phi)
        self.q_points = np.stack([EP.RL * np.cos(th), EP.RL * np.sin(th)], axis=1)
        self.q_normals = np.stack([np.cos(th), np.sin(th)], axis=1)
        self.q_hats = EP.ring_basis(phi, self.alpha)
        self.q_weight = (np.radians(np.concatenate(wt))
                         * (self.R + EP.RL * np.cos(th)) * EP.RL)
        #: int phi_j rho ds, by quadrature
        self.hat_mass = (self.q_hats * self.q_weight[:, None]).sum(axis=0)

    def ring_flux(self, s36):
        """Hat-weighted traction at the sensors, (M, 3)."""
        t = self.traction(self.stress(s36, self.q_points), self.q_normals)
        return (self.q_hats.T @ (t * self.q_weight[:, None])) / self.hat_mass[:, None]

    def _perturbed(self, port_name, trace):
        comp = port_name.split(":", 1)[1]
        s = self.base36.copy()
        tr = np.asarray(trace, dtype=float).ravel()
        if comp == "all":
            for k in range(3):
                s[self.mat, k] += tr[k * self.M:(k + 1) * self.M]
        else:
            s[self.mat, COMP[comp]] += tr
        return comp, s

    def respond(self, port_name, trace):
        comp, s = self._perturbed(port_name, trace)
        f = self.ring_flux(s)
        return f.T.ravel() if comp == "all" else f[:, COMP[comp]]

    def respond_pointwise(self, port_name, trace):
        comp, s = self._perturbed(port_name, trace)
        t = self.traction(self.stress(s, self.points), self.normals)
        return t.T.ravel() if comp == "all" else t[:, COMP[comp]]

    # -- which branch of the production forward a state is on -------------

    def regime(self, s36):
        """A read-only replica of the production pre-processing, for reporting."""
        m = self.model
        bid = int(m.branch_input_dim)
        two = 2 * (bid // 3)
        x = torch.as_tensor(np.concatenate([self.branch(s36), self.params])[None],
                            dtype=torch.float32)
        norm = torch.sqrt(
            torch.linalg.vector_norm(x[:, :two], dim=-1, keepdim=True) ** 2
            + torch.linalg.vector_norm(x[:, two:bid] * x[:, [bid]], dim=-1,
                                       keepdim=True) ** 2)
        aux = x.clone()
        aux[:, :bid] /= norm
        with torch.no_grad():
            sign = torch.sign(m.signnet(aux) - 0.5)
            aux[:, :two] *= sign[:, [0]]
            aux[:, two:bid] *= sign[:, [1]]
            yx = m.yieldnet(aux)
        y0 = float(yx[0, 0] * norm[0, 0])
        y1 = float(yx[0, 1])
        return {"elastic_von_mises_over_sy": y0,
                "small_scale_indicator": y1,
                "branch": "elastic (rescaled)" if y0 <= 1.0 + m.yield_tol else "plastic",
                "small_scale_plasticity_violated": bool(y1 + m.small_scale_pl_tol < y0),
                "sign_tension": float(sign[0, 0]), "sign_torsion": float(sign[0, 1]),
                "branch_norm": float(norm[0, 0])}


def homogeneous_sensors(alpha_deg, R, nu, axial=0.0, shear=0.0):
    """The uniform far field sampled straight onto the 36 sensors, normalised.

    ``axial`` and ``shear`` are in units of sigma_y, so the result is already
    u E / (sigma_y R_n).  A smoke-test base only: it carries no notch.
    """
    th = np.radians(EP.SENSOR_ANGLES)
    pts = np.stack([EP.RL * np.cos(th), EP.RL * np.sin(th)], axis=1)
    u, ut = EP.homogeneous_field(pts, R, nu, axial=axial, shear=shear)
    return np.stack([u[:, 0], u[:, 1], ut], axis=1)


def smoke():
    torch.set_num_threads(1)
    ver = verify()
    model = load_model()
    alpha, R, nu = 30.0, 50.0, 0.3
    s36 = homogeneous_sensors(alpha, R, nu, axial=1.0)
    nn = NeuberNetPatch(model, alpha, R, 3e-3, 1e-2, nu, base36=s36)
    print("files verified:", all(v["matches"] for v in ver.values()))
    print("material sensors:", nn.M, "void:", nn.void.size,
          "half-angle %.2f deg" % EP.void_half_angle(alpha))
    reg = nn.regime(s36)
    print("unit axial far field:", reg)
    scale = 0.7 / reg["elastic_von_mises_over_sy"]
    s_el = scale * s36
    print("elastic base scale %.4f ->" % scale, nn.regime(s_el))
    xs, ys = np.meshgrid(np.linspace(-5, 5, 101), np.linspace(-5, 5, 101))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    keep = (np.hypot(pts[:, 0], pts[:, 1]) <= 5.0) & ~EP.in_void(pts[:, 0], pts[:, 1], alpha)
    pts = pts[keep]
    sig = nn.stress(s_el, pts)
    k = int(np.argmax(sig[:, 1]))
    print("max s_yy %.4f at (%.2f, %.2f); s_yy at the tip (0,0): %.4f"
          % (sig[k, 1], pts[k, 0], pts[k, 1],
             nn.stress(s_el, np.array([[0.0, 0.0]]))[0, 1]))
    t = nn.respond("ring:uy", np.zeros(nn.M))
    print("t_y on the ring (first 5):", np.round(t[:5], 5), "calls", nn.calls)


if __name__ == "__main__":
    smoke()
