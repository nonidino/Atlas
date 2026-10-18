"""The Expert Card of the Stage 0 expert ([[chart-operator-architecture]] section 4; the
public document's Appendix A), with its certificates filled from the one evaluation.

The host reads the card before admitting an expert.  **Host rules**: a chart outside the
envelope, or a Fourier number outside the declared range, is refused and the violated
quantity is named (`certificate.refusals`); G0 (tier 0) is admitted for constraint-mode
output on linear symmetric physics with no further certificate; a tier-2 convergence
claim needs a certified contraction factor below one, which this card does not carry;
a missing certificate is reported, never assumed.
"""

from __future__ import annotations

import hashlib
import json

from . import data, gates as G

HOST_RULES = (
    "a chart outside the envelope, or a Fourier number outside the declared range, is refused, "
    "naming the quantity",
    "tier 0 (G0) is admitted for constraint-mode output on linear symmetric physics with no "
    "further certificate",
    "a tier-2 convergence claim needs a certified contraction factor below one: none is carried",
    "a missing certificate is reported, never assumed",
)


def card(evaluation: dict, weights_path: str, plan: dict, config: dict) -> dict:
    env = data.ENVELOPE
    with open(weights_path, "rb") as fh:
        wsha = hashlib.sha256(fh.read()).hexdigest()
    armB = next(a for a in plan["arms"] if a["name"] == "B")
    e1 = evaluation.get("E1", {})
    se1 = [r["arms"]["B"]["se1"]["identity_rel"] for r in evaluation.get("benchmarks", [])
           if "se1" in r.get("arms", {}).get("B", {})]
    dual_rows = [d for d in evaluation.get("DUAL", {}).get("benchmark_records", [])]
    gaps = [x["computable_gap"] / x["network_error"] for d in dual_rows for x in d["data"]
            if x["network_error"] > 0]
    return {
        "expert": "chart-diffusion-2d",
        "version": "0.1.0 (Stage 0, W355)",
        "licence": "the owner's to choose; trained from scratch on this repository's own classical "
                   "data, with no Poseidon (CC-BY-NC) or NeuberNet (unlicensed) weights anywhere in "
                   "the chain (S12)",
        "family": "CO-D", "trace_space": "H1", "class": "G0",
        "label_map": {"u": "temperature", "port": "THERM"}, "ports": ["THERM"],
        "chart": {"reference": "unit-square",
                  "methods": ["conformal (harmonic with the layout's boundary correspondence, or "
                              "closed form)", "winslow-arc-length", "transfinite", "affine"],
                  "certificate": "bilinear-positive-jacobian + simple-boundary (Proposition 3)",
                  "inputs": ["logJ_centred", "re_mu", "im_mu", "log_kappa_hat", "source / rms"],
                  "envelope": {"sup_abs_mu": env.k_mu, "logJ_range": round(env.l_J, 4),
                               "smoothness_grad_logJ_plus_grad_mu": env.s_grad,
                               "abs_log_kappa_hat": env.l_kappa}},
        "discretisation": "bilinear finite elements on the reference grid's vertices (decision D7)",
        "time": {"scheme": "implicit-euler", "fourier_range": [env.fo_min, env.fo_max],
                 "steady": env.steady},
        "resolution": {"reference": 64, "trained": [32, 64, 128]},
        "output": {"format": "constraint-modes", "port_basis": config.get("kind", "sine-hat"),
                   "port_modes_per_side": config.get("m", 16),
                   "internal_modes_per_side": "16 (conforming layouts need no reparametrisation "
                                              "inside the expert; the estimate of 20 was not used)",
                   "principal_part": "plain-square, DST-I, float32 inside the network",
                   "particular_field": True, "form": config.get("form", "bubble")},
        "tiers": [0, 2],
        "backbone": {"name": config.get("backbone"), "size": config.get("size"),
                     "parameters": config.get("parameters")},
        "weights": {"sha256": wsha, "arm": "B", "steps": armB["steps"],
                    "data": plan["data"], "chosen_by": "median e_H on the validation split"},
        "certificates": {
            "gram_identity_residual_SE1": {"value": max(se1) if se1 else None,
                                           "method": "max over the benchmark pieces, relative"},
            "energy_error_by_mu_bin_E1": {"cells": e1.get("cells"), "met": e1.get("met")},
            "dual_gap_over_network_error_smooth_data": {
                "median": sorted(gaps)[len(gaps) // 2] if gaps else None,
                "pieces": len(dual_rows)},
            "contraction_factor_tier2": None,
            "date": evaluation.get("written"),
        },
        "precision": {"network": "bfloat16 on a GPU, float32 on a CPU", "host": "float64"},
        "host_rules": list(HOST_RULES),
        "registered_gates_sha256": evaluation.get("gates_sha256"),
    }


def to_json(c: dict) -> str:
    return json.dumps(c, indent=1)
