"""Atlas -- the generalized plug-in composition layer.

A case study is a graph of declared agent capability records plus port
connections.  The coupling scheme is **compiled** from those declarations, not
written per case; there is no hand-written coupling code anywhere in this
package, and adding a case study adds none.

    from atlas import compile_scheme
    from atlas.cases import wind_farm

    result = compile_scheme(wind_farm.build())
    print(result.report())

Nine layers on a seven-hypothesis envelope, with a three-verdict compiler:

    L1  capability record        capability.py
    L1.5 routing                 routing.py
    L2  decomposition, cuts      compiler.py
    L3  port algebra             ports.py, transfer.py, admissibility.py
    L4  transmission operator    probe.py
    L5  interface solve          scheme.py, compiler.py, solve.py
    L6  assembly                 assembly.py
    L7  time integration         compiler.py, solve.py
    L8  emit                     emit.py
    L9  typing of claims         claims.py

plus closure and substitution (composition.py), the conformance suite
(conformance.py), and the five named holes with their required measurements
(holes.py).

Nothing here fills a hole.  Where the specification left a named slot --
SeamReference, InterfaceMotion, TopologyEvent, AssemblyCertificate,
PortAmendment -- this package refuses or decertifies and emits the number the
missing rule will constrain.  Where it left a constant unmeasured -- L, beta,
sigma -- ``holes.Unmeasured`` refuses to behave like a number, so no bound can be
quoted from a constant nobody measured.
"""

from __future__ import annotations

from .admissibility import achievable_rung, check_connection
from .assembly import AssemblyCertificate, PartitionOfUnity, certify
from .capability import (
    BCChannel,
    ClaimType,
    Differentiable,
    Direction,
    ExpertCapabilities,
    EllipticSubsolve,
    TimeDiscretization,
    MotionClass,
    PortDecl,
    Transmission,
    linear_response,
    port_decl,
    zero_response,
)
from .claims import (
    ClaimTypeTag,
    HorizonBranch,
    QuantityClaim,
    TypedClaim,
    abstention_horizon,
    refuse_statistical_claim,
    type_claim,
)
from .compiler import CompileResult, compile_scheme
from .composition import (
    CompositeExpert,
    CompositionRefused,
    SubstitutionCertificate,
    certify_substitution,
    compose,
    schur_complement,
)
from .conformance import ConformanceCertificate, run_conformance
from .emit import EMIT_GROUPS, BoundTerms, RunArtifact
from .envelope import ENVELOPE, EnvelopeStamp, Hypothesis, Status
from .graph import (
    Agent,
    CaseGraph,
    Connection,
    Decomposition,
    FluxMatching,
    GlobalField,
    MeasuredConstants,
)
from .holes import (
    NAMED_HOLES,
    UNMEASURED_CONSTANTS,
    NamedHole,
    NamedHoleError,
    Unmeasured,
    UnmeasuredError,
)
from .ports import (PORT_SPECS, MappingClass, PairingCheck, PortType, ResponseHalf,
                    Role, check_response_half, check_scales)
from .probe import ProbeBudget, ProbedBlock, SeamOperator, assemble_seam, probe_block
from .multiphysics import (
    SeamDefect,
    SigmaLagCheck,
    TightCoupling,
    check_sigma_lag,
    interface_power,
    lag_distance,
    seam_defect_split,
    seam_jacobian,
    tight_couple,
)
from .routing import RoutingRefused, route
from .scheme import RULES, Accelerator, Budget, Levels, Ordering, Scheme
from .solve import (
    Declination,
    InterfaceProblem,
    RunRefused,
    StepResult,
    coupled_step,
    rollout,
    solve_interface,
)
from .transfer import (
    InterfaceSpace,
    Prolongation,
    SeamTransfer,
    dim_M,
    identity_prolongation,
    lumped_prolongation,
    mapping_class,
)
from .verdict import ADMIT, ADMIT_UNCERTIFIED, REFUSE, Decision, DecisionRecord, Verdict

__all__ = [
    "compile_scheme", "CompileResult",
    "CaseGraph", "Agent", "Connection", "Decomposition", "FluxMatching", "GlobalField",
    "ExpertCapabilities", "PortDecl", "port_decl",
    "BCChannel", "Differentiable", "Direction", "MotionClass", "ClaimType", "Transmission",
    "EllipticSubsolve", "MeasuredConstants", "TimeDiscretization",
    "linear_response", "zero_response",
    "PortType", "Role", "MappingClass", "PORT_SPECS", "check_scales",
    "ResponseHalf", "PairingCheck", "check_response_half",
    "InterfaceSpace", "Prolongation", "SeamTransfer", "mapping_class", "dim_M",
    "identity_prolongation", "lumped_prolongation",
    "check_connection", "achievable_rung",
    "probe_block", "assemble_seam", "ProbedBlock", "SeamOperator", "ProbeBudget",
    "EnvelopeStamp", "Hypothesis", "Status", "ENVELOPE",
    "Verdict", "ADMIT", "ADMIT_UNCERTIFIED", "REFUSE", "Decision", "DecisionRecord",
    "Scheme", "Budget", "Ordering", "Accelerator", "Levels", "RULES",
    "PartitionOfUnity", "AssemblyCertificate", "certify",
    "TypedClaim", "ClaimTypeTag", "HorizonBranch", "QuantityClaim", "type_claim",
    "abstention_horizon", "refuse_statistical_claim",
    "RunArtifact", "BoundTerms", "EMIT_GROUPS",
    "compose", "CompositeExpert", "CompositionRefused", "schur_complement",
    "certify_substitution", "SubstitutionCertificate",
    "tight_couple",
    "seam_defect_split",
    "interface_power",
    "TightCoupling",
    "SeamDefect",
    "seam_jacobian",
    "lag_distance", "check_sigma_lag", "SigmaLagCheck",
    "run_conformance", "ConformanceCertificate",
    "route", "RoutingRefused",
    "coupled_step", "rollout", "solve_interface", "StepResult", "InterfaceProblem",
    "RunRefused", "Declination",
    "NamedHole", "NamedHoleError", "NAMED_HOLES",
    "Unmeasured", "UnmeasuredError", "UNMEASURED_CONSTANTS",
]

__version__ = "0.1.0"
