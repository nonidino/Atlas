"""Interface transfer: one declared object, and the reduction is forced.

interface-transfer-theory §1-§6.  This is the module the rest of the package
leans on hardest, because it is where six of the nine importable mechanisms
collapse into a single declaration.

Fix a seam joining agents A and B through one port type.  Three spaces, not two:

    V_A   the trace space A can accept and return on its own discretization
    V_B   the same for B
    M     a common interface space, chosen once per seam

Each side declares exactly one operator, the prolongation

    P_i : M -> V_i

which presents an interface datum in the form agent i accepts.  Nothing else
about the transfer is declared.  Requiring that the transfer neither create nor
destroy interface power gives, for all mu in M and all e_i in V_i,

    < R_i e_i , mu >_M  =  < e_i , P_i mu >_{V_i}

which is the definition of an adjoint, so R_i = P_i^*.  In discrete form with
Gram (mass) matrices G_M and G_V that is

    R_i = G_M^{-1} P_i^T G_V

and any other choice leaks power at the interface.

Three things follow and this module derives all three rather than declaring
them:

  * the mapping class -- efforts consistent through P_i, flows conservative
    through P_i^* -- so getting it backwards is not expressible;
  * field-to-lumped adjointness is the case dim M = 1;
  * non-conforming geometry is admissible, geometric coincidence being the
    special case M = V_A = V_B, P_i = identity.

The accuracy of P_i itself is NOT bought by any of this.  Adjointness buys no
power leak, never accuracy; a wrong P_i is a real error, it is consistent with
its own adjoint, and it enters sigma through sigma_nc (§4.2), which is measured
here and reduced by nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .ports import MappingClass, PortType, Role, spec_for


class TransferDeclarationError(ValueError):
    """A prolongation or interface space that cannot be used as declared."""


@dataclass(frozen=True)
class InterfaceSpace:
    """The common interface space M for one seam.

    ``gram`` is the M-inner-product that makes effort times flow a power.  The
    default is the identity, which is correct for an orthonormal modal basis and
    is the case a probe on a Fourier basis produces.
    """

    seam_id: str
    dim: int
    basis: str = "orthonormal-modal"
    gram: np.ndarray | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise TransferDeclarationError(f"interface space {self.seam_id!r} has dim {self.dim}")
        if self.gram is not None:
            g = np.asarray(self.gram, dtype=float)
            if g.shape != (self.dim, self.dim):
                raise TransferDeclarationError(
                    f"gram for {self.seam_id!r} has shape {g.shape}, expected {(self.dim, self.dim)}"
                )

    @property
    def G(self) -> np.ndarray:
        return np.eye(self.dim) if self.gram is None else np.asarray(self.gram, dtype=float)

    @property
    def is_lumped(self) -> bool:
        """dim M = 1: the field-to-lumped case, where adjointness is G3/I5."""
        return self.dim == 1


def dim_M(effective_resolutions: dict[str, int]) -> int:
    """dim M = min_i m_i_eff  (interface-transfer-theory §4.3).

    The multiplier space and the probe basis are the same space.  Representing
    the trace more finely than the coarser expert can respond to buys nothing,
    because that side's operator returns noise on those modes -- which is also
    the operational form of the inf-sup condition: it is measured as beta on the
    assembled matrix rather than proved, the only version available to a black
    box with no approximation theory.
    """
    if not effective_resolutions:
        raise TransferDeclarationError("no effective interface resolutions declared")
    bad = {k: v for k, v in effective_resolutions.items() if v is None or int(v) < 1}
    if bad:
        raise TransferDeclarationError(f"effective interface resolution unset or invalid: {bad}")
    return int(min(int(v) for v in effective_resolutions.values()))


@dataclass
class Prolongation:
    """P_i : M -> V_i, the one operator a side declares.

    ``matrix`` is (n_i, dim_M).  ``gram_V`` is the V_i inner product; identity by
    default.  ``nondim`` is the diagonal unit conversion D_i, so that
    P_i = D_i^{-1} P_hat_i with P_hat_i the purely geometric transfer
    (interface-transfer-theory §6).

    The reduction is not a field on this class.  It is ``adjoint()``.
    """

    agent_id: str
    port_name: str
    matrix: np.ndarray
    gram_V: np.ndarray | None = None
    nondim_diag: np.ndarray | None = None
    label: str = ""
    #: An escape hatch that exists only so a hand-written reduction is *catchable*.
    #: Leave it None. If it is set, the adjointness residual stops being machine
    #: precision by construction and the conformance suite has something to fail on.
    declared_reduction: np.ndarray | None = None

    def __post_init__(self) -> None:
        P = np.asarray(self.matrix, dtype=float)
        if P.ndim != 2:
            raise TransferDeclarationError(
                f"prolongation for {self.agent_id}.{self.port_name} must be 2-D, got shape {P.shape}"
            )
        self.matrix = P
        if self.gram_V is not None:
            g = np.asarray(self.gram_V, dtype=float)
            if g.shape != (P.shape[0], P.shape[0]):
                raise TransferDeclarationError(
                    f"gram_V for {self.agent_id}.{self.port_name} has shape {g.shape}, "
                    f"expected {(P.shape[0], P.shape[0])}"
                )
            self.gram_V = g
        if self.nondim_diag is not None:
            d = np.asarray(self.nondim_diag, dtype=float).reshape(-1)
            if d.shape[0] != P.shape[0]:
                raise TransferDeclarationError(
                    f"nondim_diag for {self.agent_id}.{self.port_name} has length {d.shape[0]}, "
                    f"expected {P.shape[0]}"
                )
            if np.any(d == 0.0):
                raise TransferDeclarationError(
                    f"nondim_diag for {self.agent_id}.{self.port_name} has a zero entry; "
                    "the unit conversion would not be invertible"
                )
            self.nondim_diag = d

    # -- shape -------------------------------------------------------------

    @property
    def n_V(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def dim_M(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def G_V(self) -> np.ndarray:
        return np.eye(self.n_V) if self.gram_V is None else self.gram_V

    @property
    def is_identity(self) -> bool:
        """Geometric coincidence, the special case M = V_i and P_i = identity."""
        if self.n_V != self.dim_M:
            return False
        return bool(np.allclose(self.effective_matrix(), np.eye(self.dim_M), atol=1e-12))

    def effective_matrix(self) -> np.ndarray:
        """P_i including the unit conversion:  P_i = D_i^{-1} P_hat_i."""
        if self.nondim_diag is None:
            return self.matrix
        return self.matrix / self.nondim_diag[:, None]

    # -- the forced half ---------------------------------------------------

    def adjoint(self, space: InterfaceSpace) -> np.ndarray:
        """R_i = P_i^*, forced by power preservation. Never declared.

        R_i = G_M^{-1} P_i^T G_V, taking the adjoint with respect to the interface
        pairings that make effort times flow a power.
        """
        P = self.effective_matrix()
        if P.shape[1] != space.dim:
            raise TransferDeclarationError(
                f"{self.agent_id}.{self.port_name}: prolongation maps from dim "
                f"{P.shape[1]}, interface space {space.seam_id!r} has dim {space.dim}"
            )
        G_M = space.G
        A = P.T @ self.G_V
        if space.gram is None:
            return A
        return np.linalg.solve(G_M, A)

    def reduce(self, e_V: np.ndarray, space: InterfaceSpace) -> np.ndarray:
        """Map an effort or flux from V_i back to M through the forced adjoint."""
        return self.adjoint(space) @ np.asarray(e_V, dtype=float)

    def prolong(self, mu_M: np.ndarray, space: InterfaceSpace) -> np.ndarray:
        """Map an interface datum from M into V_i through the declared P_i."""
        if space.dim != self.dim_M:
            raise TransferDeclarationError(
                f"{self.agent_id}.{self.port_name}: dim mismatch with {space.seam_id!r}"
            )
        return self.effective_matrix() @ np.asarray(mu_M, dtype=float)

    # -- diagnostics the compiler turns into verdicts -----------------------

    def norm(self) -> float:
        """||P_i||, the stability half of the revised connection rule."""
        return float(np.linalg.norm(self.effective_matrix(), 2))

    def is_stable(self, ceiling: float = 1e8) -> bool:
        n = self.norm()
        return bool(np.isfinite(n) and n <= ceiling)

    def constant_reproduction_residual(self) -> float:
        """How far P_i is from reproducing constants -- the consistency property.

        A consistent map is an interpolation: it reproduces constants, i.e. its
        rows sum to one when M carries a constant mode.  Reported rather than
        enforced, because a modal M has no constant mode to reproduce and the
        number is then meaningless; the compiler treats it as evidence.
        """
        P = self.effective_matrix()
        ones_M = np.ones(self.dim_M)
        return float(np.linalg.norm(P @ ones_M - np.ones(self.n_V), np.inf))

    def adjointness_residual(self, space: InterfaceSpace, trials: int = 8, seed: int = 0) -> float:
        """The numerical identity test for < R e, mu >_M == < e, P mu >_V.

        Derived reductions pass this at machine precision by construction.  The
        test is still run, because a caller can inject a hand-written reduction
        through ``declared_reduction`` and because that injection is exactly the
        classic actuator-disk failure -- the disk was not missing a check, it was
        missing a declaration, and an R was written by hand.
        """
        rng = np.random.default_rng(seed)
        R = self._reduction_in_use(space)
        P = self.effective_matrix()
        G_M, G_V = space.G, self.G_V
        worst = 0.0
        for _ in range(trials):
            mu = rng.standard_normal(space.dim)
            e = rng.standard_normal(self.n_V)
            lhs = float((R @ e) @ (G_M @ mu))
            rhs = float(e @ (G_V @ (P @ mu)))
            scale = max(abs(lhs), abs(rhs), 1e-30)
            worst = max(worst, abs(lhs - rhs) / scale)
        return worst

    # -- the escape hatch, so that a wrong declaration is catchable ---------

    def _reduction_in_use(self, space: InterfaceSpace) -> np.ndarray:
        if self.declared_reduction is None:
            return self.adjoint(space)
        return np.asarray(self.declared_reduction, dtype=float)

    def uses_hand_written_reduction(self) -> bool:
        return self.declared_reduction is not None


def mapping_class(role: Role) -> MappingClass:
    """Derived, not chosen. interface-transfer-theory §2.2.

        effort --P_i-->      consistent   (interpolation; reproduces constants)
        flow   --P_i^*-->    conservative (the adjoint; preserves integrals)

    The classic partitioned-coupling bug needs two independent declarations that
    can disagree.  With one declaration there is nothing to disagree with, which
    is why this function takes no declaration and cannot be overridden.
    """
    return MappingClass.CONSISTENT if role is Role.EFFORT else MappingClass.CONSERVATIVE


def transfer_operator(role: Role, prolongation: Prolongation, space: InterfaceSpace) -> np.ndarray:
    """The operator that actually moves a quantity of this role across the seam."""
    if role is Role.EFFORT:
        return prolongation.effective_matrix()
    return prolongation.adjoint(space)


@dataclass
class SeamTransfer:
    """The complete declared transfer for one seam: M, and one P_i per side."""

    seam_id: str
    port_type: PortType
    space: InterfaceSpace
    prolongations: dict[str, Prolongation]   # agent_id -> P_i
    passengers: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if len(self.prolongations) != 2:
            raise TransferDeclarationError(
                f"seam {self.seam_id!r} declares {len(self.prolongations)} prolongations; "
                "a port connection has exactly two sides"
            )
        for agent_id, P in self.prolongations.items():
            if P.dim_M != self.space.dim:
                raise TransferDeclarationError(
                    f"seam {self.seam_id!r}: prolongation for {agent_id} maps from dim "
                    f"{P.dim_M}, interface space has dim {self.space.dim}"
                )

    @property
    def sides(self) -> tuple[str, str]:
        return tuple(self.prolongations)  # type: ignore[return-value]

    @property
    def conforming(self) -> bool:
        """Geometric coincidence: the special case every prolongation is identity."""
        return all(P.is_identity for P in self.prolongations.values())

    def sigma_nc_factor(self, lambda_star: np.ndarray | None = None) -> float | None:
        """The non-conforming consistency term's factor, ||(I - Pi_M) lambda_star||.

        A genuinely new error term for non-conforming coupling, and it belongs in
        sigma, not in tau -- the agents are innocent of it.  Returns None when no
        reference trace is supplied, which is the usual case and is reported as
        such rather than defaulted to zero.
        """
        if lambda_star is None:
            return None
        lam = np.asarray(lambda_star, dtype=float)
        P = next(iter(self.prolongations.values())).effective_matrix()
        if P.shape[0] != lam.shape[0]:
            return None
        # Pi_M is the M-projection expressed on V through the declared P.
        Pi = P @ np.linalg.pinv(P)
        return float(np.linalg.norm(lam - Pi @ lam))

    def diagnostics(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "seam": self.seam_id,
            "port_type": self.port_type.value,
            "dim_M": self.space.dim,
            "conforming": self.conforming,
            "lumped": self.space.is_lumped,
        }
        for agent_id, P in self.prolongations.items():
            out[f"norm_P[{agent_id}]"] = P.norm()
            out[f"adjointness_residual[{agent_id}]"] = P.adjointness_residual(self.space)
            out[f"hand_written_reduction[{agent_id}]"] = P.uses_hand_written_reduction()
        return out


def identity_prolongation(agent_id: str, port_name: str, dim: int) -> Prolongation:
    """The conforming case, written explicitly so it is a declaration like any other."""
    return Prolongation(
        agent_id=agent_id,
        port_name=port_name,
        matrix=np.eye(dim),
        label="identity (geometric coincidence)",
    )


def lumped_prolongation(
    agent_id: str,
    port_name: str,
    weights: np.ndarray,
    label: str = "",
) -> Prolongation:
    """The field-to-lumped case: dim M = 1, P spreads one scalar over the seam.

    The reduction is then forced to be the corresponding weighted integral, and
    no other reduction preserves the bond.  This is I5 / G3 in constructive form.
    """
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    return Prolongation(
        agent_id=agent_id,
        port_name=port_name,
        matrix=w,
        label=label or "lumped (dim M = 1)",
    )


def check_lumped_port_declares_transfer(port_type: PortType, space: InterfaceSpace) -> str | None:
    """A lumped port type on a space of dim > 1 is a declaration mismatch."""
    spec = spec_for(port_type)
    if spec.lumped and not space.is_lumped:
        return (
            f"{port_type.value} is a lumped port type but its seam declares dim M = "
            f"{space.dim}; a lumped exchange is one scalar, so dim M = 1"
        )
    return None
