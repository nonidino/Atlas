"""Array backend for the windowed solve: numpy on CPU, torch on CPU or CUDA.

**Why this exists, measured rather than assumed.** Profiling one wind-farm
macro-step (124 windows, 128x128 each -- `scripts/windfarm_profile_step.py`)
puts **95.2%** of 182 s inside `WindowNS`'s finite-difference kernels
(`_ddx`, `_ddy`, `_lap`, `_adv`) and 0.1% in the global pressure solve. The
whole macro-step is 122 756 Python calls, so essentially none of it is
interpreter overhead: it is batched elementwise work on a `[124, 128, 128]`
float64 array -- 16 MB a pass, roughly 1700 passes per macro-step.

That diagnosis was then checked against the obvious cheap fix before any
hardware was rented. A thread pool over the batch dimension (numpy releases the
GIL inside large ufunc loops, so the slices need no copy) gives **2.8x at 8
threads and no more at 22** on a 22-core box. Saturating that early means the
wall is **memory bandwidth, not cores** -- which is exactly the case where a
GPU's advantage is real and quantifiable, because a GPU's edge on a 5-point
stencil is its ~1 TB/s against a desktop's ~40 GB/s, not its FLOPs. It is also
the case where renting a bigger CPU box would have bought almost nothing, which
is worth having on the record.

**The design constraint is that there is no GPU on the development machine.**
So the port cannot be "write CUDA and hope". The torch path has to be
verifiable *without* a GPU -- and it is, because torch runs the identical code
on CPU. `tests/atlas/windfarm/test_backend.py` pins torch-CPU against numpy on
a real `step_batch`, so `device='cuda'` is the only variable that remains
untested off the box: one flag, exercised the moment a run starts there, rather
than a whole second implementation that can drift.

**Contract.** `WindowNS` is written once, against this interface. The numpy
backend is a transparent pass-through by design: the default path -- the one
whose correctness was established the hard way -- must execute exactly the
calls it always did, with no behaviour hidden behind the abstraction.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.fft import dct as _scipy_dct, idct as _scipy_idct


def dct_matrix(n: int) -> np.ndarray:
    """The orthonormal DCT-II matrix `C`, built by transforming the identity.

    Deriving DCT-II from an FFT by hand (Makhoul reordering plus a twiddle) is
    the usual way to get one where none is provided, and it is also an easy way
    to get a normalisation off by a factor of `sqrt(2)` in the k=0 row only.
    That particular slip does not raise: it produces a smooth, plausible
    pressure field with a wrong constant mode, in a solve that is *already*
    singular in the constant mode. This case study has spent enough sessions on
    errors that looked like physics.

    So the matrix is taken from scipy directly -- `C[:, j] = dct(e_j)` -- which
    is correct by construction and checked against scipy again in the tests.
    `C` is orthogonal, so the inverse transform is its transpose, and at n=128
    it is a 131 KB constant. The matmul form is also the *right* shape for a
    GPU: unlike the stencils it is compute-bound, which is where a GPU has its
    other large advantage.
    """
    return _scipy_dct(np.eye(n, dtype=np.float64), type=2, axis=0, norm="ortho")


class NumpyBackend:
    """The reference path. Every method is the numpy call it replaces."""

    name = "numpy"
    device = "cpu"
    is_torch = False

    # -- construction / interop -------------------------------------------
    def asarray(self, x) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def array(self, x) -> np.ndarray:
        return np.array(x, dtype=np.float64, copy=True)

    def to_numpy(self, x) -> np.ndarray:
        return np.asarray(x)

    def empty_like(self, x) -> np.ndarray:
        return np.empty_like(x)

    def bool_asarray(self, x) -> np.ndarray:
        return np.asarray(x, dtype=bool)

    # -- elementwise -------------------------------------------------------
    def where(self, c, a, b):
        return np.where(c, a, b)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def hypot(self, a, b):
        return np.hypot(a, b)

    def abs(self, x):
        return np.abs(x)

    def to_float(self, x):
        return np.asarray(x, dtype=np.float64)

    # -- reductions --------------------------------------------------------
    def amax(self, x) -> float:
        return float(np.max(x))

    def absmax(self, x) -> float:
        return float(np.abs(x).max())

    def all_finite(self, x) -> bool:
        return bool(np.isfinite(x).all())

    def all_true(self, x) -> bool:
        return bool(np.all(x))

    def mean_keepdims(self, x, axes):
        return x.mean(axis=tuple(axes), keepdims=True)

    def sum_axis(self, x, axis):
        return x.sum(axis)

    # -- transforms --------------------------------------------------------
    def prepare_dct(self, n: int) -> None:
        """No-op: scipy's DCT needs no precomputed operator."""

    def dct2(self, x):
        """Orthonormal DCT-II over the last two axes of `[B, n, n]`."""
        return _scipy_dct(_scipy_dct(x, type=2, axis=2, norm="ortho"),
                          type=2, axis=1, norm="ortho")

    def idct2(self, x):
        return _scipy_idct(_scipy_idct(x, type=2, axis=1, norm="ortho"),
                           type=2, axis=2, norm="ortho")


class TorchBackend:
    """The same operations in torch, so `device='cuda'` is all that changes.

    Every divergence from numpy's spelling below is a real API difference, not
    a preference: `torch.maximum` will not take a python float, `clip` is
    `clamp`, reductions take `dim`/`keepdim`, and `torch.where` wants its
    branches as tensors on the same device. Each is normalised here so
    `WindowNS` never has to know which backend it is running on.
    """

    name = "torch"
    is_torch = True

    def __init__(self, device: str = "cpu"):
        import torch  # lazy: a numpy run must not require torch to be installed

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float64
        self._C = None          # DCT matrix, materialised on first use

    # -- construction / interop -------------------------------------------
    def asarray(self, x):
        t = self.torch
        if isinstance(x, t.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return t.as_tensor(np.ascontiguousarray(np.asarray(x, dtype=np.float64)),
                           dtype=self.dtype, device=self.device)

    def array(self, x):
        return self.asarray(x).clone()

    def to_numpy(self, x) -> np.ndarray:
        if isinstance(x, self.torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def empty_like(self, x):
        return self.torch.empty_like(x)

    def bool_asarray(self, x):
        t = self.torch
        if isinstance(x, t.Tensor):
            return x.to(device=self.device, dtype=t.bool)
        return t.as_tensor(np.ascontiguousarray(np.asarray(x, dtype=bool)),
                           device=self.device)

    # -- elementwise -------------------------------------------------------
    def _t(self, x):
        """Promote a python/numpy scalar so `where`/`maximum` will accept it."""
        if isinstance(x, self.torch.Tensor):
            return x
        return self.torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def where(self, c, a, b):
        c = c if isinstance(c, self.torch.Tensor) else self.bool_asarray(c)
        return self.torch.where(c, self._t(a), self._t(b))

    def maximum(self, a, b):
        return self.torch.maximum(self._t(a), self._t(b))

    def clip(self, x, lo, hi):
        return self.torch.clamp(self._t(x), min=lo, max=hi)

    def hypot(self, a, b):
        return self.torch.hypot(self._t(a), self._t(b))

    def abs(self, x):
        return self.torch.abs(self._t(x))

    def to_float(self, x):
        return self._t(x).to(self.dtype)

    # -- reductions --------------------------------------------------------
    def amax(self, x) -> float:
        return float(self.torch.max(self._t(x)).item())

    def absmax(self, x) -> float:
        return float(self.torch.abs(self._t(x)).max().item())

    def all_finite(self, x) -> bool:
        return bool(self.torch.isfinite(self._t(x)).all().item())

    def all_true(self, x) -> bool:
        return bool(self.torch.all(x).item())

    def mean_keepdims(self, x, axes):
        return x.mean(dim=tuple(axes), keepdim=True)

    def sum_axis(self, x, axis):
        return x.sum(dim=axis)

    # -- transforms --------------------------------------------------------
    def prepare_dct(self, n: int) -> None:
        if self._C is None or self._C.shape[0] != n:
            self._C = self.asarray(dct_matrix(n))

    def dct2(self, x):
        """`C x C^T` -- the separable DCT-II over the last two axes."""
        self.prepare_dct(x.shape[-1])
        C = self._C
        return C @ x @ C.T

    def idct2(self, x):
        """`C^T y C`. `C` is orthogonal, so this is the exact inverse."""
        self.prepare_dct(x.shape[-1])
        C = self._C
        return C.T @ x @ C


def get_backend(name: str = "numpy", device: str = "cpu") -> Any:
    """`name` in {'numpy', 'torch'}; `device` is meaningful only for torch."""
    if name == "numpy":
        if device not in ("cpu", None):
            raise ValueError(f"the numpy backend has no device {device!r}; "
                             f"use backend='torch' for CUDA")
        return NumpyBackend()
    if name == "torch":
        return TorchBackend(device=device)
    raise ValueError(f"unknown backend {name!r} (expected 'numpy' or 'torch')")
