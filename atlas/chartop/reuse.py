"""Arm H2r of `gates_squares`: the classical superelement, with one response per class of
squares.

On a square piece of one material, the operator in chart coordinates is the plain square's,
whatever the square's size and place: the conductivity enters through the host's
``kappa_ref`` only, and a step through its Fourier number and its source.  So squares that
share the resolution, the Fourier number and the source have the same classical response.
`ReuseExpert` solves one square of each class and answers every square of the class with
that response; every other piece gets its own classical solve.  The cache lives for one call
of `respond`, so its price is that of one coupled solve, not of a march that reuses it over
many steps.

A class's Fourier number is read to 12 significant digits, and a uniform source by its value,
because two squares in different places differ in the last bits of their computed areas.
The response shared is then wrong by about the last bits too, far under every bar.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .host import ClassicalExpert


class ReuseExpert:
    name = "classical, squares reused"
    envelope = None

    def __init__(self, is_square):
        self.is_square = is_square
        self.classical = ClassicalExpert()
        self.solves = 0

    @staticmethod
    def key(p) -> tuple:
        g = np.asarray(p.g, float)
        top = float(np.abs(g).max()) if g.size else 0.0
        if g.size == 0 or float(np.ptp(g)) <= 1e-12 * max(1.0, top):
            gk = f"uniform {float(g.flat[0]) if g.size else 0.0:.12g}"
        else:
            gk = hashlib.sha256(np.ascontiguousarray(g).tobytes()).hexdigest()
        return (p.n, "steady" if p.steady else f"{float(p.fo):.12g}", gk)

    def respond(self, problems: list, pb_of) -> list:
        cache, out = {}, []
        for p in problems:
            if self.is_square(p):
                k = self.key(p)
                if k not in cache:
                    cache[k] = self.classical.respond([p], pb_of)[0]
                    self.solves += 1
                out.append(cache[k])
            else:
                out.append(self.classical.respond([p], pb_of)[0])
                self.solves += 1
        return out
