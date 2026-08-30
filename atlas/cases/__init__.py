"""Case-study fixtures.

These are **declarations only**.  Every module here builds a ``CaseGraph`` and
returns it; none of them contains coupling code, and none of them is imported by
anything in the package proper.  They exist so the compiler has something
concrete to be tested against, and they are fixtures rather than the target of
the build: the point of the package is that a new case study is another file
exactly like these.

``wind_farm`` and ``rocket`` are **fixtures**: their ``boundary_response`` is a
matrix somebody wrote down.  The **real case studies** call an actual solver or
checkpoint, and only they can move a Tier 0 row.  There are six:

    window_ns       four reference.WindowNS windows; the whole Tier 0 stack, and
                    the only graph that reaches `admit`
    channel_ns      a second discretization -- whether a rule survives different
                    internals
    poseidon        a frozen 20.8M-parameter neural operator -- whether a rule
                    survives having no internals
    wind_farm_real  the fixture's topology with real experts -- the port algebra
    thermal_seam    compressible gas against a thermoelastic shell -- the first
                    seam whose two sides do not solve the same equations, and the
                    only exercise E3 and the bc_channel ladder have ever had
                    against a genuine disagreement
    wake_array      three turbines at 3.5 D in an L over six frozen Poseidon-T
                    windows -- the first REAL GEOMETRY, the first graph whose
                    seams carry an ABSOLUTE trace, and the only place the Tier
                    14-16 attribution machinery has met a pretrained checkpoint
                    inside a composed wake

None of them is imported here, because importing one would make the package
depend on a checkout of another repository; import them explicitly.
"""

from . import rocket, wind_farm

__all__ = ["wind_farm", "rocket"]
