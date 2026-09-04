"""Case-study fixtures.

These are **declarations only**.  Every module here builds a ``CaseGraph`` and
returns it; none of them contains coupling code, and none of them is imported by
anything in the package proper.  They exist so the compiler has something
concrete to be tested against, and they are fixtures rather than the target of
the build: the point of the package is that a new case study is another file
exactly like these.

``wind_farm`` and ``rocket`` are **fixtures**: their ``boundary_response`` is a
matrix somebody wrote down.  The **real case studies** call an actual solver or
checkpoint, and only they can move a Tier 0 row.  There are eleven:

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
    scaling_ladder  one geometry at five sizes -- the variable is the SIZE of the
                    graph, and the answer is the F1 criterion the whole programme
                    is scheduled on
    reuse_probe     the variable is the STATE a certificate is measured at, which
                    is the foundation-model claim rather than the coupling one
    thermal_strain  ThermoStruct2D split into a conduction agent and an elasticity
                    agent -- the first CO-LOCATED split, two agents on the same
                    mesh over the same region, and the first exercise of
                    PortAmendment's six-field procedure. It refuses: thermal
                    strain is a BOND and is not a PORT
    seam_placement  the variable is WHERE THE CUT GOES -- a search over
                    decompositions under a cost budget, graded against the
                    hand-chosen tilings every other case study here typed in
    ground_effect   a wing over a moving floor against a two-line algebraic
                    suspension -- the first graph whose interface GEOMETRY is a
                    function of the solution, so `motion_class` is
                    SOLUTION_DEPENDENT, L2 refuses, and InterfaceMotion's three
                    measurements exist for the first time. Also the first design
                    PARAMETER, differentiated through the composed stack

None of them is imported here, because importing one would make the package
depend on a checkout of another repository; import them explicitly.
"""

from . import rocket, wind_farm

__all__ = ["wind_farm", "rocket"]
