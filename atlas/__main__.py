"""Run the compiler on a declared case and print (or persist) its artifact.

    python -m atlas wind-farm
    python -m atlas wind-farm --boundary-capable --declare-transfer
    python -m atlas rocket --staging
    python -m atlas rocket --json out/rocket.json

The artifact is written even when the verdict is a refusal -- especially then.
A refusal that reports only that it refused is a refusal nobody can act on.
"""

from __future__ import annotations

import argparse
import sys

from .cases import rocket, wind_farm
from .compiler import compile_scheme
from .verdict import REFUSE

CASES = {
    "wind-farm": lambda a: wind_farm.build(
        boundary_capable=a.boundary_capable, declare_transfer=a.declare_transfer
    ),
    "rocket": lambda a: rocket.build(with_staging=a.staging),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atlas", description=__doc__)
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--boundary-capable", action="store_true",
                        help="wind farm: swap the frozen checkpoint for one with a "
                             "boundary channel, so R2 can lift the rung")
    parser.add_argument("--declare-transfer", action="store_true",
                        help="wind farm: derive the interface space from the declared "
                             "effective resolutions")
    parser.add_argument("--primal-cross-points", action="store_true",
                        help="declare primal cross-point degrees of freedom, which a "
                             "non-overlapping decomposition needs")
    parser.add_argument("--staging", action="store_true",
                        help="rocket: add the stage-separation topology event")
    parser.add_argument("--json", metavar="PATH", help="write the run artifact as JSON")
    parser.add_argument("--decisions", action="store_true",
                        help="print every decision, not only failures")
    args = parser.parse_args(argv)

    graph = CASES[args.case](args)
    if args.primal_cross_points:
        graph.primal_cross_point_dofs = tuple(a.agent_id for a in graph.agents)

    result = compile_scheme(graph)
    print(result.report())
    if args.decisions:
        print()
        print("full decision record:")
        print(result.decisions.report())
    if args.json:
        print()
        print(f"artifact written to {result.artifact.write(args.json)}")

    # A refusal is a successful compile that says no. Exit 1 so a pipeline that
    # runs this cannot mistake it for an admission.
    return 1 if result.verdict is REFUSE else 0


if __name__ == "__main__":
    sys.exit(main())
