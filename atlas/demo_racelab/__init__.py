"""PoC 3 -- RaceLab: a car as a graph of interchangeable physics experts.

    python -m atlas.demo_racelab --open

Then open <http://127.0.0.1:8013/>.

**The car is the stage; the point is the switch.**  Every fluid window can be
flipped between `reference.WindowNS` and Poseidon-T, and what the flip costs in
speed and in accuracy is shown per window and for the whole machine -- including
the two things that make the honest answer a negative here, which the screen
states rather than hides:

  * **the lead ratio.**  The composition layer exchanges four times a
    macro-step, so a learned window is asked for a step **1/32** of the one the
    checkpoint was trained to take, and CS-20 measured the error at a minimum
    AT the native lead and 4.3x that minimum at an eighth of it.
  * **the envelope.**  A learned column leaves the fluid expert's own declared
    bound within a handful of macro-steps, and the page turns red and says so
    rather than showing a field the model declines to stand behind.

See `README.md` for what is on the screen and what each number rests on.
"""

from .engine import Engine, RaceConfig                          # noqa: F401
from .server import create_app, run                             # noqa: F401

__all__ = ["Engine", "RaceConfig", "create_app", "run"]
