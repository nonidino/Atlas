"""One adapter per physics family: how a case file of that family is marched.

Each module here turns a committed case (`spec.CaseSpec`) into something the
runner (`runner.py`) can step or solve, decomposed and on the undivided domain.
There is no universal stepper: each family knows its own solver, its own
coupling style and its own sanity checks, and the registry (`registry.py`)
names the module for each family.
"""
