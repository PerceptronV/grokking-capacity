"""Replay an exact archived confirmation controller with its original helpers."""
import argparse
import importlib
from pathlib import Path
import runpy
import sys
import types

CONTROLLERS = ("confirm_discrete_p149", "confirm_discrete_20261122",
               "confirm_discrete_unrestricted_20261122")
HELPERS = {"assess_discrete_match": "capacity",
           "discrete_source_compatibility": "compatibility",
           "discrete_status": "status", "confirm_discrete": "confirmation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controller", choices=CONTROLLERS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    names = ["scripts", *("scripts." + name for name in HELPERS)]
    previous = {name: sys.modules.get(name) for name in names}
    previous_args = sys.argv
    try:
        if "scripts" not in sys.modules:
            parent = types.ModuleType("scripts")
            parent.__path__ = []
            sys.modules["scripts"] = parent
        for old, new in HELPERS.items():
            sys.modules["scripts." + old] = importlib.import_module(
                "grokking_capacity.experiments.discrete_workflow." + new)
        path = Path(__file__).with_name(args.controller + ".py")
        sys.argv = [str(path), *args.arguments]
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = previous_args
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


if __name__ == "__main__":
    main()
