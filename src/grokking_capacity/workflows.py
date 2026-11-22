"""Run experiment and analysis workflows grouped by purpose.

Use --list, or select a group and command. Arguments after the command are
passed through unchanged. Training commands require the registered local data
and hardware; listing commands does not import or execute them.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys

from .utils.workflow_paths import ROOT, catalog


def main(argv=None):
    commands = catalog()
    parser = argparse.ArgumentParser(prog="gc-workflow", description=__doc__)
    parser.add_argument("--list", action="store_true", help="List available commands")
    parser.add_argument("group", nargs="?", choices=sorted(commands))
    parser.add_argument("command", nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.list or args.command is None:
        for group, entries in sorted(commands.items()):
            if args.group and args.group != group:
                continue
            print(group + ":")
            for name, entry in sorted(entries.items()):
                print(f"  {name:42} {entry['description']}")
        return
    if args.command not in commands[args.group]:
        parser.error(f"Unknown {args.group} command: {args.command}")
    path = ROOT / commands[args.group][args.command]["path"]
    previous_args, previous_cwd = sys.argv, Path.cwd()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    try:
        sys.argv = [str(path), *arguments]
        os.chdir(ROOT)
        if path.is_relative_to(ROOT / "src"):
            module = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)
            runpy.run_module(module, run_name="__main__")
        else:
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = previous_args
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
