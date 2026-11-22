"""Shared campaign utilities; training planners remain gc-workflow commands."""
import argparse
from .status import status
from ..registered import export_legacy
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('status').add_argument('root', type=Path)
    sub.add_parser('export-legacy').add_argument('destination', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'status':
        print(json.dumps(status(args.root), indent=2))
    else:
        print(export_legacy(args.destination))


if __name__ == '__main__':
    main()
