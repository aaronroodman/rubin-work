"""Command line for `rubinwork.studies`: ``index`` and ``check``."""

import argparse
import sys

from . import core, index


def main(argv=None):
    """Run one subcommand.

    Parameters
    ----------
    argv : `list` [`str`], optional
        Arguments, ``sys.argv[1:]`` by default.

    Returns
    -------
    status : `int`
        Process exit status: 0, or 1 if ``check`` found an error.
    """
    parser = argparse.ArgumentParser(prog="python -m rubinwork.studies")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index", help="write STUDIES.md at the repo root")
    check = sub.add_parser("check", help="validate every study.md")
    check.add_argument("topic", nargs="?", default=None)
    check.add_argument("--no-catalog", action="store_true",
                       help="do not resolve uses: entries through the catalog")
    args = parser.parse_args(argv)

    if args.command == "index":
        changed = index.write_index()
        print(f"{core.repo_root() / 'STUDIES.md'}: {'written' if changed else 'unchanged'}")
        return 0

    n_errors = 0
    for path in core.find_studies(topic=args.topic):
        try:
            study = core.load_study(path)
        except ValueError as exc:
            print(f"error: {path}: {exc}")
            n_errors += 1
            continue
        for problem in core.validate(study, check_catalog=not args.no_catalog):
            print(problem)
            n_errors += problem.level == "error"
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
