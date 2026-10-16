"""Command line for `rubinwork.studies`: ``index``, ``check``, ``new`` and ``variants``."""

import argparse
import sys

from . import core, index, new


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
    make = sub.add_parser("new", help="create <topic>/<study>/ (the /start-study skill)")
    make.add_argument("topic")
    make.add_argument("study")
    make.add_argument("--question", required=True)
    make.add_argument("--title", default=None)
    make.add_argument("--uses", action="append", default=[],
                      metavar="PRODUCT=VARIANT[@BUILD]")
    make.add_argument("--new-topic", action="store_true")
    sub.add_parser("variants", help="each product's variants.yaml and its variants")
    args = parser.parse_args(argv)

    if args.command == "new":
        try:
            uses = [new.parse_uses_arg(u) for u in args.uses]
            directory = new.create_study(args.topic, args.study, args.question, uses,
                                         title=args.title, new_topic=args.new_topic)
        except (ValueError, FileNotFoundError) as exc:
            print(f"not created: {exc}", file=sys.stderr)
            return 1
        print(f"created {directory}")
        print(f"output -> {(directory / 'output').readlink()}")
        return 0
    if args.command == "variants":
        for product, (path, names) in new.product_variants().items():
            print(f"{product}: {path.relative_to(core.repo_root())}")
            for name in names:
                print(f"  {name}")
        return 0

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
