"""Command line for `rubinwork.studies`: ``index``, ``check``, ``new``, ``variants``,
``wrap-gather`` and ``wrap-apply``."""

import argparse
import json
import pathlib
import sys

from . import core, index, new, wrap


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
    gather = sub.add_parser("wrap-gather", help="what /wrap writes the entry from")
    gather.add_argument("study", help="<topic>/<study>")
    put = sub.add_parser("wrap-apply", help="insert the Log entry, update, commit")
    put.add_argument("study", help="<topic>/<study>")
    put.add_argument("--title", default=None, help="Log entry title; none if omitted")
    put.add_argument("--body-file", default=None, help="Markdown file with the entry")
    put.add_argument("--date", default=None)
    put.add_argument("--status", default=None, choices=core.STATUSES)
    put.add_argument("--answer", default=None)
    put.add_argument("--uses", action="append", default=[],
                     metavar="PRODUCT=VARIANT[@BUILD]")
    put.add_argument("--figure", action="append", default=[])
    put.add_argument("--no-commit", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "wrap-gather":
        print(json.dumps(wrap.gather(args.study).as_dict(), indent=1))
        return 0
    if args.command == "wrap-apply":
        uses = core.format_uses(new.parse_uses_arg(u) for u in args.uses)
        body = pathlib.Path(args.body_file).read_text() if args.body_file else None
        sha = wrap.apply(args.study, args.title, body, date=args.date,
                         status=args.status, answer=args.answer, uses=uses,
                         figures=args.figure, commit=not args.no_commit)
        print(f"committed {sha}, not pushed" if sha else "no commit")
        return 0

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
