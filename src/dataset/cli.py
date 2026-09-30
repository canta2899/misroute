"""Command line entry point.

    ds plan --scale 0.02 --name pilot   # decide what should exist
    ds facts --direct                   # constraints per scenario
    ds generate --kind single --direct  # fill the gaps in the plan
    ds audit --kind single              # oracle: are the labels recoverable
    ds analyze --kind single            # baselines, pair quality, flags
    ds repair --kind single --regenerate
    ds export                           # dataset/<kind>.csv + <kind>.meta.json

Every command targets the most recent run unless `--run` says otherwise, and
every command is safe to re-run: generation and audit only do what is missing,
analysis only measures, repair only deletes.
"""

import argparse
import json
import sys
from pathlib import Path

from . import analyze as analyze_
from . import audit as audit_
from . import config, csvexport, generate, plan
from . import repair as repair_
from .store import (
    KINDS,
    Run,
    load_scenarios,
    new_run_id,
    open_run,
    read_jsonl,
    write_jsonl,
)

_KIND_ALIASES = {"single": "single_turn", "multi": "multi_turn"}


def _kind(value: str) -> str:
    return _KIND_ALIASES.get(value, value)


def _dump(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _mode(args) -> str:
    return "direct" if args.direct else "batch"


def _report_failures(result: dict) -> None:
    for key in ("failures", "missing_facts", "partial_requests"):
        values = result.get(key)
        if values:
            print(f"{key}: {len(values)}")
            _dump(values[:10] if key == "failures" else values[:20])


# --- commands ----------------------------------------------------------------


def cmd_plan(args) -> None:
    run = Run(args.name or new_run_id())
    if run.plan.exists() and not args.force:
        sys.exit(f"{run.plan} already exists; pass --force to rebuild it")
    kinds = [_kind(k) for k in args.kinds.split(",")]
    rows = plan.build(load_scenarios(), kinds=kinds, scale=args.scale)
    write_jsonl(run.plan, rows)
    print(f"[{run.id}] planned {len(rows)} samples -> {run.plan}")
    _dump(plan.summarize(rows))


def cmd_facts(args) -> None:
    run = open_run(args.run)
    result = generate.generate_facts(run, mode=_mode(args), limit=args.limit)
    print(
        f"[{run.id}] facts: {result['written']} written, "
        f"{result['already']} already present, {len(result['failures'])} failed"
    )
    _report_failures(result)


def cmd_generate(args) -> None:
    run = open_run(args.run)
    kind = _kind(args.kind)
    result = generate.generate_samples(
        run,
        kind,
        mode=_mode(args),
        limit=args.limit,
        ids=set(args.ids.split(",")) if args.ids else None,
    )
    print(
        f"[{run.id}] {kind}: {result['written']} samples from {result['requests']} "
        f"requests, {result['already']} already present, {len(result['failures'])} failed"
    )
    _report_failures(result)


def cmd_audit(args) -> None:
    run = open_run(args.run)
    kind = _kind(args.kind)
    result = audit_.audit(
        run,
        kind,
        mode=_mode(args),
        limit=args.limit,
        ids=set(args.ids.split(",")) if args.ids else None,
        redo=args.redo,
    )
    print(
        f"[{run.id}] {kind} oracle: {result['written']} judged, "
        f"{result['already']} already present, {len(result['failures'])} failed"
    )
    _report_failures(result)
    report = audit_.summarize(run, kind)
    block = report.get("oracle")
    if block:
        print(f"  oracle: accuracy {block['accuracy']} over {block['n']}")
    if report["unjudged"]:
        print(f"  still unjudged: {len(report['unjudged'])}")
    if report["disputed"]:
        print(f"  kept against the oracle by hand: {len(report['disputed'])}")
    print(f"  next: ds analyze --kind {args.kind} to refresh the regenerate flags")


def cmd_analyze(args) -> None:
    run = open_run(args.run)
    kind = _kind(args.kind)
    result = analyze_.analyze(run, kind)
    print(
        f"[{run.id}] {kind}: {result['n_valid']}/{result['n_samples']} structurally valid"
    )
    print("\nbaselines (user turns, grouped CV by scenario):")
    block = result["baselines"]["user_turns"]
    for name in ("lexical_bow", "embedding"):
        row = block.get(name)
        if row:
            per = " ".join(
                f"{c}={row[c]}"
                for c in config.CONSTRUCTION_IDS
                if row.get(c) is not None
            )
            print(f"  {name:12s} overall={row['overall']}  {per}")
    print(f"  chance={block['chance']}")

    pairs = result["pairs"]
    print(
        f"\npairs: {pairs['n_pairs']}, mean overlap {pairs['mean_overlap']}, "
        f"{pairs['n_weak']} below {pairs['min_overlap_threshold']}"
    )

    print(f"\nmarked for regeneration: {result['n_regenerate']}")
    for reason, n in result["reasons"].items():
        if n:
            print(f"  {reason:20s} {n}")
    print(f"\nreports -> {result['reports_dir']}")


def cmd_repair(args) -> None:
    run = open_run(args.run)
    kind = _kind(args.kind)
    if args.drop_plan and args.regenerate:
        sys.exit("--drop-plan and --regenerate contradict each other")
    result = repair_.repair(
        run,
        kind,
        request_ids=args.request.split(",") if args.request else None,
        sample_ids=args.id.split(",") if args.id else None,
        drop_plan=args.drop_plan,
    )
    if not result["removed"]:
        print("nothing to repair")
        return
    print(
        f"[{run.id}] {kind}: removed {result['removed']} samples "
        f"({len(result['removed_requests'])} requests); {result['kept']} kept"
    )
    print(f"backup -> {result['backup']}")
    if result["plan_rows_dropped"]:
        print(
            f"dropped {result['plan_rows_dropped']} plan rows: these samples "
            "will not be regenerated"
        )
    elif not args.regenerate:
        print("plan rows kept: the next `generate` recreates these samples")
    if args.regenerate:
        args.limit, args.ids = None, None
        cmd_generate(args)
    else:
        print(f"next: ds generate --kind {kind}")


def cmd_export(args) -> None:
    run = open_run(args.run)
    written = csvexport.export(
        run, args.out, edited=args.edited, keep_flagged=args.keep_flagged
    )
    if not written:
        sys.exit(f"[{run.id}] no samples to export")
    for w in written:
        print(f"[{run.id}] {w['kind']}: {w['rows']} rows ({w['dropped']} flagged dropped) -> {w['csv']}")


def cmd_status(args) -> None:
    run = open_run(args.run)
    rows = read_jsonl(run.plan)
    facts = read_jsonl(run.facts)
    print(f"run {run.id}  ({run.dir})")
    print(f"  facts   {len(facts)}/{len(load_scenarios())} scenarios")
    for kind in KINDS:
        planned = plan.by_kind(rows, kind)
        if not planned:
            continue
        samples = read_jsonl(run.samples(kind))
        judged = sum(1 for s in samples if s.get("oracle_predicted"))
        marked = sum(1 for s in samples if s.get("regenerate"))
        print(
            f"  {kind:12s} {len(samples)}/{len(planned)} generated, "
            f"judged {judged}, marked for regeneration {marked}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ds",
        description="Constraint-based expert-routing dataset generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run", help="run id under data/ (default: most recent)")
    sub = parser.add_subparsers(dest="command", required=True)

    def kind_arg(p, required=True):
        p.add_argument(
            "--kind",
            required=required,
            default="single_turn",
            choices=("single", "single_turn", "multi", "multi_turn"),
        )

    def llm_args(p):
        p.add_argument(
            "--direct",
            action="store_true",
            help="one call at a time instead of the batch API",
        )
        p.add_argument("--limit", type=int, help="stop after N requests (pilots)")

    p = sub.add_parser("plan", help="decide the whole dataset up front (creates a run)")
    p.add_argument("--name", help="run folder name (default: timestamp)")
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="scale every target, e.g. 0.02 for a pilot",
    )
    p.add_argument("--kinds", default="single_turn,multi_turn")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("facts", help="generate the five constraints per scenario")
    llm_args(p)
    p.set_defaults(func=cmd_facts)

    p = sub.add_parser("generate", help="generate every planned sample that is missing")
    kind_arg(p)
    llm_args(p)
    p.add_argument("--ids", help="comma-separated plan ids to generate")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser(
        "audit",
        help="route each sample with the rubric (oracle): are the labels recoverable",
    )
    kind_arg(p)
    llm_args(p)
    p.add_argument("--ids", help="comma-separated sample ids to audit")
    p.add_argument(
        "--redo",
        action="store_true",
        help="re-judge samples that already have a verdict (use after hand-editing)",
    )
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("analyze", help="baselines, pair quality, duplicates, flags")
    kind_arg(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser(
        "repair", help="delete samples marked `regenerate` so they can be rebuilt"
    )
    kind_arg(p)
    p.add_argument("--request", help="comma-separated request ids to drop")
    p.add_argument(
        "--id", help="comma-separated sample ids to drop (drops their pair too)"
    )
    p.add_argument(
        "--regenerate", action="store_true", help="generate the gaps straight after"
    )
    p.add_argument(
        "--drop-plan",
        action="store_true",
        help="also delete the plan rows, so these samples are gone for good",
    )
    p.add_argument("--direct", action="store_true")
    p.set_defaults(func=cmd_repair)

    p = sub.add_parser(
        "export", help="write dataset/<kind>.csv + <kind>.meta.json for simulate and analysis"
    )
    p.add_argument("-o", "--out", type=Path, default=Path("dataset"), help="destination folder")
    p.add_argument(
        "--edited", action="store_true", help="read samples.edited.jsonl where it exists"
    )
    p.add_argument(
        "--keep-flagged", action="store_true", help="keep samples marked regenerate"
    )
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("status", help="what exists and what is missing")
    p.set_defaults(func=cmd_status)

    return parser


def main(argv=None) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ValueError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
