"""Single entrypoint: ``fedastro run <config>``, ``fedastro sweep <config>``, ``fedastro peek <run_dir>``.

Non-interactive by design so it runs unchanged on a headless cluster.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fedastro.config import ConfigError, load_config
from fedastro.manifest import dumps
from fedastro.metrics.aggregate import format_summary, load_results, summarise
from fedastro.pipeline import (build_federation, build_hier_federation,
                               format_distribution_table, format_node_table, run_and_write)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fedastro", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("config", help="path to an experiment YAML")
        sp.add_argument("--output-dir", help="override output_dir (default from config)")
        sp.add_argument("--device", choices=("cpu", "cuda"), help="override device")
        sp.add_argument("--synthetic", action="store_true",
                        help="force the SYNTHETIC smoke-test dataset (results tagged SYNTHETIC)")
        sp.add_argument("--real-sleep", type=float, nargs="?", const=1.0, metavar="SCALE",
                        help="LIVE DEMO ONLY: actually sleep simulated link time x SCALE (default 1.0); "
                             "results are unchanged, only slower")

    run = sub.add_parser("run", help="run one experiment for one seed")
    common(run)
    run.add_argument("--seed", type=int, help="override seed (default from config)")
    run.add_argument("--draw", type=int, default=0, help="holdout draw (needs evaluation.holdout_draws > 1)")

    sweep = sub.add_parser("sweep", help="run one experiment for several seeds (x holdout draws), sequentially")
    common(sweep)
    sweep.add_argument("--seeds", type=int, nargs="+", help="override seeds list")

    summ = sub.add_parser("summarize", help="mean ± std across seeds from result.json files")
    summ.add_argument("results", nargs="+", help="result.json paths (one experiment, several seeds)")
    summ.add_argument("--out", help="also write the summary JSON here")

    grid = sub.add_parser("grid", help="run a grid sweep (variants x seeds) from a sweep YAML")
    grid.add_argument("sweep", help="path to a sweep YAML (configs/sweeps/)")
    grid.add_argument("--list", action="store_true", help="list variants and exit")
    grid.add_argument("--index", type=int, nargs="+", help="run only these variant indices")
    grid.add_argument("--resume", action="store_true", help="skip runs whose result.json exists")
    grid.add_argument("--output-dir", help="override output_dir")
    grid.add_argument("--report", action="store_true",
                      help="collect existing per-variant summaries into one table; runs nothing")

    sel = sub.add_parser("select", help="pick a tuning sweep's winner per arm from VALIDATION scores only "
                                        "(Phase H8; never reads test results)")
    sel.add_argument("sweep", help="the tuning sweep YAML whose runs exist")
    sel.add_argument("--arms", nargs="+", required=True,
                     choices=("centralized", "local_only", "federated", "hierarchical"))
    sel.add_argument("--output-dir", help="where the runs are (default: the sweep's output_dir)")
    sel.add_argument("--emit-dir", help="also write one generated sweep per arm here")
    sel.add_argument("--template", help="with --emit-dir: next-stage axes (stage B); omit for the evaluation sweep")
    sel.add_argument("--eval-seeds", type=int, nargs="+", help="with --emit-dir and no --template: seeds")

    gap = sub.add_parser("gap", help="centralized vs federated / hierarchical gap table (Phase H8)")
    gap.add_argument("spec", help="YAML: {families: {<family>: {centralized: <sweep>, federated: <sweep>, "
                                  "hierarchical: <sweep>}}}, paths relative to the spec")
    gap.add_argument("--output-dir", help="where the runs are (default: each sweep's output_dir)")
    gap.add_argument("--out", help="also write the table as JSON here")

    peek = sub.add_parser("peek", help="read-only terminal view of a hierarchical run (writes nothing)")
    peek.add_argument("run_dir", help="runs/<run_id> (or its result.json)")
    peek.add_argument("--follow", action="store_true", help="tail a run in progress until it ends")
    peek.add_argument("--interval", type=float, default=1.0, help="--follow poll interval in real seconds")
    peek.add_argument("--timeout", type=float, help="--follow: give up after this many real seconds")

    inspect = sub.add_parser("inspect", help="print the per-silo class distribution; writes nothing")
    common(inspect)
    inspect.add_argument("--seed", type=int, help="override seed (default from config)")
    inspect.add_argument("--draw", type=int, default=0, help="holdout draw (needs evaluation.holdout_draws > 1)")
    return p


def _summarize(paths: list[Path], out: Path | None) -> None:
    summary = summarise(load_results(paths))
    print(format_summary(summary))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(dumps(summary))
        print(out)


def _grid(args: argparse.Namespace) -> int:
    from fedastro.grid import load_grid
    from fedastro.manifest import make_run_id, run_keys

    try:
        sweep_name, variants = load_grid(args.sweep)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if args.report:
        from fedastro.metrics.aggregate import grid_report

        out_dir = Path(args.output_dir or variants[0].config.output_dir)
        rows, text = grid_report(variants, out_dir)
        print(text)
        out = out_dir / f"{sweep_name}-grid.json"
        out.write_text(dumps({"sweep": sweep_name, "rows": rows}))
        print(out)
        return 0
    if args.list:
        for v in variants:
            print(f"[{v.index}] {v.name}  {v.overrides}  seeds={list(v.config.seeds)}")
        return 0
    chosen = [v for v in variants if args.index is None or v.index in args.index]
    if args.index and len(chosen) != len(set(args.index)):
        print(f"unknown variant index in {args.index}; have 0..{len(variants) - 1}", file=sys.stderr)
        return 2
    failed = []
    for v in chosen:
        cfg = v.config.with_overrides(output_dir=args.output_dir)
        paths = []
        keys = run_keys(cfg, cfg.seeds)
        for seed, draw in keys:
            path = Path(cfg.output_dir) / make_run_id(cfg, seed, draw) / "result.json"
            if not (args.resume and path.exists()):
                try:
                    path = run_and_write(cfg, seed, draw) / "result.json"
                except Exception as e:  # one failed run must not abort a long sweep
                    failed.append(f"{v.name} seed {seed} draw {draw}: {type(e).__name__}: {e}")
                    print(f"FAILED {failed[-1]}", file=sys.stderr)
                    continue
            paths.append(path)
            print(path)
        if len(paths) == len(keys):
            _summarize(paths, Path(cfg.output_dir) / f"{cfg.name}-{cfg.config_hash()[:10]}-summary.json")
        else:
            print(f"SKIPPED summary for {v.name}: {len(paths)}/{len(keys)} runs succeeded", file=sys.stderr)
    if failed:
        print(f"{len(failed)} run(s) failed; re-run with --resume to fill gaps:", file=sys.stderr)
        for f in failed:
            print(f"  {f}", file=sys.stderr)
        return 1
    return 0


def _select(args: argparse.Namespace) -> int:
    import yaml

    from fedastro.grid import load_grid
    from fedastro.tuning import emit_sweep, run_select

    try:
        name, variants = load_grid(args.sweep)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    out_dir = Path(args.output_dir or variants[0].config.output_dir)
    try:
        res = run_select(Path(args.sweep), out_dir, args.arms)
    except (FileNotFoundError, ValueError) as e:
        print(f"select: {e}", file=sys.stderr)
        return 2
    for arm, sel in res["selections"].items():
        w = sel["winner"]
        print(f"{arm}: winner {w['name']} (validation macro-F1 {w['mean']:.4f}; best {sel['best_mean']:.4f}; "
              f"{len(sel['candidates'])} candidates, seeds {sel['seeds']})")
    path = out_dir / f"{name}-selection.json"
    path.write_text(dumps(res))
    print(path)
    if args.emit_dir:
        template = yaml.safe_load(Path(args.template).read_text()) if args.template else None
        for arm, sel in res["selections"].items():
            stem = f"{template['name']}_{name}_{arm}" if template else f"eval_{name}_{arm}"
            target = Path(args.emit_dir) / f"{stem}.yaml"
            emit_sweep(Path(args.sweep), sel, target, stem, template, args.eval_seeds if not template else None)
            print(target)
    return 0


def _gap(args: argparse.Namespace) -> int:
    import yaml

    from fedastro.grid import load_grid
    from fedastro.manifest import make_run_id
    from fedastro.metrics.gap import format_gap, gap_rows

    spec_path = Path(args.spec)
    spec = yaml.safe_load(spec_path.read_text())
    rows = []
    for family, arms in spec["families"].items():
        runs = {}
        for arm, sweep in arms.items():
            _, variants = load_grid(spec_path.parent / sweep)
            if len(variants) != 1:
                print(f"gap: {sweep} must have exactly one variant (an evaluation sweep)", file=sys.stderr)
                return 2
            cfg = variants[0].config
            out_dir = Path(args.output_dir or cfg.output_dir)
            paths = [out_dir / make_run_id(cfg, s) / "result.json" for s in cfg.seeds]
            missing = [str(p) for p in paths if not p.is_file()]
            if missing:
                print(f"gap: missing runs {missing}", file=sys.stderr)
                return 2
            runs[arm] = load_results(paths)
        rows += gap_rows(family, runs)
    print(format_gap(rows))
    if args.out:
        Path(args.out).write_text(dumps({"spec": str(spec_path), "rows": rows}))
        print(args.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "summarize":
        _summarize([Path(p) for p in args.results], Path(args.out) if args.out else None)
        return 0
    if args.command == "grid":
        return _grid(args)
    if args.command == "select":
        return _select(args)
    if args.command == "gap":
        return _gap(args)
    if args.command == "peek":
        from fedastro.peek import follow, peek

        if args.follow:
            return follow(args.run_dir, interval=args.interval, timeout=args.timeout)
        return peek(args.run_dir)
    try:
        cfg = load_config(args.config)
        overrides = {"output_dir": args.output_dir, "device": args.device,
                     "real_sleep_scale": getattr(args, "real_sleep", None)}
        if args.synthetic and cfg.data.name != "synthetic":
            import dataclasses
            overrides["data"] = dataclasses.replace(cfg.data, name="synthetic")
        if args.command in ("run", "inspect"):
            overrides["seed"] = args.seed
        else:
            overrides["seeds"] = tuple(args.seeds) if args.seeds else None
        cfg = cfg.with_overrides(**overrides)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    if args.command in ("run", "inspect") and not 0 <= args.draw < cfg.evaluation.holdout_draws:
        print(f"config error: --draw {args.draw} not in [0, evaluation.holdout_draws = "
              f"{cfg.evaluation.holdout_draws})", file=sys.stderr)
        return 2
    if cfg.hierarchy is not None and args.command == "inspect":
        bundle, hfed, shift = build_hier_federation(cfg, cfg.seed)
        p, h = bundle.provenance, cfg.hierarchy
        print(f"dataset={p.dataset} kind={p.kind} version={p.version}")
        print(f"hierarchy={h.name} partition={h.partition} min_participants={h.min_participants} "
              f"seed={cfg.seed} (region, agency and node names are scenario labels)")
        print(format_node_table(hfed, bundle.class_names))
        from fedastro.data.shift import format_shift_table

        print(format_shift_table(shift))
        return 0
    if args.command == "inspect":
        bundle, fed = build_federation(cfg, cfg.seed, args.draw)
        p = bundle.provenance
        print(f"dataset={p.dataset} kind={p.kind} version={p.version}")
        print(f"partition={cfg.partition.method} alpha={cfg.partition.alpha} seed={cfg.seed}")
        print(format_distribution_table(fed, bundle.class_names))
        return 0

    from fedastro.manifest import run_keys

    keys = [(cfg.seed, args.draw)] if args.command == "run" else run_keys(cfg, cfg.seeds)
    paths = []
    for seed, draw in keys:
        run_dir = run_and_write(cfg, seed, draw)
        paths.append(run_dir / "result.json")
        print(paths[-1])
    if args.command == "sweep":
        out = Path(cfg.output_dir) / f"{cfg.name}-{cfg.config_hash()[:10]}-summary.json"
        _summarize(paths, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
