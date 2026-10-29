"""Single entrypoint: ``fedastro run <config>`` and ``fedastro sweep <config>``.

Non-interactive by design so it runs unchanged on a headless cluster.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fedastro.config import ConfigError, load_config
from fedastro.manifest import dumps
from fedastro.metrics.aggregate import format_summary, load_results, summarise
from fedastro.pipeline import build_federation, format_distribution_table, run_and_write


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

    sweep = sub.add_parser("sweep", help="run one experiment for several seeds, sequentially")
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

    inspect = sub.add_parser("inspect", help="print the per-silo class distribution; writes nothing")
    common(inspect)
    inspect.add_argument("--seed", type=int, help="override seed (default from config)")
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
    from fedastro.manifest import make_run_id

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
        for seed in cfg.seeds:
            path = Path(cfg.output_dir) / make_run_id(cfg, seed) / "result.json"
            if not (args.resume and path.exists()):
                try:
                    path = run_and_write(cfg, seed) / "result.json"
                except Exception as e:  # one failed run must not abort a long sweep
                    failed.append(f"{v.name} seed {seed}: {type(e).__name__}: {e}")
                    print(f"FAILED {failed[-1]}", file=sys.stderr)
                    continue
            paths.append(path)
            print(path)
        if len(paths) == len(cfg.seeds):
            _summarize(paths, Path(cfg.output_dir) / f"{cfg.name}-{cfg.config_hash()[:10]}-summary.json")
        else:
            print(f"SKIPPED summary for {v.name}: {len(paths)}/{len(cfg.seeds)} seeds succeeded", file=sys.stderr)
    if failed:
        print(f"{len(failed)} run(s) failed; re-run with --resume to fill gaps:", file=sys.stderr)
        for f in failed:
            print(f"  {f}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "summarize":
        _summarize([Path(p) for p in args.results], Path(args.out) if args.out else None)
        return 0
    if args.command == "grid":
        return _grid(args)
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

    if args.command == "inspect":
        bundle, fed = build_federation(cfg, cfg.seed)
        p = bundle.provenance
        print(f"dataset={p.dataset} kind={p.kind} version={p.version}")
        print(f"partition={cfg.partition.method} alpha={cfg.partition.alpha} seed={cfg.seed}")
        print(format_distribution_table(fed, bundle.class_names))
        return 0

    seeds = [cfg.seed] if args.command == "run" else list(cfg.seeds)
    paths = []
    for seed in seeds:
        run_dir = run_and_write(cfg, seed)
        paths.append(run_dir / "result.json")
        print(paths[-1])
    if args.command == "sweep":
        out = Path(cfg.output_dir) / f"{cfg.name}-{cfg.config_hash()[:10]}-summary.json"
        _summarize(paths, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
