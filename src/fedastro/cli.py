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

    run = sub.add_parser("run", help="run one experiment for one seed")
    common(run)
    run.add_argument("--seed", type=int, help="override seed (default from config)")

    sweep = sub.add_parser("sweep", help="run one experiment for several seeds, sequentially")
    common(sweep)
    sweep.add_argument("--seeds", type=int, nargs="+", help="override seeds list")

    summ = sub.add_parser("summarize", help="mean ± std across seeds from result.json files")
    summ.add_argument("results", nargs="+", help="result.json paths (one experiment, several seeds)")
    summ.add_argument("--out", help="also write the summary JSON here")

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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "summarize":
        _summarize([Path(p) for p in args.results], Path(args.out) if args.out else None)
        return 0
    try:
        cfg = load_config(args.config)
        overrides = {"output_dir": args.output_dir, "device": args.device}
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
