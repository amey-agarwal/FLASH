"""Single entrypoint: ``fedastro run <config>`` and ``fedastro sweep <config>``.

Non-interactive by design so it runs unchanged on a headless cluster.
"""

from __future__ import annotations

import argparse
import sys

from fedastro.config import ConfigError, load_config
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

    inspect = sub.add_parser("inspect", help="print the per-silo class distribution; writes nothing")
    common(inspect)
    inspect.add_argument("--seed", type=int, help="override seed (default from config)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
    for seed in seeds:
        run_dir = run_and_write(cfg, seed)
        print(run_dir / "result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
