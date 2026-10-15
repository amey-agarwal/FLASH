"""Download raw datasets into data/raw/. Non-interactive; safe to re-run.

    python scripts/fetch_data.py wesad --accept-license

WESAD (Schmidt et al., ICMI 2018) is distributed by the University of Siegen for
non-commercial academic use. Passing --accept-license records that you have read and
accept the dataset's terms; nothing is downloaded without it.

The archive is ~2.25 GB. It is never extracted: the loader reads subject pickles
directly from the zip. Interrupted downloads resume from the .part file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

DATASETS = {
    "wesad": {
        "url": "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download",
        "filename": "WESAD.zip",
        "landing_page": "https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/",
        "license_note": "WESAD: non-commercial academic research use only; see landing page.",
    },
}

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "fedastro-fetch/1"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:
            have = 0  # server ignored Range; start over
        total = have + int(resp.headers.get("Content-Length", 0))
        mode = "ab" if have else "wb"
        done, last = have, 0.0
        with part.open(mode) as f:
            for block in iter(lambda: resp.read(CHUNK), b""):
                f.write(block)
                done += len(block)
                now = time.monotonic()
                if now - last > 10:
                    pct = f"{100 * done / total:.1f}%" if total else "?"
                    print(f"  {done / 1e9:.2f} GB ({pct})", flush=True)
                    last = now
    part.rename(dest)


def fetch(name: str, raw_dir: Path) -> Path:
    spec = DATASETS[name]
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / spec["filename"]
    meta_path = raw_dir / f"{spec['filename']}.meta.json"
    if dest.exists() and meta_path.exists():
        print(f"{dest} already present; verifying checksum")
        meta = json.loads(meta_path.read_text())
        if sha256_file(dest) != meta["sha256"]:
            sys.exit(f"checksum mismatch for {dest}; delete it and re-run")
        return dest
    if not dest.exists():
        print(f"downloading {name} from {spec['url']}")
        download(spec["url"], dest)
    meta = {
        "dataset": name,
        "url": spec["url"],
        "landing_page": spec["landing_page"],
        "bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"wrote {dest} ({meta['bytes']} bytes, sha256 {meta['sha256']})")
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=sorted(DATASETS))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--accept-license", action="store_true")
    args = p.parse_args(argv)
    print(DATASETS[args.dataset]["license_note"])
    if not args.accept_license:
        print("refusing to download without --accept-license", file=sys.stderr)
        return 2
    fetch(args.dataset, args.raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
