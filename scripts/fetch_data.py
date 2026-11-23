"""Download raw datasets into data/raw/. Non-interactive; safe to re-run.

    python scripts/fetch_data.py wesad --accept-license
    python scripts/fetch_data.py osd435 --accept-license
    python scripts/fetch_data.py cinc2021 --subset 2000 --accept-license   (0 = every record)

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
    "osd435": {  # one 66 KB CSV; its sha256 is also pinned in src/fedastro/data/osdr.py
        "url": "https://osdr.nasa.gov/geode-py/ws/studies/OSD-435/download?source=datamanager"
               "&file=LSDS-22_Echocardiography_aorta_SUBMITTED.csv",
        "filename": "LSDS-22_Echocardiography_aorta_SUBMITTED.csv",
        "landing_page": "https://osdr.nasa.gov/bio/repo/data/studies/OSD-435",
        "license_note": "OSD-435 (NASA OSDR, public): cite the study and OSDR; see landing page.",
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


CINC_LICENSE = ("PhysioNet/CinC Challenge 2021 v1.0.3 (open access, CC BY 4.0): cite the Challenge "
                "(Reyna et al. 2021) and the source databases; see https://physionet.org/content/challenge-2021/1.0.3/")


def _get(url: str, dest: Path, retries: int = 9) -> bool:
    """Download url to dest (atomic). False on HTTP 404; raises after retries otherwise.

    Backoff doubles up to 60 s (~5 min in all), enough to ride out a PhysioNet 502/503 spell."""
    import urllib.error

    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fedastro-fetch/1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            tmp.rename(dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(min(60, 2 ** (attempt + 1)))
    return False


def _get_many(jobs: list[tuple[str, Path]], what: str, workers: int = 8) -> list[Path]:
    """Download every (url, dest) whose dest is missing; returns the dests that were 404."""
    from concurrent.futures import ThreadPoolExecutor

    todo = [(u, d) for u, d in jobs if not d.exists()]
    print(f"{what}: {len(jobs) - len(todo)} present, {len(todo)} to download", flush=True)
    absent, done, last = [], 0, time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (url, dest), ok in zip(todo, pool.map(lambda j: _get(*j), todo)):
            done += 1
            if not ok:
                absent.append(dest)
            if time.monotonic() - last > 30:
                print(f"  {done}/{len(todo)}", flush=True)
                last = time.monotonic()
    return absent


def fetch_cinc2021(raw_root: Path, subset: int) -> None:
    """Metadata (pinned checksums), every header, then the signals of the chosen subset."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from fedastro.data import cinc2021 as c

    root = raw_root / "cinc2021"
    _get_many([(url, root / rel) for rel, url in c.METADATA.items()], "metadata")
    c.verify_metadata(root)
    lists = [(f"{c.BASE_URL}/{d}RECORDS", root / "lists" / (d.replace("/", "_") + "RECORDS"))
             for d in c.group_dirs(root)]
    if _get_many(lists, "record lists"):
        sys.exit("a record list is missing on PhysioNet; re-inspect before continuing")
    recs = c.listed_records(root)
    absent = _get_many([(f"{c.BASE_URL}/{r}.hea", root / f"{r}.hea") for r in recs], "headers")
    unavailable_file = root / "unavailable_records.txt"
    names = sorted({str(p.relative_to(root))[: -len(".hea")] for p in absent}
                   | set(c._read_unavailable(root) if unavailable_file.exists() else []))
    unavailable_file.write_text("".join(n + ".hea\n" for n in names))
    c._read_unavailable(root)  # refuses a set different from the pinned one
    rows = c.select_subset(c.build_index(root, c.Cinc2021Params().window_s), subset)
    if _get_many([(f"{c.BASE_URL}/{r['path']}.mat", root / f"{r['path']}.mat") for r in rows], "signals"):
        sys.exit("signal files of the subset are missing on PhysioNet; re-inspect before continuing")
    listing = root / f"subset_{subset}.txt"
    listing.write_text("".join(f"{r['path']}\t{r['source']}\t{r['patient']}\t{r['label']}\n" for r in rows))
    print(f"subset {subset}: {len(rows)} records, {len({r['patient'] for r in rows})} patients -> {listing} "
          f"(index {c.index_checksum(rows)})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=sorted([*DATASETS, "cinc2021"]))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--accept-license", action="store_true")
    p.add_argument("--subset", type=int, default=2000,
                   help="cinc2021 only: records to download signals for (0 = all usable records)")
    args = p.parse_args(argv)
    print(CINC_LICENSE if args.dataset == "cinc2021" else DATASETS[args.dataset]["license_note"])
    if not args.accept_license:
        print("refusing to download without --accept-license", file=sys.stderr)
        return 2
    if args.dataset == "cinc2021":
        fetch_cinc2021(args.raw_dir, args.subset)
    else:
        fetch(args.dataset, args.raw_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
