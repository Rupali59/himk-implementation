"""
Download and prepare KTH Human Action dataset.
KTH has 6 classes: boxing, handclapping, handwaving, jogging, running, walking.
25 subjects; videos are typically 160x120, ~4 seconds.
Standard split: persons 1-16 train, 17-21 validation, 22-25 test (or similar).

Downloads are parallelised when multiple URLs are provided. File organisation (copy/split) is parallelised.
"""

import argparse
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

# Default: expect data in implementation/data/kth with structure:
#   kth/
#     person01_boxing_d1_uncomp.avi  ... (or in class subdirs after we organize)
KTH_CLASSES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

# Common split: 16 train, 4 val, 5 test (by person number)
TRAIN_PERSONS = list(range(1, 17))   # 1-16
VAL_PERSONS = list(range(17, 22))    # 17-21
TEST_PERSONS = list(range(22, 26))   # 22-25

# Single zip URL (may require manual download if link is dead). Add more URLs to download in parallel.
KTH_ZIP_URLS = [
    "https://www.csc.kth.se/cvap/actions/00sequences.zip",
]
# Optional: per-class or per-split URLs for parallel download, e.g.:
# KTH_ZIP_URLS = ["https://.../boxing.zip", "https://.../handclapping.zip", ...]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download_one(url: str, dest: Path) -> tuple[str, bool, str]:
    """Download one URL to dest. Returns (url, success, error_message)."""
    try:
        urlretrieve(url, dest)
        return (url, True, "")
    except Exception as e:
        return (url, False, str(e))


def download_urls_parallel(urls: list[str], dest_dir: Path, max_workers: int = 4) -> list[Path]:
    """
    Download multiple URLs in parallel. Each URL is saved to dest_dir / base_name(url).
    Returns list of successfully downloaded paths.
    """
    downloaded = []
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    def task(url: str):
        name = Path(url).name or "download"
        dest = dest_dir / name
        return _download_one(url, dest)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(task, url): url for url in urls}
        for fut in as_completed(futures):
            url, ok, err = fut.result()
            if ok:
                name = Path(url).name or "download"
                downloaded.append(dest_dir / name)
                print(f"  Downloaded: {url}")
            else:
                print(f"  Failed {url}: {err}", file=sys.stderr)
    return downloaded


def _copy_one(args: tuple[Path, Path]) -> tuple[Path, None]:
    """Copy file to dest; return (dest, None). Used for parallel organise."""
    src, dest = args
    if not dest.exists() or dest.stat().st_size != src.stat().st_size:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return (dest, None)


def organize_by_class_and_split(raw_dir: Path, out_dir: Path, max_workers: int = 8) -> None:
    """
    raw_dir contains files like person01_boxing_d1_uncomp.avi.
    Create out_dir/train/<class>/*.avi and out_dir/test/<class>/*.avi (and val).
    Copy operations are parallelised.
    """
    files = list(raw_dir.rglob("*.avi"))
    if not files:
        files = list(raw_dir.glob("*.avi")) + list(raw_dir.glob("*.AVI"))
    pattern = re.compile(r"person(\d+)_(\w+)_", re.I)
    tasks = []
    for f in files:
        m = pattern.search(f.name)
        if not m:
            continue
        person = int(m.group(1))
        action = m.group(2).lower()
        if action not in KTH_CLASSES:
            continue
        if person in TRAIN_PERSONS:
            split = "train"
        elif person in TEST_PERSONS:
            split = "test"
        else:
            split = "val"
        dest_dir = ensure_dir(out_dir / split / action)
        dest = dest_dir / f.name
        tasks.append((f, dest))

    if not tasks:
        print("No matching .avi files found in", raw_dir)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_copy_one, tasks))
    print("Organized into", out_dir)


def main():
    ap = argparse.ArgumentParser(description="Prepare KTH dataset for implementation")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output root (default: implementation/data/kth)")
    ap.add_argument("--raw-dir", type=Path, default=None, help="If you already have raw KTH .avi files in one dir, we organize them")
    ap.add_argument("--download", action="store_true", help="Try to download zip(s) (parallel if multiple URLs) and organize")
    ap.add_argument("--workers", type=int, default=4, help="Max parallel download workers (default 4)")
    ap.add_argument("--organize-workers", type=int, default=8, help="Max parallel copy workers for organise (default 8)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = args.out_dir or root / "data" / "kth"

    if args.raw_dir:
        raw = Path(args.raw_dir)
        if not raw.exists():
            print("Raw dir does not exist:", raw)
            sys.exit(1)
        organize_by_class_and_split(raw, out_dir, max_workers=args.organize_workers)
        print("Done. Set config data_dir to", out_dir)
        return

    if args.download:
        ensure_dir(out_dir.parent)
        zip_dir = out_dir.parent
        print("Downloading", len(KTH_ZIP_URLS), "URL(s) in parallel (workers=%d) ..." % args.workers)
        downloaded = download_urls_parallel(KTH_ZIP_URLS, zip_dir, max_workers=args.workers)
        if not downloaded:
            print("No files downloaded. Manual: download KTH from https://www.csc.kth.se/cvap/actions/")
            print("Then run: python scripts/download_kth.py --raw-dir /path/to/extracted/avi/folder")
            sys.exit(1)
        # Extract first zip that looks like main archive (or all)
        import zipfile
        for zpath in downloaded:
            if zpath.suffix.lower() == ".zip":
                print("Extracting", zpath, "...")
                with zipfile.ZipFile(zpath) as z:
                    z.extractall(zip_dir)
                break
        # Find .avi and organise
        raw = zip_dir
        raw_dir = zip_dir
        for d in raw.rglob("*.avi"):
            raw_dir = d.parent
            break
        organize_by_class_and_split(raw_dir, out_dir, max_workers=args.organize_workers)
        print("Done. Set config data_dir to", out_dir)
        return

    print("Usage:")
    print("  1) If you have KTH .avi files in a folder:")
    print("     python scripts/download_kth.py --raw-dir /path/to/avi/folder")
    print("  2) Try automatic download (parallel):")
    print("     python scripts/download_kth.py --download [--workers 4] [--organize-workers 8]")
    print("  3) Set config data_dir to", out_dir)
    print("     Then place train/<class>/*.avi and test/<class>/*.avi under", out_dir)


if __name__ == "__main__":
    main()
