"""
Download and prepare KTH Human Action dataset.
KTH has 6 classes: boxing, handclapping, handwaving, jogging, running, walking.
25 subjects; videos are typically 160x120, ~4 seconds.
Standard split: persons 1-16 train, 17-21 validation, 22-25 test (or similar).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Default: expect data in implementation/data/kth with structure:
#   kth/
#     person01_boxing_d1_uncomp.avi  ... (or in class subdirs after we organize)
# We create train/ and test/ with class subdirs and (optionally) resize to 160x120.
KTH_CLASSES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

# Common split: 16 train, 4 val, 5 test (by person number)
TRAIN_PERSONS = list(range(1, 17))   # 1-16
VAL_PERSONS = list(range(17, 22))    # 17-21
TEST_PERSONS = list(range(22, 26))   # 22-25

# Alternative: single zip URL (may require manual download if link is dead)
KTH_ZIP_URL = "https://www.csc.kth.se/cvap/actions/00sequences.zip"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def organize_by_class_and_split(raw_dir: Path, out_dir: Path) -> None:
    """
    raw_dir contains files like person01_boxing_d1_uncomp.avi.
    Create out_dir/train/<class>/*.avi and out_dir/test/<class>/*.avi (and val if desired).
    """
    import shutil
    import re
    files = list(raw_dir.rglob("*.avi"))
    if not files:
        files = list(raw_dir.glob("*.avi")) + list(raw_dir.glob("*.AVI"))
    pattern = re.compile(r"person(\d+)_(\w+)_", re.I)
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
        if not dest.exists() or dest.stat().st_size != f.stat().st_size:
            shutil.copy2(f, dest)
    print("Organized into", out_dir)


def download_url(url: str, dest: Path) -> bool:
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print("Download failed:", e)
        return False


def main():
    ap = argparse.ArgumentParser(description="Prepare KTH dataset for implementation")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output root (default: implementation/data/kth)")
    ap.add_argument("--raw-dir", type=Path, default=None, help="If you already have raw KTH .avi files in one dir, we organize them")
    ap.add_argument("--download", action="store_true", help="Try to download 00sequences.zip (may require manual download)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = args.out_dir or root / "data" / "kth"

    if args.raw_dir:
        raw = Path(args.raw_dir)
        if not raw.exists():
            print("Raw dir does not exist:", raw)
            sys.exit(1)
        organize_by_class_and_split(raw, out_dir)
        print("Done. Set config data_dir to", out_dir)
        return

    if args.download:
        ensure_dir(out_dir.parent)
        zip_path = out_dir.parent / "00sequences.zip"
        print("Attempting download from", KTH_ZIP_URL)
        if download_url(KTH_ZIP_URL, zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(out_dir.parent)
            # KTH zip often has subdirs; find .avi and organize
            raw = out_dir.parent
            for d in raw.rglob("*.avi"):
                raw_dir = d.parent
                break
            else:
                raw_dir = raw
            organize_by_class_and_split(raw_dir, out_dir)
        else:
            print("Manual step: download KTH from https://www.csc.kth.se/cvap/actions/")
            print("Then run: python scripts/download_kth.py --raw-dir /path/to/extracted/avi/folder")
        return

    print("Usage:")
    print("  1) If you have KTH .avi files in a folder:")
    print("     python scripts/download_kth.py --raw-dir /path/to/avi/folder")
    print("  2) Try automatic download:")
    print("     python scripts/download_kth.py --download")
    print("  3) Set config data_dir to", out_dir)
    print("     Then place train/<class>/*.avi and test/<class>/*.avi under", out_dir)


if __name__ == "__main__":
    main()
