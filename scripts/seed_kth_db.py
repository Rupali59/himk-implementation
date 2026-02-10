"""
Seed a KTH-like MongoDB collection from filesystem layout (data_dir/train/<class>/*.avi etc.).
Use this to connect the pipeline to a DB: run once to populate the DB, then set data.source: mongodb in config.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# KTH-like classes
CLASSES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]


def collect_paths(data_dir: Path) -> list:
    """Collect (path, label, split) from data_dir/split/<class>/*.avi."""
    out = []
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.glob("*.avi")) + sorted(cls_dir.glob("*.mp4")) + sorted(cls_dir.glob("*.mov")):
                out.append({"path": str(p.absolute()), "label": cls, "split": split})
    return out


def main():
    ap = argparse.ArgumentParser(description="Seed MongoDB with KTH-like video metadata from filesystem")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "kth", help="Root with train/<class>/, test/<class>/")
    ap.add_argument("--uri", default="mongodb://localhost:27017", help="MongoDB URI")
    ap.add_argument("--database", default="kth", help="Database name")
    ap.add_argument("--collection", default="videos", help="Collection name")
    ap.add_argument("--drop", action="store_true", help="Drop collection before inserting")
    args = ap.parse_args()

    try:
        from pymongo import MongoClient
    except ImportError:
        print("Install pymongo: pip install pymongo")
        sys.exit(1)

    records = collect_paths(args.data_dir)
    if not records:
        print("No videos found under", args.data_dir)
        print("Expected: data_dir/train/<class>/*.avi and data_dir/test/<class>/*.avi")
        sys.exit(1)

    client = MongoClient(args.uri)
    coll = client[args.database][args.collection]
    if args.drop:
        coll.drop()
    coll.insert_many(records)
    print("Inserted", len(records), "documents into", args.database, args.collection)
    for split in ("train", "val", "test"):
        n = sum(1 for r in records if r["split"] == split)
        if n:
            print(" ", split, n)


if __name__ == "__main__":
    main()
