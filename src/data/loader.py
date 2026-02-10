"""
Data loaders for KTH-like video datasets.
Supports filesystem (data_dir/train/<class>/*.avi) or a DB (e.g. MongoDB) storing
metadata: path (or URL), label, split.
"""

from pathlib import Path
from typing import List, Optional, Tuple

# KTH-like class set when not provided by DB
DEFAULT_CLASSES = [
    "boxing", "handclapping", "handwaving", "jogging", "running", "walking"
]


def get_split_filesystem(data_dir: Path, split: str, classes: Optional[List[str]] = None) -> Tuple[List[Path], List[str]]:
    """
    Load (paths, labels) from data_dir/split/<class>/*.avi.
    """
    classes = classes or DEFAULT_CLASSES
    split_dir = data_dir / split
    if not split_dir.exists():
        return [], []
    paths = []
    labels = []
    for cls in classes:
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.glob("*.avi")) + sorted(cls_dir.glob("*.mp4")) + sorted(cls_dir.glob("*.mov")):
            paths.append(Path(p))
            labels.append(cls)
    return paths, labels


def get_split_mongodb(
    uri: str,
    database: str,
    collection: str,
    split: str,
    path_key: str = "path",
    label_key: str = "label",
    split_key: str = "split",
    base_dir: Optional[str] = None,
) -> Tuple[List[Path], List[str]]:
    """
    Load (paths, labels) from MongoDB. Each document should have:
    - path_key: path to video file (local or URL string)
    - label_key: class label (e.g. "boxing")
    - split_key: "train" | "test" | "val"

    If base_dir is set, path is interpreted as relative to base_dir (unless path is absolute or a URL).
    Returns paths as Path objects (use str(p) for cv2.VideoCapture; URLs are supported by OpenCV for some backends).
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        raise ImportError("pymongo is required for MongoDB source. Install with: pip install pymongo")

    client = MongoClient(uri)
    coll = client[database][collection]
    cursor = coll.find({split_key: split}, {path_key: 1, label_key: 1})
    paths = []
    labels = []
    for doc in cursor:
        p = doc.get(path_key)
        if not p:
            continue
        p = str(p).strip()
        if base_dir and not p.startswith(("/", "http://", "https://")):
            p = str(Path(base_dir) / p)
        paths.append(Path(p))
        labels.append(str(doc.get(label_key, "")))
    return paths, labels


def get_data_loader(config_data: dict):
    """
    Return a loader object with get_split(split) -> (paths, labels).
    config_data is the 'data' section of config.yaml.
    """
    source = (config_data.get("source") or "filesystem").lower()
    if source == "filesystem":
        return FilesystemLoader(config_data)
    if source == "mongodb":
        return MongoLoader(config_data)
    raise ValueError("data.source must be 'filesystem' or 'mongodb', got: %s" % source)


def list_splits(config_data: dict) -> List[str]:
    """Return available split names for the configured source."""
    loader = get_data_loader(config_data)
    return getattr(loader, "splits", ["train", "test", "val"])


class FilesystemLoader:
    """Uses data_dir with train/<class>/, test/<class>/ layout."""

    def __init__(self, config_data: dict):
        self.data_dir = Path(config_data["data_dir"])
        self.classes = config_data.get("classes") or DEFAULT_CLASSES

    def get_split(self, split: str) -> Tuple[List[Path], List[str]]:
        return get_split_filesystem(self.data_dir, split, self.classes)


class MongoLoader:
    """Uses MongoDB collection for path/label/split metadata."""

    def __init__(self, config_data: dict):
        db = config_data.get("mongodb") or {}
        self.uri = db.get("uri") or "mongodb://localhost:27017"
        self.database = db.get("database") or "kth"
        self.collection = db.get("collection") or "videos"
        self.path_key = db.get("path_key") or "path"
        self.label_key = db.get("label_key") or "label"
        self.split_key = db.get("split_key") or "split"
        self.base_dir = db.get("base_dir")  # optional: resolve relative paths

    def get_split(self, split: str) -> Tuple[List[Path], List[str]]:
        return get_split_mongodb(
            self.uri,
            self.database,
            self.collection,
            split,
            path_key=self.path_key,
            label_key=self.label_key,
            split_key=self.split_key,
            base_dir=self.base_dir,
        )
