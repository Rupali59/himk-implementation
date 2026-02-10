"""
MongoDB revision logging: record every pipeline run and thesis build so revisions reflect in the DB.

Data models:
- pipeline_run: { type, created_at, config_snapshot, git_commit?, accuracy_pct, split, class_names,
                  confusion_matrix_pct, confusion_matrix_counts, results_path, model_dir, status }
- thesis_build:  { type, created_at, thesis_path, source_dir, success, log_excerpt? }
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Default collection name for all revision types
DEFAULT_REVISIONS_COLLECTION = "revisions"


def _get_git_commit(repo_path: Path) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def _get_client(uri: str):
    try:
        from pymongo import MongoClient
        return MongoClient(uri)
    except ImportError:
        raise ImportError("pymongo is required for revision logging. pip install pymongo")


def log_pipeline_run(
    uri: str,
    database: str,
    collection: str = DEFAULT_REVISIONS_COLLECTION,
    *,
    config_snapshot: Optional[dict] = None,
    results_path: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    repo_path: Optional[Path] = None,
    status: str = "success",
) -> Optional[str]:
    """
    Append a pipeline_run revision to MongoDB. If results_path exists, read accuracy and confusion matrix from it.
    Returns inserted document _id as string, or None if logging disabled/failed.
    """
    if not uri or not database:
        return None
    doc: dict[str, Any] = {
        "type": "pipeline_run",
        "created_at": datetime.now(timezone.utc),
        "status": status,
        "config": config_snapshot or {},
    }
    if repo_path:
        doc["git_commit"] = _get_git_commit(repo_path)
    if model_dir:
        doc["model_dir"] = str(model_dir)
    if results_path and results_path.exists():
        try:
            with open(results_path) as f:
                data = json.load(f)
            doc["accuracy_pct"] = data.get("accuracy_pct")
            doc["split"] = data.get("split", "test")
            doc["class_names"] = data.get("class_names")
            doc["confusion_matrix_pct"] = data.get("confusion_matrix_pct")
            doc["confusion_matrix_counts"] = data.get("confusion_matrix_counts")
            doc["results_path"] = str(results_path)
        except Exception:
            doc["results_path"] = str(results_path)
    try:
        client = _get_client(uri)
        coll = client[database][collection]
        ins = coll.insert_one(doc)
        return str(ins.inserted_id)
    except Exception as e:
        print(f"Revision log (MongoDB) failed: {e}", flush=True)
        return None


def log_thesis_build(
    uri: str,
    database: str,
    collection: str = DEFAULT_REVISIONS_COLLECTION,
    *,
    thesis_path: Path,
    source_dir: Optional[Path] = None,
    success: bool = True,
    log_excerpt: Optional[str] = None,
    repo_path: Optional[Path] = None,
) -> Optional[str]:
    """Append a thesis_build revision. Returns inserted _id or None."""
    if not uri or not database:
        return None
    doc: dict[str, Any] = {
        "type": "thesis_build",
        "created_at": datetime.now(timezone.utc),
        "thesis_path": str(thesis_path),
        "success": success,
    }
    if source_dir:
        doc["source_dir"] = str(source_dir)
    if log_excerpt:
        doc["log_excerpt"] = log_excerpt[:2000]
    if repo_path:
        doc["git_commit"] = _get_git_commit(repo_path)
    try:
        client = _get_client(uri)
        coll = client[database][collection]
        ins = coll.insert_one(doc)
        return str(ins.inserted_id)
    except Exception as e:
        print(f"Revision log (thesis, MongoDB) failed: {e}", flush=True)
        return None
