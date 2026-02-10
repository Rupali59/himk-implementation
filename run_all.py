#!/usr/bin/env python3
"""
Single script to run the full HIMK pipeline: thesis build (optional), demo data (optional),
train, evaluate, log revisions to MongoDB, and optionally serve the frontend.

Usage (from repo root, i.e. implementation/):
  python run_all.py --all                    # demo + train + evaluate
  python run_all.py --all --thesis           # build thesis then demo + train + eval
  python run_all.py --all --serve            # pipeline then start web UI
  python run_all.py --train --eval          # assume data already exists
  python run_all.py --serve                 # only start frontend

Paths in config.yaml are relative to the directory containing run_all.py (implementation root).
Revisions: set config revisions.uri/database or env MONGODB_URI/MONGODB_DATABASE to log every run to MongoDB.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Implementation root = directory containing this script
ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], step_name: str, cwd: Path, env: dict | None = None) -> bool:
    """Run a command; return True on success."""
    print(f"\n--- {step_name} ---")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, env=env or os.environ)
    if result.returncode != 0:
        print(f"Failed: {step_name} (exit {result.returncode})", file=sys.stderr)
        return False
    return True


def load_config(config_path: Path) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_revision_config(config: dict) -> tuple[str | None, str | None, str | None]:
    """Return (uri, database, collection) for revision logging. Env overrides config."""
    rev = config.get("revisions") or {}
    uri = os.environ.get("MONGODB_URI") or rev.get("uri")
    database = os.environ.get("MONGODB_DATABASE") or rev.get("database")
    collection = rev.get("collection") or "revisions"
    if rev.get("enabled") is True or (uri and database):
        return (uri, database, collection)
    return (None, None, collection)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run HIMK pipeline: thesis build, demo, train, evaluate, revisions, serve.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml", help="Config file path")
    ap.add_argument("--thesis", action="store_true", help="Build thesis PDF (make -C latex pdf) before pipeline")
    ap.add_argument("--demo", action="store_true", help="Create demo videos under data/kth")
    ap.add_argument("--train", action="store_true", help="Run training (CDHMM + HIMK Gram + SVM)")
    ap.add_argument("--eval", action="store_true", help="Run evaluation and export data/results.json")
    ap.add_argument("--serve", action="store_true", help="Start Flask frontend (port 5050) after pipeline")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Run demo (unless --no-demo) + train + eval in one go",
    )
    ap.add_argument(
        "--no-demo",
        action="store_true",
        help="With --all: skip creating demo videos (use existing data)",
    )
    ap.add_argument("--port", type=int, default=5050, help="Port for frontend (default 5050)")
    ap.add_argument("--no-revisions", action="store_true", help="Do not log to MongoDB even if configured")
    args = ap.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    config = load_config(config_path)
    rev_uri, rev_database, rev_collection = get_revision_config(config)
    if args.no_revisions:
        rev_uri, rev_database = None, None
    config_snapshot = {
        "data": config.get("data", {}),
        "himk": config.get("himk", {}),
        "svm": config.get("svm", {}),
    }

    do_thesis = args.thesis
    do_demo = args.demo or (args.all and not args.no_demo)
    do_train = args.train or args.all
    do_eval = args.eval or args.all
    do_serve = args.serve

    if not (do_thesis or do_demo or do_train or do_eval or do_serve):
        ap.print_help()
        print("\nNo steps selected. Use --all or --thesis/--demo/--train/--eval/--serve.")
        return 0

    # Ensure we run from implementation root so relative paths in config work
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    thesis_path: Path | None = None
    if do_thesis:
        latex_dir = Path(os.environ.get("LATEX_DIR", ROOT.parent / "latex"))
        if latex_dir.exists() and (latex_dir / "Makefile").exists():
            if not run(
                [sys.executable, str(ROOT / "scripts" / "build_thesis.py")],
                "Build thesis PDF",
                ROOT,
                env={**os.environ, "LATEX_DIR": str(latex_dir)},
            ):
                return 1
            thesis_path = latex_dir / "build" / "Thesis.pdf"
            if thesis_path.exists() and rev_uri and rev_database:
                from src.revisions import log_thesis_build
                log_thesis_build(
                    rev_uri, rev_database, rev_collection,
                    thesis_path=thesis_path, source_dir=latex_dir, success=True, repo_path=ROOT,
                )
        else:
            print(f"Thesis build skipped (LATEX_DIR not found or no Makefile): {latex_dir}")

    if do_demo:
        if not run(
            [sys.executable, str(ROOT / "scripts" / "make_demo_videos.py")],
            "Create demo videos",
            ROOT,
        ):
            return 1

    if do_train:
        cmd = [sys.executable, str(ROOT / "pipelines" / "train.py"), "--config", str(config_path)]
        if not run(cmd, "Train (CDHMM + HIMK + SVM)", ROOT):
            return 1

    if do_eval:
        results_path = ROOT / "data" / "results.json"
        cmd = [
            sys.executable,
            str(ROOT / "pipelines" / "evaluate.py"),
            "--config",
            str(config_path),
            "--export",
            str(results_path),
        ]
        if not run(cmd, "Evaluate and export results", ROOT):
            return 1
        if rev_uri and rev_database:
            model_dir = Path(config.get("data", {}).get("model_dir", "data/models"))
            model_dir = model_dir if model_dir.is_absolute() else ROOT / model_dir
            from src.revisions import log_pipeline_run
            rid = log_pipeline_run(
                rev_uri, rev_database, rev_collection,
                config_snapshot=config_snapshot,
                results_path=results_path,
                model_dir=model_dir,
                repo_path=ROOT,
                status="success",
            )
            if rid:
                print(f"Revision logged to MongoDB: {rid}")

    if do_serve:
        env = os.environ.copy()
        env["PORT"] = str(args.port)
        if thesis_path and thesis_path.exists():
            env["THESIS_PDF_PATH"] = str(thesis_path)
        elif (ROOT.parent / "latex" / "build" / "Thesis.pdf").exists():
            env["THESIS_PDF_PATH"] = str(ROOT.parent / "latex" / "build" / "Thesis.pdf")
        cmd = [sys.executable, str(ROOT / "frontend" / "app.py")]
        print(f"\n--- Serve frontend (http://localhost:{args.port}) ---")
        print("Press Ctrl+C to stop.")
        subprocess.run(cmd, cwd=ROOT, env=env)
        return 0

    print("\nDone.")
    if do_eval and (ROOT / "data" / "results.json").exists():
        print("View results: python frontend/app.py  or  docker compose up --build -d")
    return 0


if __name__ == "__main__":
    sys.exit(main())
