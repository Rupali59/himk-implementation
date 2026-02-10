"""
Build the thesis PDF (LaTeX). Used by run_all.py and optionally in Docker.
Expects latex source at LATEX_DIR (env) or ../latex relative to implementation root.
Output: <latex_dir>/build/Thesis.pdf
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Default: sibling of implementation/
DEFAULT_LATEX_DIR = ROOT.parent / "latex"


def main() -> int:
    latex_dir = Path(os.environ.get("LATEX_DIR", DEFAULT_LATEX_DIR))
    if not latex_dir.exists():
        print(f"LaTeX directory not found: {latex_dir}", file=sys.stderr)
        return 1
    makefile = latex_dir / "Makefile"
    if not makefile.exists():
        print(f"No Makefile in {latex_dir}", file=sys.stderr)
        return 1
    print(f"Building thesis in {latex_dir} ...")
    r = subprocess.run(["make", "pdf"], cwd=latex_dir)
    if r.returncode != 0:
        return r.returncode
    out = latex_dir / "build" / "Thesis.pdf"
    if out.exists():
        print(f"Thesis built: {out}")
        return 0
    print("Build reported success but Thesis.pdf not found.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
