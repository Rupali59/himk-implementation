"""
Simple web app to publish evaluation results and thesis PDF preview.
Run: python app.py  (or from implementation: python frontend/app.py)
Open http://localhost:5000
"""

import json
import os
from pathlib import Path
import sys

from flask import Flask, send_file, send_from_directory

# Ensure we can import the shared config helpers from src/.
IMPLEMENTATION_ROOT = Path(__file__).resolve().parent.parent
if str(IMPLEMENTATION_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPLEMENTATION_ROOT))

from src.config import default_results_path, resolve_thesis_pdf_path  # noqa: E402


app = Flask(__name__, static_folder="static", static_url_path="", template_folder="")

# Shared paths (results JSON + thesis PDF) resolved via src.config.
RESULTS_PATH = default_results_path()
THESIS_PDF_PATH = resolve_thesis_pdf_path()


@app.route("/")
def index():
    # For now, serve the existing static HTML. When the React/Vite build is in
    # place, this can be updated to serve frontend/dist/index.html instead.
    return send_from_directory(Path(__file__).parent, "index.html")


@app.route("/api/results")
def api_results():
    if not RESULTS_PATH.exists():
        return {
            "error": "No results yet. Run: python pipelines/evaluate.py --export data/results.json"
        }, 404
    with open(RESULTS_PATH) as f:
        return json.load(f)


@app.route("/thesis.pdf")
def thesis_pdf():
    if not THESIS_PDF_PATH.exists():
        return f"Thesis PDF not found at {THESIS_PDF_PATH}", 404
    return send_file(THESIS_PDF_PATH, mimetype="application/pdf", as_attachment=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Open http://localhost:{port} for results and thesis preview.")
    app.run(host="0.0.0.0", port=port, debug=False)
