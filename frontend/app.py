"""
Simple web app to publish evaluation results and thesis PDF preview.
Run: python app.py  (or from implementation: python frontend/app.py)
Open http://localhost:5000
"""

import json
import os
from pathlib import Path

from flask import Flask, send_file, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="", template_folder="")

# Paths relative to implementation/
IMPLEMENTATION_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = IMPLEMENTATION_ROOT / "data" / "results.json"
# Thesis PDF: env (e.g. in Docker) or sibling of implementation/
THESIS_PDF_PATH = Path(os.environ.get("THESIS_PDF_PATH", ""))
if not THESIS_PDF_PATH or not THESIS_PDF_PATH.exists():
    THESIS_PDF_PATH = IMPLEMENTATION_ROOT.parent / "latex" / "build" / "Thesis.pdf"
if not THESIS_PDF_PATH.exists():
    THESIS_PDF_PATH = IMPLEMENTATION_ROOT.parent / "Thesis.pdf"
if not THESIS_PDF_PATH.exists():
    THESIS_PDF_PATH = IMPLEMENTATION_ROOT.parent / "report.pdf"


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.route("/api/results")
def api_results():
    if not RESULTS_PATH.exists():
        return {"error": "No results yet. Run: python pipelines/evaluate.py --export data/results.json"}, 404
    with open(RESULTS_PATH) as f:
        return json.load(f)


@app.route("/thesis.pdf")
def thesis_pdf():
    if not THESIS_PDF_PATH.exists():
        return f"Thesis PDF not found at {THESIS_PDF_PATH}", 404
    return send_file(THESIS_PDF_PATH, mimetype="application/pdf", as_attachment=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print("Open http://localhost:{} for results and thesis preview.".format(port))
    app.run(host="0.0.0.0", port=port, debug=False)
