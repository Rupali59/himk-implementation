# Human Activity Recognition using HIMK

Implementation of **Human Activity Recognition using HMM-based Intermediate Matching Kernel** (thesis: videos as sequences of feature vectors, HoG + CDHMM + HIMK + SVM). Includes training, evaluation, optional MongoDB data source, and a web frontend for results and thesis PDF preview.

**Dataset:** KTH-style (6 actions: boxing, handclapping, handwaving, jogging, running, walking).

---

## Quick start (Docker)

```bash
git clone <your-repo-url>
cd himk-implementation   # or whatever you name the repo

# Optional: add demo data so the app has something to show
python scripts/make_demo_videos.py

# Train and export results (or skip and run container; results page will ask you to run evaluate)
python pipelines/train.py --config config.yaml
python pipelines/evaluate.py --config config.yaml --export data/results.json

# Run the frontend in Docker (standard port 5050)
docker compose up --build -d
```

Open **http://localhost:5050** for the results UI. To change the port: `APP_PORT=8080 docker compose up -d`.

---

## Pipeline

1. **Feature extraction** — HoG per frame (160×120); optional BoVW with histogram-matching K-medoid (`src/features/`).
2. **Kernel** — HIMK (complete-sequence): one CDHMM on all training data; virtual vectors from \(R_{iq}\); \(K_{HIMK}(X_m,X_n) = \sum_{i,q} k(x^*_{miq}, x^*_{niq})\) (`src/kernel/`).
3. **Classifier** — SVM with precomputed HIMK kernel (`src/classify/`).

---

## Requirements

- **Python** 3.9+
- **Install:** `pip install -r requirements.txt`

---

## Configuration

Edit **`config.yaml`**:

| Section   | Key              | Description |
|----------|------------------|-------------|
| `data`   | `source`         | `filesystem` or `mongodb` |
| `data`   | `data_dir`       | Path to KTH root: `data_dir/train/<class>/*.avi`, `data_dir/test/<class>/*.avi` |
| `data`   | `cache_dir`      | Where to cache HoG features |
| `data`   | `model_dir`      | Where to save CDHMM and SVM |
| `himk`   | `n_states`       | HMM states (e.g. 6 for demo, 15+ for full KTH) |
| `himk`   | `n_mix`          | GMM components per state (e.g. 2 or 3) |
| `himk`   | `base_kernel_gamma` | RBF kernel parameter |
| `svm`    | `C`              | SVM regularization |

---

## Dataset (KTH)

1. **Demo data (no download):**  
   `python scripts/make_demo_videos.py` — creates minimal AVI files under `data/kth/`.

2. **Real KTH:**  
   Download from [KTH Actions](https://www.csc.kth.se/cvap/actions/), then:
   ```bash
   python scripts/download_kth.py --raw-dir /path/to/folder/with/avi/files
   ```
   This creates `data/kth/train/<class>/` and `data/kth/test/<class>/`.

3. **MongoDB (KTH-like DB):**  
   Set `data.source: mongodb` and `data.mongodb.*` in `config.yaml`. Documents: `{ "path", "label", "split" }`.  
   Seed from filesystem: `python scripts/seed_kth_db.py --data-dir data/kth --uri mongodb://localhost:27017 --database kth --collection videos`.

---

## Commands

| Command | Description |
|---------|-------------|
| `python pipelines/train.py --config config.yaml` | Train CDHMM and SVM |
| `python pipelines/evaluate.py --config config.yaml` | Print accuracy and confusion matrix |
| `python pipelines/evaluate.py --config config.yaml --export data/results.json` | Export results for the frontend |
| `python frontend/app.py` | Run frontend locally (port 5050) |
| `docker compose up --build -d` | Run frontend in Docker |

---

## Docker (standard values)

**`docker-compose.yml`** uses:

- **Service name:** `himk-app`
- **Image:** `himk-implementation:latest`
- **Port:** `5050` (override with `APP_PORT`, e.g. `APP_PORT=8080 docker compose up -d`)
- **Volumes:** `./data` → `/app/data` (persists models, cache, results)
- **Thesis PDF (optional):** Uncomment the thesis volume and set `THESIS_PDF_PATH=/app/thesis.pdf` in `environment` to serve the PDF from the frontend.

**Run from this directory:**

```bash
docker compose up --build -d
```

**Stop:** `docker compose down`

---

## Project layout

```
.
├── README.md
├── LICENSE
├── config.yaml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/                 # data_dir, cache, models (gitignored except .gitignore)
├── src/
│   ├── features/         # hog.py, bovw.py
│   ├── kernel/           # hmm_utils.py, himk.py
│   ├── data/             # loader.py (filesystem + MongoDB)
│   └── classify/         # svm_himk.py
├── pipelines/
│   ├── train.py
│   └── evaluate.py
├── frontend/
│   ├── app.py
│   └── index.html
└── scripts/
    ├── download_kth.py
    ├── make_demo_videos.py
    └── seed_kth_db.py
```

---

## Reference

- Thesis: *Human Activity Recognition using HMM based Intermediate Matching Kernel by representing videos as sequence of sets of feature vectors.*
- HIMK: Dileep & Chandra, "HMM Based Intermediate Matching Kernel for Classification of Sequential Patterns of Speech Using Support Vector Machines," IEEE TASLP 2013.

---

## Publishing this repo on GitHub

1. Create a new repository on GitHub (e.g. `himk-implementation` or `human-activity-recognition-himk`).
2. From the implementation directory (this folder):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: HIMK thesis implementation"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```
3. Ensure `data/` contents are gitignored (only `data/.gitignore` is tracked) so videos and models are not pushed. Use [Git LFS](https://git-lfs.github.com/) if you need to share sample data.

## License

MIT — see [LICENSE](LICENSE).
