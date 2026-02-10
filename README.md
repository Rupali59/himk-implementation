# Human Activity Recognition using HIMK

Implementation of **Human Activity Recognition using HMM-based Intermediate Matching Kernel** (thesis: videos as sequences of feature vectors, HoG + CDHMM + HIMK + SVM). Includes training, evaluation, optional MongoDB data source, and a web frontend for results and thesis PDF preview.

**Dataset:** KTH-style (6 actions: boxing, handclapping, handwaving, jogging, running, walking).

---

## Quick start (Docker)

**Build from the repo root (Thesis/)** so the image includes the thesis PDF and the full pipeline:

```bash
cd Thesis   # or your repo root containing implementation/ and latex/
docker compose -f implementation/docker-compose.yml up --build -d
```

This starts **MongoDB** (revisions DB) and the **app** (runs demo + train + evaluate, then serves the frontend). Open **http://localhost:5050**.  
To only serve (skip pipeline): override command: `docker compose -f implementation/docker-compose.yml run --rm app python run_all.py --serve`.

**Without Docker** (local):

```bash
cd implementation
python scripts/make_demo_videos.py
python pipelines/train.py --config config.yaml
python pipelines/evaluate.py --config config.yaml --export data/results.json
python frontend/app.py
```

Open **http://localhost:5050**. To change the port: `APP_PORT=8080`.

### Single script (run everything)

From the implementation directory:

```bash
# Demo + train + evaluate in one go (skip demo if you already have data with --no-demo)
python run_all.py --all

# Build thesis PDF then run pipeline (thesis is created as part of the subsystem)
python run_all.py --thesis --all

# Then optionally serve the frontend
python run_all.py --all --serve

# Or run only specific steps
python run_all.py --train --eval
python run_all.py --serve
```

**Revisions (MongoDB):** Every pipeline run and thesis build can be logged to MongoDB. Set `MONGODB_URI` and `MONGODB_DATABASE` (e.g. in Docker they point to the `mongo` service), or set `revisions.uri` / `revisions.database` in `config.yaml`. See [Revisions and data models](#revisions-and-data-models) below.

See `python run_all.py --help` for options. Known limitations: [PROBLEMS.md](PROBLEMS.md).

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
   Or try parallel download (multiple URLs and parallel organise):
   ```bash
   python scripts/download_kth.py --download [--workers 4] [--organize-workers 8]
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
| `python run_all.py --all` | Demo + train + evaluate (single script) |
| `python run_all.py --all --serve` | Pipeline + start frontend |
| `python run_all.py --thesis --all` | Build thesis then demo + train + eval |
| `docker compose -f implementation/docker-compose.yml up -d` | Run app + MongoDB from repo root |

---

## Docker (standard values)

**Build from repo root (Thesis/):** `docker compose -f implementation/docker-compose.yml up --build -d`

- **Services:** `mongo` (MongoDB 7, port 27017), `app` (pipeline + frontend, port 5050)
- **App:** Runs `run_all.py --all --serve` (demo + train + evaluate, then serve). Revisions are logged to MongoDB when `MONGODB_URI` / `MONGODB_DATABASE` are set (default in compose).
- **Thesis:** Built in the Docker image (multi-stage); PDF is at `/app/thesis.pdf` and served by the frontend.
- **Volumes:** `implementation/data` → `/app/data` (persists models, cache, results); `mongo_data` for MongoDB.

---

## Revisions and data models

Every pipeline run and thesis build can be written to MongoDB so each revision is recorded.

**Enable:** Set `MONGODB_URI` and `MONGODB_DATABASE` (e.g. `mongodb://mongo:27017` and `himk` in Docker), or in `config.yaml` under `revisions`: `uri`, `database`, and optionally `enabled: true`.

**Collections:** One collection (default `revisions`) with documents of two types:

- **`pipeline_run`:** `type`, `created_at`, `config` (snapshot), `git_commit`, `accuracy_pct`, `split`, `class_names`, `confusion_matrix_pct`, `confusion_matrix_counts`, `results_path`, `model_dir`, `status`
- **`thesis_build`:** `type`, `created_at`, `thesis_path`, `source_dir`, `success`, `log_excerpt`, `git_commit`

Use `--no-revisions` with `run_all.py` to skip logging.

**Stop containers:** `docker compose -f implementation/docker-compose.yml down`

---

## Project layout

```
.
├── README.md
├── PROBLEMS.md            # Known limitations and issues
├── run_all.py             # Single script: demo + train + eval + serve
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
