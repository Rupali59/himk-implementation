# Current Implementation — Known Problems and Limitations

This document lists issues, limitations, and possible improvements in the HIMK thesis implementation.

---

## 1. Data and pipeline

### Demo data is not representative
- **Problem:** `make_demo_videos.py` creates random-pixel AVI files. They do not represent real actions, so accuracy on this “demo” set is meaningless and can be unstable.
- **Impact:** Good for testing that the pipeline runs; not for reporting or comparing methods.
- **Fix:** Use real KTH (or similar) data for any reported results; keep demo only for smoke tests.

### Config paths are CWD-relative
- **Problem:** `config.yaml` uses relative paths (`data/kth`, `data/cache`, etc.). These are resolved against the **current working directory**, not the config file location. Running `python pipelines/train.py` from a different directory (e.g. project root) can point to wrong or missing paths.
- **Impact:** Failures or wrong data when not run from the implementation root.
- **Fix:** Resolve paths relative to the config file (or a fixed “project root”) in the code that loads config; or document “must run from implementation/” and enforce in scripts (e.g. `run_all.py` already chdirs to ROOT).

### Evaluation depends on train HoG cache
- **Problem:** `evaluate.py` needs the **train** HoG cache (`hog_train.joblib`) to compute the test–train HIMK block. If the user deletes the cache but keeps the model, evaluation fails.
- **Impact:** Cannot “just run evaluate” on another machine with only the saved model and test data.
- **Fix:** Either (a) document that train cache is required for evaluation, or (b) re-extract train HoG when missing (slower, but self-contained), or (c) store a subset of train virtual vectors with the model for fixed-size test-time kernel (more invasive).

---

## 2. Model and algorithm

### Single global CDHMM
- **Problem:** One CDHMM is trained on all classes. The thesis sometimes uses per-class HMMs; a single shared HMM may be less discriminative.
- **Impact:** Possible loss in accuracy vs per-class HMMs or other topologies.
- **Fix:** Add an option to train one CDHMM per class and define a combined kernel (e.g. sum or concatenation of per-class HIMK values) if desired.

### HIMK Gram matrix is O(n²) in memory and time
- **Problem:** Full train–train Gram matrix is stored and computed in one go. For large n this is memory-heavy and slow; no chunking or approximate methods.
- **Impact:** Does not scale to very large datasets.
- **Fix:** Chunked Gram computation, optional Nyström or other low-rank approximation, or online/streaming SVM if needed.

### No cross-validation or repeated runs
- **Problem:** Only a single train/test split; no k-fold CV or multiple seeds. Metrics can be noisy, especially with little data.
- **Impact:** Reported accuracy may be unstable or overfit to one split.
- **Fix:** Add a small runner that does k-fold or repeated train/test with different seeds and reports mean ± std.

---

## 3. Numerical and robustness

### HMM convergence on tiny/demo data
- **Problem:** With very few or degenerate sequences (e.g. demo), the GMM-HMM can converge poorly (e.g. singular covariances, bad local optima).
- **Impact:** Warnings, NaN, or poor accuracy even though kernel scaling and nan_to_num are applied later.
- **Fix:** Stronger regularization (e.g. larger `min_covar`), more EM iterations, or early checks for degenerate models; keep demo only for “does it run?” tests.

### Possible label mismatch in evaluation
- **Problem:** `evaluate.py` builds `y_true_idx` by indexing `class_list.index(y)` where `class_list = list(le.classes_)`. If the test set contains a label not seen in training, this throws.
- **Impact:** Crash on hold-out classes or misconfigured data.
- **Fix:** Filter test samples to training classes only, or fail with a clear error listing unseen labels.

---

## 4. Data loaders and I/O

### MongoDB paths and OpenCV
- **Problem:** MongoDB loader can store absolute paths or URLs. OpenCV’s `VideoCapture` support for URLs and backends varies by build and OS.
- **Impact:** Some URLs or remote paths may not open; errors are often generic.
- **Fix:** Document supported path types; optionally validate or try opening one file at startup; consider a small adapter that downloads to a temp file for URLs.

### No validation split usage
- **Problem:** Data loader supports a "val" split, but the training pipeline does not use it (no validation-based early stopping or hyperparameter selection).
- **Impact:** Val split is unused in the current pipeline.
- **Fix:** Either use val for model selection / early stopping or document that only train/test are used.

---

## 5. Frontend and deployment

### Thesis PDF path is brittle
- **Problem:** Frontend looks for the thesis PDF in several fallback locations (`THESIS_PDF_PATH`, `../latex/build/Thesis.pdf`, etc.). If the repo is cloned without the thesis repo, or paths differ, the PDF tab fails.
- **Impact:** 404 or “not found” for thesis preview in some setups.
- **Fix:** Document required layout or env var; optionally make the thesis section hidden or show a message when PDF is not configured.

### No HTTPS or auth
- **Problem:** Flask app runs over HTTP with no authentication. Fine for local use; unsuitable as-is for exposed deployment.
- **Impact:** Not for production without a reverse proxy (e.g. with HTTPS and auth).
- **Fix:** Document “local/dev only”; for production, run behind a proper server (e.g. gunicorn + nginx with TLS and auth).

---

## 6. Code and maintainability

### Duplicated config loading
- **Problem:** `train.py` and `evaluate.py` each define `load_config()` and resolve paths from config. Small drift between the two is possible.
- **Impact:** Minor maintenance cost and risk of inconsistency.
- **Fix:** Move config loading (and path resolution) to a shared module (e.g. `src/config.py` or `pipelines/common.py`) and reuse.

### Limited logging
- **Problem:** Pipeline uses `print()` only; no log levels or structured logging.
- **Impact:** Hard to turn off verbose output or to redirect logs in scripts.
- **Fix:** Use the `logging` module with levels; keep a single entry point (e.g. `run_all.py`) that can set level or redirect.

### BoVW not wired into main pipeline
- **Problem:** `src/features/bovw.py` (histogram matching, K-medoid) is implemented but not used in `train.py` / `evaluate.py`. The main pipeline uses raw HoG sequences only.
- **Impact:** BoVW code is dead for the current pipeline; thesis may describe both.
- **Fix:** Either add an option to use BoVW (e.g. per-video histograms as features) or document that the current pipeline is HoG-only.

---

## Summary table

| Area        | Issue                          | Severity  | Easy fix? |
|------------|---------------------------------|-----------|-----------|
| Data       | Demo data not representative   | Medium    | Use real data for results |
| Data       | Config paths CWD-relative       | Medium    | Yes (resolve from config path) |
| Eval       | Needs train HoG cache           | Medium    | Document or re-extract |
| Model      | Single CDHMM only               | Low       | Option for per-class |
| Scale      | Full Gram O(n²)                 | High for large n | Chunking / approx |
| Robustness | No CV / repeated runs           | Medium    | Add small runner |
| Eval       | Test label not in train         | Low       | Filter or clear error |
| Frontend   | Thesis PDF path brittle         | Low       | Document / hide if missing |
| Code       | Duplicated config, no logging  | Low       | Shared module, logging |

Addressing the “config paths” and “evaluation depends on train cache” items gives the largest benefit for typical use. The rest can be tackled as needed for scale, reporting, or production.
