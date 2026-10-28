# OPENCODE.md — Claude ⇄ OpenCode channel

Shared task board + conversation log between **Claude** (plans, writes code, reviews results) and **OpenCode** (runs everything on Madhav's laptop and reports back).

**Madhav does nothing manually.** OpenCode does every step below, including setup, downloads and checks. Only if something is truly impossible without a human (e.g. a password prompt) does OpenCode ask Madhav — and says exactly what to click.

Related: `decisions.md` (why) · `experiments/EXPERIMENTS.md` (experiment registry) · `MetalAI_Project_Guide.md` (big picture) · `reports/m1/REPORT.md` (auto-generated results)

---

## 0. Rules for OpenCode (read first, every session)

1. Work through tasks **in order**, starting at the one marked `NEXT`. Stop at a **CHECKPOINT** or on any failure you can't fix with a tiny change.
2. Shell = **Windows PowerShell**, repo root `D:\python\metalAI`, venv active: `.\.venv\Scripts\Activate.ps1` (activate at the start of every new shell).
3. **Training is GPU-only** (section 0.1). Never use `--allow-cpu` unless Claude asks.
4. **Code organisation** (section 0.2) — don't add files outside the agreed layout; don't change code silently.
5. **Experiments** (section 0.3) — every training/eval run has an experiment id set via `$env:METALAI_EXP`.
6. **Secrets** — the Kaggle token lives only in `.secrets\kaggle_access_token` (git-ignored) and `%USERPROFILE%\.kaggle\access_token`. Never print it, paste it in reports, or commit it. Before every commit run `git status` and make sure `.secrets/` is not listed.
7. **Never edit numbers by hand.** Results only come from `results/m1/*.csv` → `python -m metalai.m1_surface.report`.
8. After every task: append a report to **section 3 (Log)** with the template, update the **status board** (section 1), update `experiments/EXPERIMENTS.md` if it was an experiment, then commit: `git add -A; git commit -m "T?: <short msg>"`.

### 0.1 GPU rules — RTX 4050 Laptop GPU (local training)

1. **Confirm the dGPU** (T1): `python -m metalai.common.env_check` → `cuda_available: True`, `gpu_name` contains `RTX 4050`, `gpu_matmul_ok: True`; `pip show torch` version must NOT contain `+cpu`. CUDA always runs on the NVIDIA GPU, so no Windows graphics setting is needed.
2. **Before each training run, check power** (report the value, don't block on it):
   ```powershell
   (Get-CimInstance Win32_Battery).BatteryStatus   # 2 = on AC power; anything else -> mention it in the report
   ```
   Close nothing of Madhav's; just don't start more than ONE training job at a time.
3. **Monitor the GPU during every training run** — start this in a second terminal, stop it when training ends, and summarise peak memory / max temp / typical utilisation in the report:
   ```powershell
   nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 10 | Tee-Object -FilePath reports\gpu_logs\$env:METALAI_EXP.csv
   ```
   (create `reports\gpu_logs` first). Expect utilisation > 50 % and a few GB of memory. If temperature stays > 87 °C, mention it.
4. **Expected footprint** (estimates; T4 measures real values):

   | Job | VRAM | Time |
   |---|---|---|
   | ResNet-50, bs 32, AMP | ~3–4 GB | ~5–10 min / fold |
   | EfficientNet-B0, bs 32, AMP | ~2–3 GB | ~3–6 min / fold |
   | YOLOv8n, 640 px, bs 16 | ~3 GB | ~20–45 min / fold |
   | LBP/GLCM + SVM | CPU | a few min total |

5. **Out-of-memory** → retry ONCE with CNN `--batch-size 16` / YOLO `--batch 8`; record it. Still failing → STOP and report.
6. **Interrupted run** → re-run the same command with `--skip-existing`.
7. Windows dataloader hang → CNN keep `--workers 0` (default); YOLO add `--workers 0`; report it.

### 0.2 Code organisation (keep it this way)

```
configs/            m1.yaml (hyper-parameters), app.yaml (which weights the app serves)
src/metalai/
  common/           utils (paths, seeding, experiments, results), metrics, env_check
  m1_surface/       one file per pipeline step: prepare_data, splits, eda, classical, preprocess,
                    cnn, gradcam, yolo_data, yolo_train, report, inference
  m2_steel/, m3_alloy/   (later)
app/                app.py (Gradio) + smoke_test.py (headless check)
experiments/        EXPERIMENTS.md (registry) + runs/<exp_id>/*.json (auto: command, args, config, commit)
data/raw|processed  not committed          data/splits/   committed
models/<module>/<kind>/<exp_id>/...  not committed       runs/  not committed
results/m1/         metric CSVs (committed)   reports/m1/  figures + REPORT.md (committed)
```
- All code lives in `src/metalai/...`; run it with `python -m metalai.<module>.<script>`. No loose scripts in the root, no notebooks for anything that produces reported numbers.
- Hyper-parameters go in `configs/*.yaml` or CLI flags — never hard-coded edits for one run.
- Tiny fixes allowed (import, path, library API rename) → list them under "Code changes" in the report. Anything else → ask Claude.

### 0.3 Experiment rules

- Claude assigns ids in the tasks (`E001`, `E002`, …). Throw-away tests use `SMOKE`.
- Before running: `$env:METALAI_EXP = "E00X"` in that shell.
- Scripts automatically save command + args + config + git commit to `experiments/runs/<id>/`, tag every CSV row with `exp_id`, and put weights/figures in `<id>`-named folders, so experiments never overwrite each other.
- After the run: fill the row in `experiments/EXPERIMENTS.md` (date, status `done`/`failed`, headline metric). Leave "Conclusion" for Claude.
- Never delete an experiment's results; if one is invalid, mark it `discarded` + reason in the registry.

### 0.4 Parallel execution + subagents (do this to go faster)

The laptop has **one GPU (RTX 4050, ~6 GB)** and a multi-core CPU. Parallelise across *resources*, not on the GPU:

| Lane | Runs on | What | Rule |
|---|---|---|---|
| **A — GPU lane** | GPU | T4 smoke → E002 ResNet-50 → E003 EfficientNet-B0 → E004 YOLOv8n | **Strictly one GPU job at a time.** Two trainings on a 6 GB laptop GPU = out-of-memory or both slower. |
| **B — CPU lane** | CPU | T5 classical SVM (E001), EDA, report drafts | Run **at the same time** as lane A. Use `--n-jobs 4` (about half the cores) so the GPU job's data loading isn't starved. |
| **C — I/O lane** | network/disk | T1 `pip install` and T2 Kaggle download | Download data while packages install (two terminals). |
| **D — monitor/reporter** | – | `nvidia-smi` logging, reading logs, checking PNGs exist, drafting the Log report + EXPERIMENTS.md rows | Good job for a **subagent**. |

**Subagents (OpenCode `task` / `@general` subagents):**
- Main agent = **orchestrator**: owns lane A, decides order, does all `git commit`s and edits to the status board.
- Spawn subagents for independent work, e.g.:
  - *CPU subagent*: "activate venv, `$env:METALAI_EXP='E001'`, run `python -m metalai.m1_surface.classical --n-jobs 4`, return the `==>` lines and PNG paths."
  - *Monitor subagent*: "run the nvidia-smi logger for exp E00X into `reports/gpu_logs/E00X.csv`; when training finishes, summarise peak memory / max temp / mean utilisation."
  - *Reporter subagent*: "after `python -m metalai.m1_surface.report`, check every PNG listed in T6 exists, draft the T6 Log entry and EXPERIMENTS.md rows; return the draft (don't commit)."
- **Every subagent shell sets its own `$env:METALAI_EXP`** (env vars are per-shell) and activates the venv.
- Subagents **don't commit and don't edit the status board** — they return text; the orchestrator writes it in and commits. This avoids git conflicts.
- Two processes must never write the same CSV at the same moment: lane A writes `cnn_results.csv` / `yolo_results.csv`, lane B writes `classical_results.csv` — separate files, safe. Run `report.py` only from the orchestrator, after jobs finish.

**Parallel schedule for this round:**
```
t=0   C: T1 pip installs            ||  C: T2 Kaggle download (needs only `pip install kaggle`)
      -> T1 env_check (GPU confirm)
      -> T3 prepare_data, splits, yolo_data   (sequential, ~1-2 min)
      -> A: T4 smoke (GPU)          ||  B: EDA + T5 classical E001 (CPU, --n-jobs 4)   ||  D: monitor
      -> A: E002 -> E003 -> E004    ||  B: (when T5 done) draft reports            ||  D: monitor each exp
      -> orchestrator: report.py, verify PNGs, fill EXPERIMENTS.md, Log, commit -> CHECKPOINT
```
Saves roughly the whole T5 time plus the download time. Don't parallelise anything else on the GPU.

### Report template (copy for each task)

```markdown
### [T?] <title> — DONE | FAILED | BLOCKED   (date, time taken, exp id if any)
**Commands run:** (exact, in order)
**Console output (key lines / last ~30 lines; full traceback on error):**
**Metrics:** (numbers printed by the scripts + relevant table from reports/m1/REPORT.md)
**Files produced:** (paths of csv/json/png/pt — Claude will open the PNGs)
**Hardware:** GPU name, peak VRAM, max temp, typical utilisation, sec/epoch, AC power yes/no, OOM retries
**Anything odd:**
**Code changes:** none | file — change — reason
**Questions for Claude:**
```

---

## 1. Status board

| ID | Task | Exp | Status |
|---|---|---|---|
| T1 | Environment setup + GPU check | – | **NEXT** |
| T2 | Kaggle auth + download NEU | – | TODO |
| T3 | Prepare data, splits, EDA, YOLO labels | – | TODO |
| T4 | GPU smoke tests (CNN, Grad-CAM, YOLO, app) + cleanup | SMOKE | TODO |
| T5 | Classical baseline (LBP/GLCM + SVM) | E001 | TODO |
| T6 | Fold-0: ResNet-50, EfficientNet-B0, YOLOv8n + Grad-CAM — **CHECKPOINT** | E002–E004 | TODO |
| T7+ | Remaining folds, full models, app — written by Claude after reviewing T6 | – | WAITING-FOR-CLAUDE |

---

## 2. Tasks

### T1 — Environment setup + GPU check
```powershell
cd D:\python\metalAI
if (-not (Test-Path .git)) { git init }
git config user.name  2>$null; if (-not $?) { git config user.name "MetalAI OpenCode" }
git config user.email 2>$null; if (-not $?) { git config user.email "opencode@local" }
nvidia-smi
py -3.11 -m venv .venv          # if 3.11 missing: py -3.10 -m venv .venv  (or install Python 3.11 with: winget install Python.Python.3.11)
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
python -m metalai.common.env_check
pip show torch | Select-String Version
New-Item -ItemType Directory -Force reports\gpu_logs | Out-Null
git add -A; git status          # make sure .secrets/ is NOT listed
git commit -m "T1: environment + scaffold"
```
If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force`, then retry.
If the cu124 wheel fails or `cuda_available` is False: try `--index-url https://download.pytorch.org/whl/cu121`; still failing → STOP and report.
**Report:** `nvidia-smi` output, `reports/env.json` contents, torch version line.

### T2 — Kaggle auth + download NEU
The token was provided by Madhav and saved by Claude to `.secrets\kaggle_access_token` (git-ignored). Install it for the Kaggle client — **don't print it**:
```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
Copy-Item .secrets\kaggle_access_token "$env:USERPROFILE\.kaggle\access_token" -Force
pip install -U kaggle
kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database -p data/raw/neu --unzip
Get-ChildItem data\raw\neu -Recurse -Directory | Select-Object -First 40 FullName
Get-ChildItem data\raw\neu -Recurse -File | Group-Object Extension | Select-Object Name, Count
```
If auth fails with the file: try `$env:KAGGLE_API_TOKEN = (Get-Content .secrets\kaggle_access_token -Raw).Trim()` in the same shell and re-run the download. Report only the error text, never the token.
Then fill the NEU section of `data/README.md` (download date, "downloaded by OpenCode", licence text from `kaggle datasets metadata kaustubhdikshit/neu-surface-defect-database -p data/raw/neu` if available).
**Report:** folder tree, file counts per extension, 3 example file names per class.

### T3 — Prepare data, splits, EDA, YOLO labels
```powershell
python -m metalai.m1_surface.prepare_data
python -m metalai.m1_surface.splits
python -m metalai.m1_surface.eda
python -m metalai.m1_surface.yolo_data
python -m metalai.m1_surface.report
```
Expected: 1,800 images, 300/class, all 200×200, every image ≥1 box, 0 label mismatches, 360 test images/fold. If different → report, continue.
**Report:** full console output of `prepare_data` + `splits`, `reports/m1/data_summary.json`, EDA PNG paths (`reports/m1/eda/`). Commit `data/splits/neu_folds.csv`.

### T4 — GPU smoke tests (exp `SMOKE`, thrown away afterwards)
```powershell
$env:METALAI_EXP = "SMOKE"
python -m metalai.m1_surface.cnn --model resnet50 --folds 0 --epochs 1
python -m metalai.m1_surface.gradcam --model resnet50 --fold 0
python -m metalai.m1_surface.yolo_train --folds 0 --epochs 1
python app\smoke_test.py                      # headless: runs the app's inspect() on 6 images
python -m metalai.m1_surface.report
```
Check `reports/m1/app_smoke/` has `*_det.png` and `*_cam.png` and the script printed `OK`.
Optional live check: `Start-Process python -ArgumentList "app\app.py"`; wait 20 s; `Invoke-WebRequest http://127.0.0.1:7860 -UseBasicParsing | Select-Object StatusCode` (expect 200); then stop it: `Get-Process python | Where-Object { $_.Path -like "*metalAI*" } | Stop-Process`.
**Record** sec/epoch (CNN and YOLO) and peak VRAM, then **clean up** SMOKE artefacts:
```powershell
python -m metalai.common.cleanup --exp SMOKE          # dry run - check the list looks right
python -m metalai.common.cleanup --exp SMOKE --yes
python -m metalai.m1_surface.report
```
OOM → retry with CNN `--batch-size 16` / YOLO `--batch 8`; say which batch sizes worked (Claude will put them in `configs/m1.yaml`).

### T5 — Classical baseline (exp `E001`, CPU)
```powershell
$env:METALAI_EXP = "E001"
python -m metalai.m1_surface.classical --n-jobs 4     # runs in parallel with the GPU lane (0.4)
python -m metalai.m1_surface.report                   # orchestrator only, after jobs finish
```
**Report:** `[featset] fold k` lines + `==>` lines; "Classical" tables from `reports/m1/REPORT.md`; `reports/m1/classical/confusion_*.png`. Fill E001 in `experiments/EXPERIMENTS.md`.

### T6 — Fold-0 training for all models + Grad-CAM — CHECKPOINT
Use the batch sizes that worked in T4. Run the GPU monitor (0.1 step 3) for each experiment.
```powershell
$env:METALAI_EXP = "E002"
python -m metalai.m1_surface.cnn --model resnet50 --folds 0
python -m metalai.m1_surface.gradcam --model resnet50 --fold 0

$env:METALAI_EXP = "E003"
python -m metalai.m1_surface.cnn --model efficientnet_b0 --folds 0
python -m metalai.m1_surface.gradcam --model efficientnet_b0 --fold 0

$env:METALAI_EXP = "E004"
python -m metalai.m1_surface.yolo_train --folds 0

python -m metalai.m1_surface.report
```
**Report (Claude reviews these before anything else runs):**
- Tables from `reports/m1/REPORT.md` (CNN, CNN per-class, CNN cost, YOLO, YOLO per-class, Summary).
- Per-epoch lines for both CNNs (or path to `reports/m1/cnn/E00X_<model>/fold0_history.csv`).
- PNG paths (verify they exist):
  - `reports/m1/cnn/E002_resnet50/fold0_curves.png`, `fold0_confusion.png`
  - `reports/m1/cnn/E003_efficientnet_b0/fold0_curves.png`, `fold0_confusion.png`
  - `reports/m1/gradcam/E002_resnet50_fold0_{correct,errors,boxes}.png` and the E003 ones
  - `reports/m1/yolo/E004_yolov8n_fold0/` → `train_results.png`, `test_confusion_matrix_normalized.png`, PR/F1 curves, `test_val_batch*_pred.jpg`
  - `reports/gpu_logs/E002.csv`, `E003.csv`, `E004.csv`
- Training time, peak VRAM, max temperature per experiment.
- Your observations: confused classes, does Grad-CAM sit on the defect/boxes, over/under-fitting in curves, weak YOLO classes.
Fill E002–E004 in `experiments/EXPERIMENTS.md`, mark T6 DONE, commit, **stop**.

---

## 3. Log (append below — newest at the bottom)

### [Claude] 2026-10-28 — kickoff
- Repo scaffolded (`src/metalai/common`, `src/metalai/m1_surface`, `app/`, `configs/`, `experiments/`, `decisions.md`).
- Verified in my sandbox on a synthetic 180-image dataset: `prepare_data`, `splits`, `eda`, `classical`, `yolo_data`, `report`, experiment logging all run end-to-end. `cnn`, `gradcam`, `yolo_train`, `app`, `app/smoke_test.py` are syntax-checked only (no GPU/torch in my sandbox) → T4 exists for that.

### [Claude → OpenCode] 2026-10-28 — instructions for this round
**Update:** use the parallel schedule + subagents in §0.4 (GPU lane sequential; CPU/download/monitoring/report-drafting in parallel; only the orchestrator commits).
Do **T1 → T6 without waiting for me**, unless something fails that you can't fix with a tiny change (then STOP and report). After **T6** stop (CHECKPOINT); I'll review the figures and write T7+.
- Madhav should not have to do anything. Handle setup, Kaggle auth (token is in `.secrets\`), downloads and checks yourself.
- GPU: everything trains on the local **RTX 4050**; follow 0.1 (verify dGPU, power check, `nvidia-smi` log per experiment, OOM fallback).
- Keep code organised (0.2) and experiments organised (0.3): set `$env:METALAI_EXP` before each run, fill `experiments/EXPERIMENTS.md`, never overwrite or hand-edit results.
- Report after each task with the template and commit. The T6 report must list PNG paths — I will open them.
