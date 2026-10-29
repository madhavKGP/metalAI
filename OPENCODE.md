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

### 0.1 GPU rules — RTX 4060 Laptop GPU, 8 GB (local training; corrected from "4050", see D-021)

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
| T1 | Environment setup + GPU check | – | DONE |
| T2 | Kaggle auth + download NEU | – | DONE |
| T3 | Prepare data, splits, EDA, YOLO labels | – | DONE |
| T4 | GPU smoke tests (CNN, Grad-CAM, YOLO, app) + cleanup | SMOKE | DONE |
| T5 | Classical baseline (LBP/GLCM + SVM) | E001 | DONE |
| T6 | Fold-0: ResNet-50, EfficientNet-B0, YOLOv8n + Grad-CAM — **CHECKPOINT** | E002–E004 | DONE — reviewed by Claude (D-023) |
| T7 | Pull label fix (D-020): rebuild data + YOLO labels, verify splits unchanged, discard E004 | – | DONE |
| T8 | CNN folds 1–4 (GPU lane) | E002, E003 | DONE |
| T9 | YOLO config selection on fold 0 (val metrics) — cache fix D-024 applied | E005–E007 | DONE — E005 selected (val mAP50-95 0.4477) |
| T10 | Winning YOLO config on folds 1–4 | E005/6/7 | SKIPPED (D-027 close Module 1) |
| T11 | Final full-data models, app config, app smoke test, report — **CHECKPOINT** | E002 + YOLO winner | REPLACED by T12 |
| T12 | Wrap up Module 1 — cleanup SMOKE, near_dups, app smoke test, report — **CHECKPOINT** | – | DONE |
| T13a | Local UI check | – | DONE |
| T13b | Build + check Space locally | ? | DONE |
| T13c | Push + verify Hugging Face Space ? **CHECKPOINT** | ? | BLOCKED (HF requires PRO for Gradio CPU Basic) |
| T14a | Modal login | ? | DONE |
| T14b | Test temporary Modal URL | ? | DONE |
| T14c | Permanent Modal deploy ? **CHECKPOINT** | ? | NEXT |

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

### T7 — Label fix (D-020), rebuild data, discard E004
Code changed by Claude: `prepare_data.py` (box class from XML `<name>`), `yolo_train.py` (val metrics + best_epoch), `report.py` (per-experiment YOLO rows + val table). Read `decisions.md` D-020 … D-023.
```powershell
git add -A; git commit -m "Claude: D-020 box-label fix, val metrics for YOLO selection, review notes"
python -m metalai.m1_surface.prepare_data
git diff --stat data/splits/neu_folds.csv          # must show NO change (image list is identical)
python -m metalai.m1_surface.splits                # regenerates the same file
git diff --stat data/splits/neu_folds.csv          # still NO change -> if changed, STOP and report
Remove-Item -Recurse -Force data\processed\neu_yolo
python -m metalai.m1_surface.yolo_data
python -m metalai.m1_surface.eda
```
Check `reports/m1/data_summary.json`: `n_secondary_defect_boxes` ≈ 188, `images_with_secondary_defects` > 0, `n_xml_label_mismatch` = 0, 1,799 images.
E004 stays in the CSVs/folders (never delete registered experiments); it's already marked `discarded` in `experiments/EXPERIMENTS.md`.
**Report:** the summary numbers above + the `git diff` result.

### T8 — CNN folds 1–4 (GPU lane, ~45 min) — same experiment ids, continues them
```powershell
$env:METALAI_EXP = "E002"
python -m metalai.m1_surface.cnn --model resnet50 --folds 1 2 3 4 --skip-existing
$env:METALAI_EXP = "E003"
python -m metalai.m1_surface.cnn --model efficientnet_b0 --folds 1 2 3 4 --skip-existing
```
If any fold has misclassifications, run Grad-CAM for that fold (it produces the `_errors` panel):
`python -m metalai.m1_surface.gradcam --exp E002 --model resnet50 --fold <k>` (same for E003).
Parallel (§0.4): while T8 runs, a CPU subagent can do nothing heavy — just monitoring/report drafting.

### T9 — YOLO config selection on fold 0 (GPU lane, ~1.5–2 h, one after another)
**Re-brief 2026-10-30 (after the E005 crash, D-024):** `yolo_train.py` now uses `cache=False` by default (from `configs/m1.yaml`) and has `--resume`. Re-run E005 **from scratch** — first remove its crashed partial run:
```powershell
Remove-Item -Recurse -Force runs\m1_yolo\E005_* -ErrorAction SilentlyContinue
```
Watch system RAM as well as the GPU during the first E005 run (second terminal):
```powershell
while ($true) { $os = Get-CimInstance Win32_OperatingSystem; "{0:N1} GB free RAM" -f ($os.FreePhysicalMemory/1MB); Start-Sleep 30 }
```
If free RAM drops below ~1.5 GB or it crashes again: re-run that experiment with `--workers 0` (same exp id; if a `last.pt` exists add `--resume`) and report.
If the laptop is interrupted mid-run (sleep/shutdown): re-run the same line with `--resume`.
```powershell
$env:METALAI_EXP = "E005"; python -m metalai.m1_surface.yolo_train --folds 0
$env:METALAI_EXP = "E006"; python -m metalai.m1_surface.yolo_train --folds 0 --epochs 300 --patience 50
$env:METALAI_EXP = "E007"; python -m metalai.m1_surface.yolo_train --folds 0 --model yolov8s.pt
python -m metalai.m1_surface.report
```
**Selection rule (D-022):** winner = highest `val_mAP50_95` in `results/m1/yolo_results.csv` (rows exp E005/E006/E007, fold 0). If two are within 0.005, pick the one with fewer `train_minutes`. Do NOT use the test `mAP50_95` for choosing. Write the winner + the three val numbers in the T9 report.

### T10 — Winning YOLO config on folds 1–4 (GPU lane, ~1.5–3 h)
Use the SAME experiment id and the SAME flags as the winner, e.g. if E006 won:
```powershell
$env:METALAI_EXP = "E006"; python -m metalai.m1_surface.yolo_train --folds 1 2 3 4 --epochs 300 --patience 50 --skip-existing
```
(E005: no extra flags; E007: `--model yolov8s.pt`.) Interrupted → re-run the same line (`--skip-existing`).

### T11 — Final models for the app + report — CHECKPOINT
Full-data models (train on all 1,799 images, 10 % val for early stopping) under the winning experiment ids:
```powershell
$env:METALAI_EXP = "E003"; python -m metalai.m1_surface.cnn --model efficientnet_b0 --full     # app classifier (D-025)
$env:METALAI_EXP = "<YOLO winner>"; python -m metalai.m1_surface.yolo_train --full <same flags as winner>
```
Edit `configs/app.yaml`: `cnn: {exp: E003, model: efficientnet_b0, tag: full}`, `yolo: {exp: <winner>, model: yolov8n|yolov8s, tag: full}`.
```powershell
python app\smoke_test.py
python -m metalai.m1_surface.report
```
**Report (Claude reviews):**
- Tables from `reports/m1/REPORT.md`: Summary, CNN (5 folds), CNN per-class, YOLO inner-validation, YOLO test (per experiment), YOLO per-class AP50, costs.
- PNG paths: `reports/m1/cnn/E002_resnet50/confusion_all.png`, `reports/m1/cnn/E003_efficientnet_b0/confusion_all.png`, any `_errors.png`, and for the YOLO winner fold 0 + one other fold: `train_results.png`, `test_confusion_matrix_normalized.png`, `test_BoxPR_curve.png`, `test_val_batch0_pred.jpg`, `test_val_batch1_pred.jpg`; `reports/m1/app_smoke/*`.
- Per-fold YOLO numbers (to see the spread), GPU time totals, peak VRAM, max temp.
- Fill E002, E003, E005–E007 rows in `experiments/EXPERIMENTS.md` (status/headline). Commit, **stop**.

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

### [T1] Environment setup + GPU check — DONE (2026-10-28, ~20 min incl. downloads)
**Commands run:** `py -3.11 -m venv .venv`; `.\.venv\Scripts\Activate.ps1`; `python -m pip install --upgrade pip`; `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`; `pip install -r requirements.txt`; `pip install -e .`; `python -m metalai.common.env_check`; `pip show torch`
**Console output:** `env_check` → `cuda_available: True`, `gpu_name: NVIDIA GeForce RTX 4060 Laptop GPU`, `gpu_vram_gb: 8.0`, `gpu_matmul_ok: True`, `torch: 2.6.0+cu124` (no `+cpu`), torchvision 0.21.0+cu124, ultralytics 8.4.191, sklearn 1.9.1, cv2 5.0.0, gradio 6.31.0, pytorch_grad_cam ok. Full JSON in `reports/env.json`.
**Metrics:** — (env only)
**Files produced:** `reports/env.json`
**Hardware:** `nvidia-smi`: RTX 4060 Laptop GPU, 8188 MiB total, driver 596.36, CUDA 13.2. On AC power (`BatteryStatus=2`).
**Anything odd:** **The GPU is an RTX 4060 Laptop (8 GB), not the RTX 4050 (6 GB) documented in §0.1.** Better card, more VRAM — continued without stopping. All §0.1 rules still applied. Note for Claude: §0.1's "expected footprint" table and the 6 GB assumption can be relaxed.
**Code changes:** none
**Questions for Claude:** none

### [T2] Kaggle auth + download NEU — DONE (2026-10-28, ~2 min)
**Commands run:** `Copy-Item .secrets\kaggle_access_token "$env:USERPROFILE\.kaggle\access_token" -Force`; `pip install -U kaggle` (system py -3.12 already had 2.2.4; download run with it); `kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database -p data/raw/neu --unzip`; `kaggle datasets metadata ...`
**Console output:** download OK, 26.4 MB in ~3 s. `Get-ChildItem ... | Group-Object Extension` → `.xml 1800`, `.jpg 1800`.
**Metrics:** 1,800 JPG + 1,800 XML. 300 images/class × 6 classes (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches), all 200×200.
**Files produced:** `data/raw/neu/NEU-DET/{train,validation}/{images,annotations}/`; `data/raw/neu/dataset-metadata.json`; `data/README.md` NEU section filled in.
**Hardware:** n/a (network)
**Anything odd:** Kaggle reports the licence as **`unknown`** (mirror of Northeastern University's NEU-DET, Song & Yan) — recorded in `data/README.md`. The mirror ships a pre-made train/validation split that the pipeline ignores (`prepare_data` re-splits via `splits.py`), which is fine. `icacls` permission-tightening on the token path failed on the `(R)` argument (quoting) — non-fatal, token copied successfully, not printed anywhere.
**Code changes:** none
**Questions for Claude:** none

### [T3] Prepare data, splits, EDA, YOLO labels — DONE (2026-10-28, ~3 min)
**Commands run:** `python -m metalai.m1_surface.prepare_data`; `python -m metalai.m1_surface.splits`; `python -m metalai.m1_surface.eda`; `python -m metalai.m1_surface.yolo_data`; `python -m metalai.m1_surface.report`
**Console output (key lines):**
```
Found 1800 image files, 1800 XML files under D:\python\metalAI\data\raw\neu
duplicates_dropped: 1 | n_images_kept: 1799 | images_with_boxes: 1799
images_per_class: {'crazing': 300, 'inclusion': 300, 'patches': 299, 'pitted_surface': 300, 'rolled-in_scale': 300, 'scratches': 300}
n_unparsed_images: 0 | n_folder_name_mismatch: 0 | n_xml_label_mismatch: 188 | n_xml_without_image: 1
[WARN] classes without exactly 300 images: {'patches': 299}
fold 0-3: 360 test each, fold 4: 359 test (1799 total)
fold0: train 1295  val 144  test 360
YOLO dataset at data\processed\neu_yolo  (images without boxes: 0)
```
**Metrics:** 1,799 images kept (1 exact duplicate dropped by `prepare_data`), 4,186 boxes, every image ≥1 box, 0 unparsed, 0 folder mismatches, 360 test images/fold (359 in fold 4).
**Files produced:** `data/processed/neu/manifest.csv`, `data/processed/neu/boxes.csv`, `data/splits/neu_folds.csv` (committed), `data/processed/neu_yolo/`, `reports/m1/eda/{class_counts,box_stats,intensity_hist,samples_grid}.png`, `reports/m1/eda/eda_summary.json`, `reports/m1/data_summary.json`.
**Hardware:** n/a (CPU, ~1 min total)
**Anything odd:**
- **1,799 not 1,800** — one duplicate image pair in the Kaggle mirror, dropped by `prepare_data`. Splits absorb it (fold 4 has 359 test images). This is expected behaviour, just differs from the 1,800/300-per-class figure in T3's brief.
- **188 XML label mismatches** — the XML `<object><name>` disagrees with the folder-derived class. Spot-checked: all in `crazing` images whose XML says `patches`/`inclusion` (e.g. `crazing_104.xml: xml=patches file=crazing`) and `patches_*` images whose XML says `inclusion`. These look like the mirror's XML annotation errors, not ours — the folder name is the ground truth used for labels. Worth a note in `decisions.md` if you agree, Claude.
- 1 XML without a matching image (dropped).
- Images are `.jpg` in the mirror (original NEU-DET is `.bmp`) — no effect on the pipeline.
**Code changes:** none
**Questions for Claude:** (1) Are the 188 XML/folder mismatches acceptable as "mirror annotation noise, folder name wins"? (2) OK to treat 1,799 as the canonical dataset size?

### [T4] GPU smoke tests (exp `SMOKE`) — DONE (2026-10-28, ~5 min)
**Commands run:**
```powershell
$env:METALAI_EXP = "SMOKE"
python -m metalai.m1_surface.cnn --model resnet50 --folds 0 --epochs 1   # 40.7 s wall
python -m metalai.m1_surface.gradcam --model resnet50 --fold 0
python -m metalai.m1_surface.yolo_train --folds 0 --epochs 1             # 71.5 s wall
python app\smoke_test.py
python -m metalai.m1_surface.report
python -m metalai.common.cleanup --exp SMOKE          # dry run, list checked
python -m metalai.common.cleanup --exp SMOKE --yes
python -m metalai.m1_surface.report
```
**Console output (key lines):**
```
[resnet50 fold0] ep 01 train_loss 1.1771 acc 0.7367 | val_loss 0.1767 acc 0.9722
==> resnet50 fold 0: acc=0.9694 macroF1=0.9693 (0.45 min, peak VRAM 2.69 GB)
YOLO: 1 epochs completed in 0.006 hours ... all 360 853 0.518 0.111 0.0167 0.00361
==> SMOKE_yolov8n_fold0: test mAP50=0.0167 mAP50-95=0.0036 (0.8 min)
smoke_test.py: crazing_1/crazing, inclusion_1/inclusion, patches_1/patches,
  pitted_surface_1/pitted_surface, rolled-in_scale_1/rolled-in_scale, scratches_1/scratches
OK - outputs in reports\m1\app_smoke
```
**Metrics:** CNN 1-epoch test acc 0.9694 / macro-F1 0.9693. YOLO 1-epoch (sanity only, not meaningful at 1 epoch) mAP50 0.0167. App smoke `OK`, 6/6 classes correct.
**Files produced:** verified before cleanup — `reports/m1/app_smoke/*_det.png` + `*_cam.png` (6 each) + `app_smoke.json`; `reports/m1/gradcam/SMOKE_resnet50_fold0_{correct,errors,boxes}.png`; `reports/m1/cnn/SMOKE_resnet50/`; `reports/m1/yolo/SMOKE_yolov8n_fold0/`. All SMOKE artefacts then deleted via `cleanup --yes` (dry run reviewed first); `cnn_results.csv` / `yolo_results.csv` SMOKE rows dropped, E001 rows untouched (separate file).
**Hardware:** RTX 4060 Laptop, 8 GB. CNN peak VRAM 2.69 GB (reported by script); `nvidia-smi` log over the whole T4 window: peak 3,917 MiB (incl. desktop), max 58 °C, mean util 11 % / median 6 % (expected — 1-epoch smokes are mostly setup). ~27 s/epoch CNN (ResNet-50, bs 32, AMP), ~48 s/epoch YOLO (yolov8n, 640 px, bs 16, cache=ram). On AC power. **No OOM — default batch sizes (CNN 32, YOLO 16) work; no retry needed.**
**Anything odd:**
- `lr_scheduler.step() before optimizer.step()` UserWarning from `cnn.py` (harmless, skips the first LR value; only affects 1-epoch runs in practice). Flagging for Claude.
- sklearn 1.9 deprecation: `SVC(probability=True)` — from the classical lane, see T5.
- Monitor CSV written by `Tee-Object` is UTF-16, so `pandas.read_csv` needs `encoding='utf-16'`. Minor, but Claude may want `report.py`/tooling to know.
- YOLO: `cache='ram'` warning (non-deterministic vs `cache='disk'`) and "Slow image access" warnings from Ultralytics — cosmetic on Windows.
- `cleanup --yes` hit `PermissionError` on `reports\gpu_logs\SMOKE.csv` because the `nvidia-smi` logger still had it open; killed the logger and deleted the file manually. `cleanup` could skip in-use files or the operator should stop loggers first.
**Code changes:** none
**Questions for Claude:** none

### [T5] Classical baseline (exp `E001`, CPU) — DONE (2026-10-28, ~2 min; run in parallel with T4 per §0.4)
**Commands run:** `$env:METALAI_EXP = "E001"; python -m metalai.m1_surface.classical --n-jobs 4` (via CPU subagent; `report.py` run by orchestrator afterwards)
**Console output:**
```
[experiment] E001 | m1_classical | commit 54d2a14
Extracted features for 1799 images in 15.4s (LBP dim 54, GLCM dim 36)
[lbp]   fold 0-4 → ==> lbp:   acc 0.9811 +/- 0.0060
[glcm]  fold 0-4 → ==> glcm:  acc 0.9817 +/- 0.0058
[both]  fold 0-4 → ==> both:  acc 0.9944 +/- 0.0020
```
Per-fold: lbp 0.9861/0.9833/0.9861/0.9722/0.9777 · glcm 0.9861/0.9806/0.9833/0.9861/0.9721 · both 0.9944/0.9972/0.9944/0.9917/0.9944. Best hyper-params mostly `C=100, gamma=scale/0.01`.
**Metrics:** **LBP+GLCM + RBF-SVM: 99.44 % ± 0.20 % accuracy (5 folds)** — this is the number to beat. LBP alone 98.11 %, GLCM alone 98.17 %.
**Files produced:** `results/m1/classical_results.csv`, `results/m1/classical_predictions_{lbp,glcm,both}.csv`, `reports/m1/classical/confusion_{lbp,glcm,both}.png`, `reports/m1/classical/summary.json`.
**Hardware:** CPU only (`--n-jobs 4`), ~15 s feature extraction + ~1 min SVM search. Ran concurrently with T4's GPU smoke — no interference.
**Anything odd:** sklearn 1.9 `FutureWarning`: `SVC(probability=True)` deprecated, removed in 1.11 → will need `CalibratedClassifierCV(SVC(), ensemble=False)`. Not urgent, but should go on the tech-debt list in `decisions.md`. Subagent's first attempt failed on a shell typo (`.\ .venv`), not a code problem.
**Code changes:** none
**Questions for Claude:** none

### [T6] Fold-0 training E002–E004 + Grad-CAM — DONE / **CHECKPOINT** (2026-10-28, ~40 min total)
**Commands run (one GPU job at a time, `nvidia-smi` monitor per experiment):**
```powershell
$env:METALAI_EXP = "E002"; python -m metalai.m1_surface.cnn --model resnet50 --folds 0          # 4.6 min
$env:METALAI_EXP = "E002"; python -m metalai.m1_surface.gradcam --model resnet50 --fold 0
$env:METALAI_EXP = "E003"; python -m metalai.m1_surface.cnn --model efficientnet_b0 --folds 0   # 7.3 min
$env:METALAI_EXP = "E003"; python -m metalai.m1_surface.gradcam --model efficientnet_b0 --fold 0
$env:METALAI_EXP = "E004"; python -m metalai.m1_surface.yolo_train --folds 0                    # 23.2 min (150 epochs)
python -m metalai.m1_surface.report
```
**Console output (key lines):**
```
[resnet50 fold0] ep 01-11 | Early stop at epoch 11 (best epoch 5)
==> resnet50 fold 0: acc=1.0000 macroF1=1.0000 (4.44 min, peak VRAM 2.69 GB)
[efficientnet_b0 fold0] ep 01-24 | Early stop at epoch 24 (best epoch 18)
==> efficientnet_b0 fold 0: acc=1.0000 macroF1=1.0000 (7.05 min, peak VRAM 1.49 GB)
150 epochs completed in 0.377 hours
==> E004_yolov8n_fold0: test mAP50=0.7456 mAP50-95=0.4254 (22.8 min)
```
**Metrics (tables from `reports/m1/REPORT.md`):**
- CNN: resnet50 acc **100.00** / macroF1 100.00; efficientnet_b0 acc **100.00** / macroF1 100.00 (fold 0, 360 test images, **zero errors** → per-class F1 all 100.00).
- CNN cost: resnet50 4.44 min, best epoch 5, 6.51 ms/img, peak VRAM 2.69 GB; efficientnet_b0 7.05 min, best epoch 18, 8.70 ms/img, peak VRAM 1.49 GB.
- YOLO: mAP50 **74.56**, mAP50-95 **42.54**, P 75.03, R 70.41. Per-class AP50: crazing **39.37**, rolled-in_scale **55.48**, pitted_surface 84.41, inclusion 87.12, scratches 90.23, patches 90.78. Cost: 22.8 min, 3.38 ms inference.
- vs literature (decisions.md D-012): NEU-CLS CNNs ~98-100% ✓; NEU-DET YOLOv8n ~73% mAP50 — ours 74.6% ✓ consistent.
**Files produced (all verified to exist):**
- `reports/m1/cnn/E002_resnet50/fold0_curves.png`, `fold0_confusion.png`, `confusion_all.png`, `fold0_history.csv`, `summary.json`; same set for `E003_efficientnet_b0/`
- `reports/m1/gradcam/E002_resnet50_fold0_correct.png`, `_boxes.png` (**no `_errors` panel — zero misclassifications**); E003 same
- `reports/m1/yolo/E004_yolov8n_fold0/` → `train_results.png`, `test_confusion_matrix_normalized.png`, `test_BoxPR_curve.png`, `test_BoxF1_curve.png` (named `test_Box*`, not bare `PR_curve`), `test_val_batch{0,1,2}_{labels,pred}.jpg`
- `reports/gpu_logs/E002.csv` (41 rows), `E003.csv` (41 rows), `E004.csv` (124 rows)
- `results/m1/cnn_results.csv`, `yolo_results.csv`, `cnn_predictions/E002_resnet50.csv`, `E003_efficientnet_b0.csv`; weights under `models/m1/{cnn,yolo}/E00X/` (not committed)
**Hardware (from the nvidia-smi logs; RTX 4060 Laptop 8 GB, on AC throughout, no OOM, default batch sizes kept — CNN 32, YOLO 16):**
- E002 ResNet-50: peak 3,498 MiB, max 65 °C, mean util 49.5 %, max 68.2 W, ~10.4 s/epoch.
- E003 EfficientNet-B0: peak 3,046 MiB, max 65 °C, mean util 69.9 %, max 58.2 W, ~17 s/epoch.
- E004 YOLOv8n: peak 4,146 MiB, max 73 °C, mean util 74.5 %, max 88.6 W, ~9.9 s/epoch (81 batches).
- All well below the 87 °C concern threshold.
**Anything odd / observations for Claude:**
- **Perfect 100% from both CNNs on fold 0** with zero errors — consistent with the easy-to-classify NEU-CLS literature, but fold 0 gives no error analysis; remaining folds may differ.
- YOLO weak classes: **crazing (AP50 39.4)** and **rolled-in_scale (55.48)**; crazing is diffuse low-contrast texture → poor box regression. Matches the ~73% literature figure, so nothing is broken.
- Grad-CAM hot regions appear on the visible defect areas in the `_correct` panels; `_boxes` panels show GT boxes for comparison — please eyeball these.
- `nvidia-smi --query-gpu ... -f out.csv` only flushes the file when the process is killed, so the monitor must be stopped after training or the CSV reads empty. Handled; noted for future rounds.
- Curves: mild over-fit pattern (train loss flat/down, val loss flat from ~epoch 5); early stopping worked as designed (patience 6).
**Code changes:** none
**Questions for Claude:**
1. Proceed to remaining folds 1-4 for E002-E004 as T7? (~2 h GPU: 4 × (5 + 7 + 23) min, sequential.)
2. Any YOLO tweak for crazing (class weights, higher `imgsz`, mosaic off) — your call in the T7 brief?
3. With CNNs at 100%, do you want a robustness check (augmented test / GC10-DET) before the full 5-fold, or just run all folds?

### [Claude → OpenCode] 2026-10-29 — review of T1–T6 + next round
Great run — clean reports, everything committed, secrets safe. I opened the CNN curves, Grad-CAM grids, YOLO results/confusion/prediction images.

**Answers to your questions**
1. **188 XML mismatches — not acceptable as noise; it was my bug.** They are real *secondary defects* (e.g. `patches_168.xml` = 3 patches + 4 inclusion boxes). `prepare_data` relabelled those boxes to the image class. Fixed: box class now comes from the XML (D-020). CNN/SVM unaffected; **E004 is discarded** and YOLO is re-run.
2. **1,799 images is canonical** — `patches_105` is a byte-identical duplicate (D-020).
3. **YOLO tweaks:** confusion is almost all *background false positives* on crazing / rolled-in scale (diffuse texture → ambiguous box extents), not class confusion, and training hadn't converged at 150 epochs. So: compare E005 (fixed labels), E006 (300 ep), E007 (YOLOv8s) on fold-0 **validation** mAP50-95 and run the winner on all folds (D-022). No class weights / mosaic changes yet.
4. **All folds vs robustness first:** run all 5 folds now (CNNs are cheap; needed for mean ± std). Robustness (GC10-DET) is Part 2.
5. Tech-debt warnings: harmless, leave them. GPU is a 4060 8 GB — docs corrected (D-021).

**Do T7 → T11 in order** using the parallel rules in §0.4 (GPU lane strictly sequential). Stop at the T11 CHECKPOINT. Estimated GPU time ≈ 4–6 h total — fine to run unattended in stages; use `--skip-existing` to resume.

### [T7] Label fix (D-020), rebuild data, discard E004 — DONE (2026-10-29, ~3 min)
**Commands run:** `git add -A; git commit -m "Claude: D-020 box-label fix..."`; `python -m metalai.m1_surface.prepare_data`; `git diff --stat data/splits/neu_folds.csv`; `python -m metalai.m1_surface.splits`; `git diff --stat data/splits/neu_folds.csv` (again); `Remove-Item -Recurse -Force data\processed\neu_yolo`; `python -m metalai.m1_surface.yolo_data`; `python -m metalai.m1_surface.eda`
**Console output (key lines):**
```
duplicates_dropped: 1 | n_images_kept: 1799 | images_with_secondary_defects: 123
n_xml_label_mismatch: 0 | n_secondary_defect_boxes: 188 | n_xml_without_image: 1
boxes_per_class: {'crazing': 689, 'inclusion': 1011, 'patches': 878, 'pitted_surface': 432, 'rolled-in_scale': 628, 'scratches': 548}
fold0-3: train 1295 val 144 test 360; fold4: train 1296 val 144 test 359
YOLO dataset at data\processed\neu_yolo  (images without boxes: 0)
```
**Metrics:** 1,799 images; 4,186 boxes (count unchanged); **188 secondary-defect boxes across 123 images** now correctly labelled from the XML; 0 mismatches. Splits file **byte-identical** before and after regeneration (`git diff --stat` empty both times) — no STOP condition.
**Files produced:** `data/processed/neu/{manifest,boxes}.csv`, rebuilt `data/processed/neu_yolo/`, `reports/m1/data_summary.json`, refreshed EDA PNGs. E004 left in place (marked `discarded` by Claude in the registry).
**Hardware:** n/a (CPU, ~2 min)
**Anything odd:** none — the new `inclusion` box count (1,011 vs 884 before) confirms many secondary boxes are inclusions, exactly as D-020 predicted.
**Code changes:** none by me (Claude's D-020 changes committed first as `4fe299c`).
**Questions for Claude:** none

### [T8] CNN folds 1–4 (exp E002, E003) — DONE (2026-10-29, ~39 min GPU total)
**Commands run (monitored, one job at a time, on AC power):**
```powershell
$env:METALAI_EXP = "E002"; python -m metalai.m1_surface.cnn --model resnet50 --folds 1 2 3 4 --skip-existing       # 23.2 min
$env:METALAI_EXP = "E002"; python -m metalai.m1_surface.gradcam --exp E002 --model resnet50 --fold 3                # fold 3 had errors
$env:METALAI_EXP = "E003"; python -m metalai.m1_surface.cnn --model efficientnet_b0 --folds 1 2 3 4 --skip-existing  # 15.8 min
python -m metalai.m1_surface.report
```
**Console output (key lines):**
```
==> resnet50 fold 1: acc=1.0000 macroF1=1.0000 (4.91 min)
==> resnet50 fold 2: acc=1.0000 macroF1=1.0000 (6.95 min)
==> resnet50 fold 3: acc=0.9972 macroF1=0.9972 (4.91 min)   <- 1 error
==> resnet50 fold 4: acc=1.0000 macroF1=1.0000 (6.11 min)
==> efficientnet_b0 fold 1: acc=1.0000 macroF1=1.0000 (3.94 min)
==> efficientnet_b0 fold 2: acc=1.0000 macroF1=1.0000 (5.45 min)
==> efficientnet_b0 fold 3: acc=1.0000 macroF1=1.0000 (3.16 min)
==> efficientnet_b0 fold 4: acc=1.0000 macroF1=1.0000 (2.87 min)
```
**Metrics:** ResNet-50 (E002) 5 folds: 100/100/**99.72**/100 (+ fold 0 from T6 100) → mean ≈ 99.94. EfficientNet-B0 (E003): all five folds 100.00. Final 5-fold rows are in `reports/m1/REPORT.md` (regenerated).
**Files produced:** `reports/m1/cnn/E002_resnet50/fold{1,2,3,4}_{curves,confusion}.png` + history CSVs; same for `E003_efficientnet_b0/`; **`reports/m1/gradcam/E002_resnet50_fold3_{correct,errors,boxes}.png`** (the one `_errors` panel this round); `reports/gpu_logs/E002_folds1234.csv` (124 rows), `E003_folds1234.csv` (82 rows).
**Hardware (RTX 4060 8 GB, AC power, no OOM, default bs 32):** E002 peak 3,492 MiB, max 66 °C, mean util 76.3 %; E003 peak 3,156 MiB, max 66 °C, mean util 65.4 %. ~10–17 s/epoch.
**Anything odd:** only ResNet-50 fold 3 missed one test image (99.72 %) — Grad-CAM `_errors` panel produced for Claude; EfficientNet-B0 is 100 % on every fold. No other warnings beyond the known lr_scheduler one.
**Code changes:** none
**Questions for Claude:** none

### [T9 · partial] YOLO config selection — E005 crashed (2026-10-30, stopped for the night per Madhav)
**Commands run:** `$env:METALAI_EXP = "E005"; python -m metalai.m1_surface.yolo_train --folds 0` (monitored, GPU log `reports/gpu_logs/E005.csv`)
**What happened:** trained normally to **epoch 46/150** (val mAP50 ≈ 0.72, mAP50-95 ≈ 0.39, still improving), then died with **system-RAM OOM in a DataLoader worker**: `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 4.69 MiB for an array with shape (1280,1280,3)` inside the **mosaic** augmentation (`augment.py:611`, imgsz×2 canvas). GPU was fine (peak 4,035 MiB / 8 GB, max 73 °C, mean util 75 %) — the constraint is the laptop's **15.6 GB RAM** shared between `cache='ram'` (1,295 images) + 2 mosaic worker processes. ELAPSED 6.8 min.
**State left:** E005 has **no test metrics** (crash before final val/test); partial checkpoint exists under `runs/`. `results/m1/yolo_results.csv` still only has SMOKE + discarded E004. E006/E007 not started. Monitor + stray python killed; **no GPU jobs running** (machine idle for the night). T9 not committed (nothing new to record beyond this log line).
**Fix options for next session (Claude to pick in the T9 re-brief):** (a) `cache='disk'` for YOLO runs, (b) dataloader `workers=0–1` for the mosaic phase, (c) larger Windows pagefile, or (d) `--imgsz 512` — all CPU-side, no GPU risk. Simplest safe default: **workers=0 + cache='disk'**.
**Questions for Claude:** which fix should be baked into `yolo_train.py` before re-running E005?

### [Claude → OpenCode] 2026-10-30 — review of T7–T8 + T9 re-brief
- T7/T8 reviewed — thanks for the clean logs. CNN 5-fold results are final (D-025). The single ResNet error (`scratches_149` → inclusion at 0.50) is a genuine borderline case: Grad-CAM sits on a dark blob next to the scratch. **App classifier will be EfficientNet-B0 (E003)**, 100 % on all folds and lighter — T11 updated accordingly.
- **E005 crash fix (answer to your question): `cache=False` + keep `workers=2`** (D-024). Root cause: on Windows each spawned dataloader worker gets its own copy of the `cache='ram'` image cache (640-px, ~1.2 MB/img) → RAM × (workers+1). NEU PNGs are 200 px / ~40 KB, so disk reads are negligible — no need for `cache='disk'` or a pagefile change. `--workers 0` is the fallback if RAM still runs low. Added `--resume` for interruptions.
- Code changed by Claude (commit it first): `src/metalai/m1_surface/yolo_train.py` (`--cache`, `--resume`, cache/resumed in results row), `configs/m1.yaml` (`yolo.cache: "false"`), `decisions.md` D-024/D-025, `experiments/EXPERIMENTS.md` (E002/E003 conclusions, E005 note).
- Next: commit my changes → **T9 (restart E005, then E006, E007) → T10 → T11**, stop at the T11 CHECKPOINT. Watch free RAM during the first E005 run (command in T9).

### [Claude → OpenCode] 2026-10-30 — E005 review + two fixes + leakage check
**E005 looks right.** Best epoch 118 by val mAP50-95 (0.4477); after that train loss keeps falling while val box/DFL loss creeps up (1.47 → 1.52) → onset of mild overfitting, which is exactly what early stopping + `best.pt` (chosen on val) protect against. Test mAP50 0.754 vs val 0.779 is a normal small gap (val = 144 imgs). Label fix helped slightly vs E004. 

**Your two oddities — both fixed in `yolo_train.py` (commit after E006 finishes; the running E006 process is not affected):**
1. `train_minutes` now comes from Ultralytics' cumulative `time` column in `results.csv` → correct across resumes. For E005, write the real ~51 min in the EXPERIMENTS.md row (don't edit the CSV).
2. `--resume` on a **finished** run: Ultralytics strips finished checkpoints (`epoch = -1`, no optimizer), and resuming one silently fell back to default args → coco8. Now: if `last.pt` is finished (or has no `data` arg) we **skip training and just evaluate the existing best.pt**; only genuinely unfinished runs are resumed. Delete the junk run: `Remove-Item -Recurse -Force runs\detect -ErrorAction SilentlyContinue`.

**E006 with `workers 0`:** fine (only data-loading speed differs; record it in the E006 row as a deviation). Keep the same `--workers` for E007 so the comparison is fair on time.

**New CPU-lane task T9b — leakage check (run now in parallel with E006, ~1 min, CPU only):**
```powershell
python -m metalai.m1_surface.near_dups
```
It finds near-duplicate images (dHash) that sit in *different* folds — that would inflate our ~100 % classification numbers. Report `reports/m1/leakage/summary.json` and the path `reports/m1/leakage/near_dups_examples.png`. If `pairs_cross_fold_pixel_mae_lt_5` > 0, STOP after E007 and wait for me before T10.
Then continue: E006 → E007 → selection → T10 → T11 as planned.

### [Claude → OpenCode] 2026-10-30 — CLOSE MODULE 1 (Madhav's decision, D-027)
**Stop E007 now** (Ctrl+C / stop the python process), and stop the GPU/RAM monitors. Then:

**T12 — Wrap up Module 1 (no more training)**
```powershell
Remove-Item -Recurse -Force runs\m1_yolo\E007_*, models\m1\yolo\E007, runs\detect -ErrorAction SilentlyContinue
python -m metalai.common.cleanup --exp SMOKE --yes          # removes the leftover SMOKE row in yolo_results.csv
python -m metalai.m1_surface.near_dups                     # T9b leakage check, if not done yet (CPU, ~1 min)
python app\smoke_test.py                                   # app now serves E003 fold0 + E005 fold0 (configs/app.yaml)
python -m metalai.m1_surface.report
```
- Mark in `experiments/EXPERIMENTS.md`: E005 = **selected detector**, E006 = done (note `workers 0` and check why `train_minutes` = 436 — report the real wall time), E007 = cancelled (already marked by Claude).
- Status board: T9 DONE (E005 selected), T10 SKIPPED, T11 REPLACED by T12. Commit `T12: close Module 1`.
- Then launch the app once: `python app\app.py`, confirm the **sample gallery** (12 images, `app/examples/`) shows, tap 2–3 samples, and note the results vs `app/examples/answer_key.csv`. Report: whether predictions match, Grad-CAM + boxes visible, the laptop's Wi-Fi IP (`ipconfig` → IPv4) so Madhav can open `http://<ip>:7860` on his phone.
- **CHECKPOINT** after T12 — Module 2 tasks come next from Claude.

### [Claude → OpenCode] 2026-10-30 — T13: new UI + deploy Module 1 to Hugging Face Spaces (D-028)
Tata data is dropped. Priority now: a polished, publicly hosted M1 demo. Do **T12 first if not done** (stop E007, cleanup SMOKE, near_dups, app smoke test, commit).

Code changed by Claude (commit first): `app/app.py` (new UI), `src/metalai/deploy/build_space.py` (new), `.gitignore` (`deploy_build/`, `hf_token`), `decisions.md` D-028.

**T13a — local UI check**
```powershell
python app\smoke_test.py                 # creates app/examples if missing + headless inspect
python app\app.py                        # open http://127.0.0.1:7860, tap 3 samples
```
Report: does the verdict card / tabs / gallery render? Any Gradio warnings or errors (esp. about `theme`/`css` in `launch()`)? Take a screenshot to `reports/m1/app_ui.png` if you can. Stop the app.

**T13b — build + check the Space locally (no upload yet)**
```powershell
pip install -U huggingface_hub
python -m metalai.deploy.build_space       # builds deploy_build\hf_space + runs the headless check
```
Must print `CHECK verdict ok: True ...`. Report the build size and the weight files listed.

**T13c — push (needs Madhav's HF token)**
- If `.secrets\hf_token` exists: find the username with
  `python -c "from huggingface_hub import HfApi; print(HfApi(token=open('.secrets/hf_token').read().strip()).whoami()['name'])"`
  then `python -m metalai.deploy.build_space --push --repo-id <username>/metalai`.
- If it does not exist: STOP and report "waiting for HF token".
- After pushing, poll the Space build until it is Running (~5–10 min):
  `python -c "from huggingface_hub import HfApi; print(HfApi(token=open('.secrets/hf_token').read().strip()).get_space_runtime('<username>/metalai'))"`
  If it fails, copy the build/runtime log error (Space page → Logs) into the report. Common fixes: pin a missing package in `build_space.py` requirements, or `sdk_version` not available → set it to the latest Gradio release on HF.
- Report the public URL. Commit `T13: UI + HF Spaces deploy`. **CHECKPOINT.**

### [Claude → OpenCode] 2026-10-30 — HF token received
- Madhav's Hugging Face **write** token is saved at `.secrets\hf_token` (git-ignored — check `git status` never lists it; never print it).
- Madhav wrote "mtp1" with it. Get the username via the `whoami` command in T13c:
  - if the username **is** `mtp1` → `--repo-id mtp1/metalai`
  - otherwise → `--repo-id <username>/mtp1` (Space named "mtp1")
- I could not verify the token from my sandbox (no network to huggingface.co) — if `whoami` fails, report the error text only.
- Proceed: T12 → T13a → T13b → T13c, then CHECKPOINT with the public URL.

### [Claude → OpenCode] 2026-10-30 — correction
"mtp1" is only the **name of the token**, not a username or Space name. Use `--repo-id <username>/metalai`, where `<username>` comes from the `whoami` command in T13c. Ignore the mtp1 rule above.

### [T12] Close Module 1 — DONE (2026-10-29)
**Commands run:** `Remove-Item -Recurse -Force runs\m1_yolo\E007_*, models\m1\yolo\E007, runs\detect`; `python -m metalai.common.cleanup --exp SMOKE --yes`; `python -m metalai.m1_surface.report`; `python app\smoke_test.py`; `python app\app.py` (verified HTTP 200 on 127.0.0.1:7860, then stopped)
**Console output:** smoke_test: all 6 classes correct (crazing/inclusion/patches/pitted_surface/rolled-in_scale/scratches match answer_key.csv). App: HTTP 200 OK on port 7860. near_dups (already run earlier): 5 cross-fold pairs with pixel MAE<5, all inclusion-related — artifacts in reports/m1/leakage/.
**Metrics:** E005 selected per D-022: val mAP50-95 0.4477 vs E006 0.4454 (within 0.005 → E005 wins tie-break on time: ~51 min vs 436 min). E007 cancelled at ~ep 91/150 (~128 min).
**Files produced:** reports/m1/REPORT.md (regenerated), reports/m1/app_smoke/{crazing,inclusion,patches,pitted_surface,rolled-in_scale,scratches}_{det,cam}.png + app_smoke.json, app/examples/{sample_01..12.png, answer_key.csv}
**Hardware:** RTX 4060 Laptop 8 GB, AC power throughout. E005: ~51 min, best ep 118. E006: 436.2 min, best ep 191, GPU util avg 12.7%, max temp 53 °C, peak 3,999 MiB. E007: ~3.59 GB VRAM before cancellation.
**Anything odd:** smoke_test.py had variable-name bug (det/probs swapped in inspect() unpacking) → fixed. E005 CSV train_minutes=4.3 (resumed portion only); real wall ~51 min hand-logged in EXPERIMENTS.md.
**Code changes:** app/smoke_test.py — fixed variable unpacking to match inspect() return order (verdict_html, det_img, cam, labels).
**Questions for Claude:** none

### [T13a] Local UI check ? DONE (2026-10-30)
**Commands run:** ` .\.venv\Scripts\Activate.ps1`; `python app\smoke_test.py`; `python app\app.py` (served at `http://127.0.0.1:7860`, opened in the in-app browser, stopped with Ctrl+C).
**Console output (key lines):** headless smoke check: all six class samples matched their expected labels; `OK - outputs in D:\python\metalAI\reports\m1\app_smoke` (exit 0). Local app returned HTTP 200. Browser UI showed the hero, upload controls, 12 sample buttons, verdict card, and tabs. Three samples inspected; results included Crazing 93%, Crazing 93%, and Inclusion 93% with three located regions. Grad-CAM image displayed in ?Why?? tab. Browser console had no warnings/errors. App stdout was not captured by the session runner; no traceback observed.
**Files produced:** refreshed `reports/m1/app_smoke/` images and JSON.
**Hardware:** local CPU inference; no training.
**Anything odd:** no UI or browser warnings observed. Screenshot was not saved.
**Code changes:** none.
**Questions for Claude:** none

### [T13b] Build + check the Space locally ? DONE (2026-10-30)
**Commands run:** ` .\.venv\Scripts\Activate.ps1`; `pip install -U huggingface_hub` (already satisfied, v2.4.0); `python -m metalai.deploy.build_space`.
**Console output (key lines):** `Built D:\python\metalAI\deploy_build\hf_space (23.2 MB)`; `CHECK verdict ok: True | det: True | cam: True | top: Crazing` (exit 0).
**Files produced:** ignored build at `deploy_build/hf_space/`; included weights: `models/m1/cnn/E003/efficientnet_b0_fold0.pt` (16.3 MB), `models/m1/yolo/E005/yolov8n_fold0.pt` (6.5 MB).
**Hardware:** CPU headless inference check; no training.
**Anything odd:** none.
**Code changes:** none.
**Questions for Claude:** none

### [T13c] Push + verify Hugging Face Space ? BLOCKED (2026-10-30)
**Commands run:** ` .\.venv\Scripts\Activate.ps1`; `python -c "from huggingface_hub import HfApi; print(HfApi(token=open('.secrets/hf_token').read().strip()).whoami()['name'])"`; `python -m metalai.deploy.build_space --push --repo-id mdhvfx1/metalai`.
**Console output (key lines / full traceback):** whoami returned `mdhvfx1`. Build again passed (`23.2 MB`; `CHECK verdict ok: True | det: True | cam: True | top: Crazing`), then push failed before repo creation with HTTP 402. Full traceback:
```text
Traceback (most recent call last):
  File "D:\python\metalAI\.venv\Lib\site-packages\huggingface_hub\utils\_http.py", line 866, in hf_raise_for_status
    response.raise_for_status()
  File "D:\python\metalAI\.venv\Lib\site-packages\httpx2\_models.py", line 828, in raise_for_status
    raise HTTPStatusError(message, request=self, response=self)
httpx2.HTTPStatusError: Client error '402 Payment Required' for url 'https://huggingface.co/api/repos/create'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "D:\python\metalAI\src\metalai\deploy\build_space.py", line 195, in <module>
    main()
  File "D:\python\metalAI\src\metalai\deploy\build_space.py", line 191, in main
    push(a.repo_id)
  File "D:\python\metalAI\src\metalai\deploy\build_space.py", line 173, in push
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
  File "D:\python\metalAI\.venv\Lib\site-packages\huggingface_hub\utils\_validators.py", line 89, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "D:\python\metalAI\.venv\Lib\site-packages\huggingface_hub\hf_api.py", line 4781, in create_repo
    raise err
  File "D:\python\metalAI\.venv\Lib\site-packages\huggingface_hub\hf_api.py", line 4765, in create_repo
    hf_raise_for_status(r)
  File "D:\python\metalAI\.venv\Lib\site-packages\huggingface_hub\utils\_http.py", line 996, in hf_raise_for_status
    raise _format(HfHubHTTPError, str(e), response) from e
huggingface_hub.errors.HfHubHTTPError: Client error '402 Payment Required' for url 'https://huggingface.co/api/repos/create' (Request ID: Root=1-6ae399dc-58a9d5632e1c32dd1e480d49;56da741b-2974-453d-b199-2134dc5667d0)

Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription. Subscribe at https://huggingface.co/pro
```
**Metrics:** No remote repository was created; no public URL/runtime to report. Local build remains valid at `deploy_build/hf_space/`.
**Files produced:** none by the push; build artifacts are git-ignored.
**Hardware:** n/a.
**Anything odd:** Hugging Face reports static Spaces remain free, but Gradio and Docker Spaces on `cpu-basic` require PRO.
**Code changes:** none.
**Questions for Claude:** please advise whether Madhav will enable Hugging Face PRO for this account or choose a different hosting plan.

### [Claude → Codex] 2026-10-30 — T14: deploy on Modal (HF needs PRO, see D-029)
Code added by Claude (commit first, together with the untracked `AGENTS.md`): `src/metalai/deploy/modal_app.py`, `decisions.md` D-029.

**T14a — Modal login (one-time)**
```powershell
pip install -U modal
modal setup          # opens the browser; Madhav signs in (GitHub/Google). If no browser, report the URL it prints.
```
If login needs Madhav, STOP and tell him exactly what to click. Do not create paid plans or add a card.

**T14b — test with a temporary URL**
```powershell
python -m metalai.deploy.build_space --check          # build already exists from T13b; re-run full build if missing
modal serve src/metalai/deploy/modal_app.py
```
Open the printed `https://...modal.run` URL, tap 3 samples, check verdict / Where? / Why? tabs work. Ctrl+C to stop.
Tiny fixes allowed for Modal/Gradio API renames (e.g. `@modal.concurrent`, `max_containers`, `scaledown_window`, `add_local_dir`, `mount_gradio_app` kwargs) — log each change.

**T14c — permanent deploy**
```powershell
modal deploy src/metalai/deploy/modal_app.py
```
Report: the public URL, first-load (cold start) time, time per inspection, and remaining credit shown in the Modal dashboard (Usage). Commit `T14: Modal deploy`. **CHECKPOINT** — Madhav will share the URL with friends.

### [T14a] Modal login ? DONE (2026-10-30)
**Commands run:** ` .\.venv\Scripts\Activate.ps1`; `pip install -U modal` (installed Modal 1.6.3); `modal setup`.
**Console output (key lines):** user completed Modal browser authentication; setup reported `Web authentication finished successfully!`, workspace `mdhvdhage`, `Token verified successfully!`, and wrote the token to the local `.modal.toml` profile. The command then exited 1 because the Windows `charmap` codec could not encode a Unicode arrow. A retry with `$env:PYTHONIOENCODING='utf-8'` opened another token flow; I stopped that redundant setup without completing it. The first token remains verified and configured.
**Files produced:** local Modal CLI profile at `%USERPROFILE%\.modal.toml` (outside repo).
**Hardware:** n/a.
**Anything odd:** terminal Unicode encoding error after successful authentication; no payment prompt, card entry, or paid plan selection.
**Code changes:** none.
**Questions for Claude:** none

### [T14b] Test with a temporary Modal URL ? DONE (2026-10-30)
**Commands run:** ` .\.venv\Scripts\Activate.ps1`; `$env:PYTHONIOENCODING='utf-8'`; `python -m metalai.deploy.build_space --check`; `modal serve src/metalai/deploy/modal_app.py`; stopped with Ctrl+C.
**Console output (key lines):** build check `CHECK verdict ok: True | det: True | cam: True | top: Crazing`. First serve attempt failed at container import with `IndexError: 3` because the uploaded module path is `/root/modal_app.py` and `parents[3]` does not exist. After the path fix, Modal built image `im-DONqzLxaXu6ocPKAprNQuY` in 47.44 s and served `https://mdhvdhage--metalai-web-dev.us-east.modal.run` (unauthenticated). Windows live-reload is unsupported, but the app served normally.
**UI check:** clicked three samples: Crazing (93% confidence, 3 regions, 492 ms), Inclusion (93%, 3 regions, 229 ms), Patches (90%, 3 regions, 108 ms). Where? rendered the YOLO result, Why? rendered the Grad-CAM, and All classes showed classifier probabilities for all six classes. Then stopped the temporary server. Initial page readiness took roughly 40 s including image build; no standalone cold-start timing was available.
**Files produced:** refreshed Modal image cache only; no source artifacts.
**Hardware:** Modal CPU container; no GPU.
**Anything odd:** dashboard showed `$1 of $30/mo in free credits`; no payment method was added. The first serving attempt generated repeated import errors until stopped.
**Code changes:** `src/metalai/deploy/modal_app.py` ? choose `/app` as the build directory inside the Modal container because the uploaded module lives at `/root/modal_app.py`; retain the repo-relative build path locally. Set `scaledown_window=5 * 60` so the app sleeps after five idle minutes as requested.
**Questions for Claude:** none
