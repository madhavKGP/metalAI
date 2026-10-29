# Instructions for coding agents (Codex, OpenCode, ...)

This repo is MetalAI (M.Tech project). Claude plans and writes code; you run things on Madhav's Windows laptop and report back.

- **Read `OPENCODE.md` first** — it is the task board + log (the name is historical; it applies to any agent). Do the task marked NEXT / the latest "[Claude → ...]" instructions at the bottom.
- Why things are done: `decisions.md`. Experiments: `experiments/EXPERIMENTS.md`.
- Shell: PowerShell, repo root `D:\python\metalAI`, venv `.\.venv\Scripts\Activate.ps1`.
- Training is GPU-only (RTX 4060 Laptop, 8 GB). One GPU job at a time.
- Set `$env:METALAI_EXP` before any training run. Never hand-edit results CSVs.
- Secrets live in `.secrets/` (git-ignored): never print, log or commit them; check `git status` before every commit.
- Tiny fixes are OK (log them under "Code changes"); bigger changes → stop and ask Claude in the log.
- After each task: append a report to the Log in `OPENCODE.md` (template in §0), update the status board, commit.
