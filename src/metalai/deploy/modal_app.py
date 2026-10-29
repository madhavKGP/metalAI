"""Host the built MetalAI Space on Modal (free Starter plan: $30/month compute credits) - decisions.md D-029.

Prereq:  python -m metalai.deploy.build_space        (creates deploy_build/hf_space and checks it)
Login:   pip install modal ; modal setup                (one-time browser login)
Test:    modal serve src/metalai/deploy/modal_app.py    (temporary URL, live while the command runs)
Deploy:  modal deploy src/metalai/deploy/modal_app.py   (permanent URL until you `modal app stop metalai`)

CPU only, scales to zero when idle (cold start ~20-40 s for the first visitor), one container
(Gradio needs sticky sessions), many concurrent users inside it.
"""
from __future__ import annotations

import pathlib
import re

import modal

BUILD = pathlib.Path(__file__).resolve().parents[3] / "deploy_build" / "hf_space"

reqs: list[str] = []
gradio_req = "gradio"
if BUILD.exists():  # only true on the deploying machine; inside the container the image is already built
    reqs = [l.strip() for l in (BUILD / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.startswith("--")]
    m = re.search(r"sdk_version:\s*([\d.]+)", (BUILD / "README.md").read_text(encoding="utf-8"))
    if m:
        gradio_req = f"gradio=={m.group(1)}"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(*reqs, gradio_req, "fastapi[standard]", extra_index_url="https://download.pytorch.org/whl/cpu")
)
if BUILD.exists():
    image = image.add_local_dir(str(BUILD), "/app")

app = modal.App("metalai", image=image)


@app.function(max_containers=1, scaledown_window=15 * 60, timeout=10 * 60)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    import os
    import sys

    sys.path[:0] = ["/app/src", "/app"]
    os.chdir("/app")

    import gradio as gr
    from fastapi import FastAPI

    import app as metal  # /app/app.py (the MetalAI Gradio UI)

    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    try:  # Gradio >= 6: theme/css passed when serving
        return gr.mount_gradio_app(FastAPI(), metal.demo, path="/", theme=theme, css=metal.CSS)
    except TypeError:
        metal.demo.css = metal.CSS
        return gr.mount_gradio_app(FastAPI(), metal.demo, path="/")
