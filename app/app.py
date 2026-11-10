"""MetalAI web app (Gradio).

Local:   python app/app.py                      -> http://127.0.0.1:7860  (same Wi-Fi: http://<laptop-ip>:7860)
Public:  $env:METALAI_SHARE=1; python app/app.py  (temporary gradio.live link)
Hosted:  Hugging Face Spaces, built by `python -m metalai.deploy.build_space` (decisions.md D-028)

Rule: models see only the uploaded pixels - no filename / metadata shortcuts (decisions.md D-010).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE / "src", HERE.parent / "src"):  # works both in the repo and in the deployed Space
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gradio as gr  # noqa: E402

from metalai.m1_surface import CLASSES, DEFECT_INFO  # noqa: E402
from metalai.m1_surface.inference import UNCERTAIN_BELOW, SurfaceInspector  # noqa: E402
from metalai.m1_surface.preprocess import to_gray_uint8  # noqa: E402

INSPECTOR = SurfaceInspector()
PRETTY = {c: c.replace("_", " ").capitalize() for c in CLASSES}


def find_model_card() -> dict:
    for p in (HERE / "configs" / "model_card.json", HERE.parent / "configs" / "model_card.json"):
        if p.exists():
            return json.loads(p.read_text())
    return {}


def build_examples(per_class: int = 2) -> list[str]:
    """Demo gallery = held-out TEST images of fold 0 (never seen by the fold-0 models the app serves,
    decisions.md D-027). Copied to app/examples/ with neutral names (sample_01.png ...)."""
    out = HERE / "examples"
    existing = sorted(out.glob("sample_*.png"))
    if existing:
        return [str(p) for p in existing]
    try:
        import shutil

        import pandas as pd

        from metalai.common.utils import PROC
        from metalai.m1_surface.splits import load_folds

        man = pd.read_csv(PROC / "neu" / "manifest.csv").set_index("image_id")
        test = load_folds().query("fold == 0").groupby("label").head(per_class)
    except Exception:  # noqa: BLE001 - app must still start without data
        return []
    out.mkdir(exist_ok=True)
    key = []
    for n, r in enumerate(test.itertuples(), 1):
        dst = out / f"sample_{n:02d}.png"
        shutil.copy2(PROC / "neu" / man.loc[r.image_id, "path"], dst)
        key.append({"file": dst.name, "image_id": r.image_id, "true_label": r.label})
    pd.DataFrame(key).to_csv(out / "answer_key.csv", index=False)  # for us, never shown to the model
    return [str(p) for p in sorted(out.glob("sample_*.png"))]


EXAMPLES = build_examples()
CARD = find_model_card()


# ----------------------------------------------------------------------------- inference
# Out-of-scope guards (decisions.md D-032). No trained weights needed.
# Every NEU image has grey-level std >= 5.97 (median 23.4), so std < 3 means a blank, textureless image.
LOW_CONTRAST_STD = 3.0


def scope_flags(gray, probs: dict | None, dets: list) -> list[str]:
    flags = []
    if float(gray.std()) < LOW_CONTRAST_STD:
        flags.append("almost no surface texture: this does not look like a steel surface image")
    if probs and dets:
        top = max(probs, key=probs.get)
        if top not in {d["class"] for d in dets}:
            found = ", ".join(sorted({PRETTY[d["class"]] for d in dets}))
            flags.append(f"classifier says {PRETTY[top]} but the detector found {found}")
    return flags


def verdict_html(probs: dict | None, dets: list, ms_total: float | None, flags: list[str] | None = None) -> str:
    if not probs:
        return "<div class='mai-card'><b>Model not loaded.</b> Train / configure models first.</div>"
    top = max(probs, key=probs.get)
    p = probs[top]
    flags = flags or []
    blank = any(f.startswith("almost no surface texture") for f in flags)
    uncertain = p < UNCERTAIN_BELOW or blank
    title = "Uncertain — manual review" if uncertain else PRETTY[top]
    badge = "mai-warn" if uncertain else "mai-ok"
    n_boxes = len(dets)
    box_txt = f"{n_boxes} defect region{'s' if n_boxes != 1 else ''} located" if INSPECTOR.yolo else "detector not loaded"
    by_cls: dict[str, int] = {}
    for d in dets:
        by_cls[d["class"]] = by_cls.get(d["class"], 0) + 1
    chips = "".join(f"<span class='mai-chip'>{PRETTY[c]} × {n}</span>" for c, n in by_cls.items())
    info = DEFECT_INFO.get(top, "") if not uncertain else (
        f"Best guess: {PRETTY[top]} ({p:.0%}). The image may be outside what the model was trained on.")
    if flags:
        info += "".join(f"<br><span class='mai-warn'>Check: {f}.</span>" for f in flags)
    return f"""
<div class='mai-card'>
  <div class='mai-label'>Classifier verdict</div>
  <div class='mai-title {badge}'>{title}</div>
  <div class='mai-bar'><div style='width:{p * 100:.0f}%'></div></div>
  <div class='mai-sub'>confidence {p:.0%} · {box_txt} · {ms_total:.0f} ms</div>
  <div class='mai-chips'>{chips}</div>
  <p class='mai-info'>{info}</p>
</div>"""


def inspect(image, conf):
    if image is None:
        return "<div class='mai-card'><b>Please upload an image</b> or tap a sample below.</div>", None, None, None
    gray = to_gray_uint8(image)
    probs, cam, ms_c = INSPECTOR.classify(gray)
    det_img, dets, ms_d = INSPECTOR.detect(gray, conf)
    ms_total = (ms_c or 0) + (ms_d or 0)
    labels = {PRETTY[k]: v for k, v in probs.items()} if probs else None
    return verdict_html(probs, dets, ms_total, scope_flags(gray, probs, dets)), det_img, cam, labels


# ----------------------------------------------------------------------------- UI
CSS = """
.mai-hero {padding: 18px 20px; border-radius: 14px; border: 1px solid var(--border-color-primary);
           background: var(--background-fill-secondary);}
.mai-hero h1 {margin: 0 0 4px 0; font-size: 28px;}
.mai-hero p {margin: 0; opacity: .8;}
.mai-card {padding: 16px 18px; border-radius: 14px; border: 1px solid var(--border-color-primary);
           background: var(--background-fill-secondary);}
.mai-label {font-size: 12px; text-transform: uppercase; letter-spacing: .06em; opacity: .65;}
.mai-title {font-size: 30px; font-weight: 700; margin: 4px 0 8px 0;}
.mai-ok {color: var(--color-accent);} .mai-warn {color: #d97706;}
.mai-bar {height: 8px; border-radius: 6px; background: var(--border-color-primary); overflow: hidden;}
.mai-bar > div {height: 100%; background: var(--color-accent);}
.mai-sub {font-size: 13px; opacity: .75; margin-top: 6px;}
.mai-chips {margin-top: 8px;}
.mai-chip {display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0; border-radius: 999px;
           font-size: 12px; border: 1px solid var(--border-color-primary);}
.mai-info {margin: 10px 0 0 0; font-size: 14px;}
footer {display: none !important;}
"""


def card_md() -> str:
    if not CARD:
        return "Metrics appear here after `python -m metalai.deploy.build_space` creates `configs/model_card.json`."
    rows = "\n".join(f"| {r['method']} | {r['task']} | {r['metric']} | {r['value']} |" for r in CARD.get("rows", []))
    return (f"**Data:** {CARD.get('data', '')}\n\n| Method | Task | Metric | Result (held-out test folds) |\n"
            f"|---|---|---|---|\n{rows}\n\n_{CARD.get('note', '')}_")


with gr.Blocks(title="MetalAI") as demo:
    gr.HTML("""
<div class='mai-hero'>
  <h1>MetalAI · Surface Defect Inspector</h1>
  <p>Deep learning finds, names and explains defects on hot-rolled steel strip —
     crazing, inclusion, patches, pitted surface, rolled-in scale, scratches.</p>
</div>""")
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            inp = gr.Image(label="Steel surface image", type="numpy", height=320,
                           sources=["upload", "webcam", "clipboard"])
            with gr.Row():
                btn = gr.Button("Inspect", variant="primary", scale=3)
                clr = gr.ClearButton(value="Clear", scale=1)
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Detector sensitivity (confidence threshold)")
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=inp, label="Tap a sample — real steel images the model has never seen",
                            examples_per_page=12)
        with gr.Column(scale=6):
            verdict = gr.HTML("<div class='mai-card'><div class='mai-label'>Classifier verdict</div>"
                              "<div class='mai-title'>—</div><div class='mai-sub'>Tap a sample or upload an image."
                              "</div></div>")
            with gr.Tabs():
                with gr.Tab("Where? (detection)"):
                    out_det = gr.Image(label="Located defects (YOLO)", height=340)
                with gr.Tab("Why? (Grad-CAM)"):
                    out_cam = gr.Image(label="Where the classifier looked", height=340)
                with gr.Tab("All classes"):
                    out_lbl = gr.Label(num_top_classes=6, label="Classifier probabilities")
    with gr.Accordion("How it works & model card", open=False):
        gr.Markdown(f"""
1. **Classifier** — EfficientNet-B0 (ImageNet-pretrained, fine-tuned) names the defect type.
2. **Detector** — YOLOv8n draws a box around every defect region.
3. **Explanation** — Grad-CAM heat-map shows which pixels drove the classifier's decision.

{card_md()}
""")
    with gr.Accordion("Limitations", open=False):
        gr.Markdown("""
- Trained on the **NEU** dataset: grayscale images of hot-rolled steel strip from one production line.
  Photos of other materials, lighting or scales (e.g. phone photos of parts) are **out of scope** —
  the model may answer confidently but wrongly. Low-confidence answers are flagged as *uncertain*.
- There is no *no-defect* class: every image is assigned one of the six defect types.
- Research prototype (M.Tech project), not a certified inspection tool.
""")

    outs = [verdict, out_det, out_cam, out_lbl]
    btn.click(inspect, [inp, conf], outs)
    inp.change(inspect, [inp, conf], outs)          # auto-run when a sample is tapped / image uploaded
    conf.release(inspect, [inp, conf], outs)
    clr.add([inp, out_det, out_cam, out_lbl])


def _launch():
    kw = dict(server_name="0.0.0.0", share=os.environ.get("METALAI_SHARE") == "1")
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    try:  # Gradio >= 6 takes theme/css in launch()
        demo.launch(theme=theme, css=CSS, **kw)
    except TypeError:  # older Gradio: theme/css belong to Blocks; fall back without styling args
        demo.css = CSS
        demo.launch(**kw)


if __name__ == "__main__":
    _launch()
