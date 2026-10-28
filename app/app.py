"""MetalAI web app (Gradio).  Run:  python app/app.py   -> http://127.0.0.1:7860

Tab 1 = Module 1 (Surface Defect Inspector). Tabs 2-3 are placeholders until Modules 2-3 exist.
Rule: models see only the uploaded pixels - no filename / metadata shortcuts (decisions.md D-010).
"""
from __future__ import annotations

import gradio as gr

from metalai.m1_surface import DEFECT_INFO
from metalai.m1_surface.inference import UNCERTAIN_BELOW, SurfaceInspector
from metalai.m1_surface.preprocess import to_gray_uint8

INSPECTOR = SurfaceInspector()


def inspect(image, conf):
    if image is None:
        return None, None, None, "Upload an image first."
    gray = to_gray_uint8(image)
    probs, cam, ms_c = INSPECTOR.classify(gray)
    det_img, dets, ms_d = INSPECTOR.detect(gray, conf)
    lines = []
    if probs:
        top = max(probs, key=probs.get)
        if probs[top] < UNCERTAIN_BELOW:
            lines.append(f"**Classifier:** uncertain (top = {top}, {probs[top]:.2f}) - manual review recommended.")
        else:
            lines.append(f"**Classifier:** {top} ({probs[top]:.2f})")
            lines.append(f"> Reference: {DEFECT_INFO[top]}")
        lines.append(f"CNN + Grad-CAM time: {ms_c:.0f} ms ({INSPECTOR.describe(INSPECTOR.cnn_path)})")
    else:
        lines.append("CNN model not found - train it first (OPENCODE.md).")
    if INSPECTOR.yolo is not None:
        lines.append(f"**Detector:** {len(dets)} box(es) at conf >= {conf:.2f}; time {ms_d:.0f} ms ({INSPECTOR.describe(INSPECTOR.yolo_path)})")
        for d in dets:
            lines.append(f"- {d['class']} {d['conf']:.2f} at {d['xyxy']}")
    else:
        lines.append("YOLO model not found - train it first (OPENCODE.md).")
    return det_img, probs, cam, "\n".join(lines)


with gr.Blocks(title="MetalAI") as demo:
    gr.Markdown("# MetalAI\nAI toolkit for metallurgy - M.Tech project")
    with gr.Tab("1 - Surface Inspector"):
        gr.Markdown("Upload a grayscale image of a hot-rolled steel strip surface (NEU-style, ~200x200).")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(label="Input", type="numpy")
                conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Detector confidence threshold")
                btn = gr.Button("Inspect", variant="primary")
            with gr.Column():
                out_det = gr.Image(label="Detections (YOLO)")
                out_cam = gr.Image(label="Grad-CAM (CNN top class)")
        out_lbl = gr.Label(num_top_classes=3, label="Classifier probabilities (CNN)")
        out_txt = gr.Markdown()
        btn.click(inspect, [inp, conf], [out_det, out_lbl, out_cam, out_txt])
        gr.Markdown("**Model card:** trained on NEU surface defect database (6 classes, 1,800 images). "
                    "Not validated on other mills/cameras - see reports/m1/REPORT.md for metrics and limits.")
    with gr.Tab("2 - Steel Designer"):
        gr.Markdown("Coming in Module 2 (property prediction + inverse design).")
    with gr.Tab("3 - Alloy Assistant"):
        gr.Markdown("Coming in Module 3 (HEA phase prediction + LLM agent).")

if __name__ == "__main__":
    demo.launch()
