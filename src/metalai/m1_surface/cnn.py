"""CNN classifier (ResNet-50 / EfficientNet-B0, ImageNet transfer learning). GPU only.

Run (one fold first, then all):
  python -m metalai.m1_surface.cnn --model resnet50 --folds 0
  python -m metalai.m1_surface.cnn --model resnet50 --folds 1 2 3 4 --skip-existing
  python -m metalai.m1_surface.cnn --model resnet50 --full          # final model for the app
Set the experiment id first (experiments/EXPERIMENTS.md):  $env:METALAI_EXP = "E002"
Out (everything keyed by experiment id, so reruns never overwrite each other):
  models/m1/cnn/<exp>/<model>_fold<k>.pt   (and <model>_full.pt)
  results/m1/cnn_results.csv                    one row per exp x model x fold
  results/m1/cnn_predictions/<exp>_<model>.csv
  reports/m1/cnn/<exp>_<model>/fold<k>_curves.png, fold<k>_confusion.png, confusion_all.png
"""
from __future__ import annotations

import argparse
import math
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from metalai.common.metrics import cls_metrics, confusion, plot_confusion
from metalai.common.utils import (MODELS, PROC, REPORTS, RESULTS, append_result, current_exp, load_config,
                                  require_cuda, save_json, seed_everything, start_experiment)
from metalai.m1_surface import CLASSES
from metalai.m1_surface.preprocess import build_transform
from metalai.m1_surface.splits import get_fold

MODEL_DIR = MODELS / "m1" / "cnn"


def weights_path(exp: str, name: str, tag: str):
    return MODEL_DIR / exp / f"{name}_{tag}.pt"


def report_dir(exp: str, name: str):
    return REPORTS / "m1" / "cnn" / f"{exp}_{name}"


def preds_path(exp: str, name: str):
    return RESULTS / "m1" / "cnn_predictions" / f"{exp}_{name}.csv"


# ---------------------------------------------------------------- data
class NEUImages(Dataset):
    def __init__(self, df: pd.DataFrame, cache: dict, transform):
        self.ids = df.image_id.tolist()
        self.y = df.label_idx.to_numpy()
        self.cache = cache
        self.t = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.t(self.cache[self.ids[i]]), int(self.y[i])


def load_cache() -> dict:
    man = pd.read_csv(PROC / "neu" / "manifest.csv")
    return {r.image_id: cv2.imread(str(PROC / "neu" / r.path), cv2.IMREAD_GRAYSCALE) for r in man.itertuples()}


# ---------------------------------------------------------------- model
def build_model(name: str, n_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    from torchvision import models as tvm

    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, n_classes))
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    else:
        raise ValueError(f"unknown model {name}")
    return m


def gradcam_target_layer(model: nn.Module, name: str):
    return model.layer4[-1] if name == "resnet50" else model.features[-1]


def load_trained(name: str, tag: str, device: str = "cpu", exp: str | None = None) -> nn.Module:
    m = build_model(name, pretrained=False)
    ckpt = torch.load(weights_path(exp or current_exp(), name, tag), map_location=device)
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device).eval()


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys, loss_sum, n = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            logits = model(x)
        loss_sum += ce(logits.float(), y).item()
        n += len(y)
        probs.append(torch.softmax(logits.float(), 1).cpu())
        ys.append(y.cpu())
    return torch.cat(probs).numpy(), torch.cat(ys).numpy(), loss_sum / max(n, 1)


def time_inference(model, ds, device, n: int = 50) -> float:
    model.eval()
    xs = [ds[i][0].unsqueeze(0).to(device) for i in range(min(n, len(ds)))]
    with torch.no_grad():
        for x in xs[:5]:
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for x in xs:
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / len(xs) * 1000


def plot_curves(hist: pd.DataFrame, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.epoch, hist.train_loss, label="train")
    ax[0].plot(hist.epoch, hist.val_loss, label="val")
    ax[0].set_title("loss")
    ax[0].legend()
    ax[1].plot(hist.epoch, hist.train_acc, label="train")
    ax[1].plot(hist.epoch, hist.val_acc, label="val")
    ax[1].set_title("accuracy")
    ax[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_fold(name: str, fold: int, cfg: dict, cfg_all: dict, cache: dict, device: str) -> dict:
    seed_everything(cfg_all["seed"] + max(fold, 0))
    tag = "full" if fold == -1 else f"fold{fold}"
    tr, va, te = get_fold(fold, cfg_all["val_frac"], cfg_all["seed"])
    tf_train, tf_eval = build_transform(True, cfg["img_size"]), build_transform(False, cfg["img_size"])
    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["workers"], pin_memory=device == "cuda")
    dl_tr = DataLoader(NEUImages(tr, cache, tf_train), shuffle=True, drop_last=True, **kw)
    dl_va = DataLoader(NEUImages(va, cache, tf_eval), shuffle=False, **kw)

    model = build_model(name).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * len(dl_tr)
    warm = cfg["warmup_epochs"] * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best, best_state, bad, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            tn += len(y)
        pv, yv, vl = predict(model, dl_va, device)
        va_acc = float((pv.argmax(1) == yv).mean())
        hist.append({"epoch": ep, "train_loss": tl / tn, "train_acc": tc / tn, "val_loss": vl, "val_acc": va_acc,
                     "lr": opt.param_groups[0]["lr"]})
        print(f"[{name} {tag}] ep {ep:02d} train_loss {tl / tn:.4f} acc {tc / tn:.4f} | "
              f"val_loss {vl:.4f} acc {va_acc:.4f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ep = ep
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"Early stop at epoch {ep} (best epoch {best_ep})")
                break
    train_min = (time.time() - t0) / 60
    model.load_state_dict(best_state)
    exp = current_exp()
    wp = weights_path(exp, name, tag)
    wp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "model": name, "classes": CLASSES, "fold": fold, "exp_id": exp,
                "img_size": cfg["img_size"], "best_epoch": best_ep}, wp)

    rep = report_dir(exp, name)
    hist_df = pd.DataFrame(hist)
    rep.mkdir(parents=True, exist_ok=True)
    hist_df.to_csv(rep / f"{tag}_history.csv", index=False)
    plot_curves(hist_df, rep / f"{tag}_curves.png", f"{name} {tag}")
    row = {"model": name, "fold": fold, "epochs_run": len(hist), "best_epoch": best_ep,
           "train_minutes": round(train_min, 2), "batch_size": cfg["batch_size"], "img_size": cfg["img_size"],
           "lr": cfg["lr"], "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device == "cuda" else 0}
    if fold == -1:
        return row

    ds_te = NEUImages(te, cache, tf_eval)
    pt, yt, tlss = predict(model, DataLoader(ds_te, shuffle=False, **kw), device)
    yp = pt.argmax(1)
    m = cls_metrics(yt, yp, CLASSES)
    cm = confusion(yt, yp, len(CLASSES))
    plot_confusion(cm, CLASSES, rep / f"{tag}_confusion.png", f"{name} {tag} test")
    row.update({"test_loss": round(tlss, 4), "ms_per_img_bs1": round(time_inference(model, ds_te, device), 2), **m})
    preds = pd.DataFrame({"image_id": te.image_id, "fold": fold, "true": [CLASSES[i] for i in yt],
                          "pred": [CLASSES[i] for i in yp], "conf": pt.max(1)})
    for i, c in enumerate(CLASSES):
        preds[f"p_{c}"] = pt[:, i]
    return row, preds, cm


def main() -> None:
    cfg_all = load_config("m1")
    cfg = dict(cfg_all["cnn"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--full", action="store_true", help="train final model on all data (for the app)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="tiny smoke tests only")
    a = ap.parse_args()
    for k in ["epochs", "batch_size", "lr", "workers"]:
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)

    device = require_cuda(a.allow_cpu)
    exp = start_experiment(f"m1_cnn_{a.model}", a, {**cfg_all, "cnn": cfg})
    torch.backends.cudnn.benchmark = True
    cache = load_cache()
    print(f"Cached {len(cache)} images in RAM | device={device} | cfg={cfg}")

    if a.full:
        row = run_fold(a.model, -1, cfg, cfg_all, cache, device)
        append_result(RESULTS / "m1" / "cnn_full_models.csv", row)
        print(f"Saved {weights_path(exp, a.model, 'full')}")
        return

    pred_path = preds_path(exp, a.model)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    all_preds = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    for k in a.folds:
        if a.skip_existing and weights_path(exp, a.model, f"fold{k}").exists():
            print(f"skip fold {k} (exists)")
            continue
        row, preds, _ = run_fold(a.model, k, cfg, cfg_all, cache, device)
        append_result(RESULTS / "m1" / "cnn_results.csv", row)
        if len(all_preds):
            all_preds = all_preds[~((all_preds.fold == k))]
        all_preds = pd.concat([all_preds, preds], ignore_index=True)
        all_preds.to_csv(pred_path, index=False)
        print(f"==> {a.model} fold {k}: acc={row['accuracy']:.4f} macroF1={row['macro_f1']:.4f} "
              f"({row['train_minutes']} min, peak VRAM {row['peak_vram_gb']} GB)")

    # confusion summed over every fold available so far
    if len(all_preds):
        idx = {c: i for i, c in enumerate(CLASSES)}
        cm = confusion(all_preds.true.map(idx), all_preds.pred.map(idx), len(CLASSES))
        folds_done = sorted(all_preds.fold.unique().tolist())
        plot_confusion(cm, CLASSES, report_dir(exp, a.model) / "confusion_all.png",
                       f"{exp} {a.model} - folds {folds_done} summed")
        save_json({"exp_id": exp, "folds_done": folds_done, "confusion_summed": cm},
                  report_dir(exp, a.model) / "summary.json")


if __name__ == "__main__":
    main()
