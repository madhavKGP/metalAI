"""Convert NEU boxes to YOLO format and write one dataset YAML per fold (same folds as the CNN).

Run:  python -m metalai.m1_surface.yolo_data
Out:  data/processed/neu_yolo/images/<id>.png, labels/<id>.txt
      data/processed/neu_yolo/fold<k>_{train,val,test}.txt, fold<k>.yaml, full.yaml
"""
from __future__ import annotations

import shutil

import pandas as pd
import yaml

from metalai.common.utils import PROC, load_config
from metalai.m1_surface import CLASSES
from metalai.m1_surface.splits import get_fold

YDIR = PROC / "neu_yolo"


def main() -> None:
    cfg = load_config("m1")
    man = pd.read_csv(PROC / "neu" / "manifest.csv")
    boxes = pd.read_csv(PROC / "neu" / "boxes.csv")
    (YDIR / "images").mkdir(parents=True, exist_ok=True)
    (YDIR / "labels").mkdir(parents=True, exist_ok=True)

    n_empty = 0
    for r in man.itertuples():
        dst = YDIR / "images" / f"{r.image_id}.png"
        if not dst.exists():
            shutil.copy2(PROC / "neu" / r.path, dst)
        b = boxes[boxes.image_id == r.image_id]
        lines = []
        for bb in b.itertuples():
            cx = (bb.xmin + bb.xmax) / 2 / r.width
            cy = (bb.ymin + bb.ymax) / 2 / r.height
            w = (bb.xmax - bb.xmin) / r.width
            h = (bb.ymax - bb.ymin) / r.height
            lines.append(f"{bb.label_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        n_empty += not lines
        (YDIR / "labels" / f"{r.image_id}.txt").write_text("\n".join(lines))

    def write_list(df, name):
        p = YDIR / name
        p.write_text("\n".join(str((YDIR / "images" / f"{i}.png").resolve()) for i in df.image_id))
        return p.name

    for k in list(range(cfg["n_folds"])) + [-1]:
        tr, va, te = get_fold(k, cfg["val_frac"], cfg["seed"])
        tag = "full" if k == -1 else f"fold{k}"
        y = {"path": str(YDIR.resolve()), "train": write_list(tr, f"{tag}_train.txt"),
             "val": write_list(va, f"{tag}_val.txt"), "names": {i: c for i, c in enumerate(CLASSES)}}
        if k != -1:
            y["test"] = write_list(te, f"{tag}_test.txt")
        with open(YDIR / f"{tag}.yaml", "w") as f:
            yaml.safe_dump(y, f, sort_keys=False)
        print(f"{tag}: train {len(tr)}  val {len(va)}  test {len(te)}")
    print(f"YOLO dataset at {YDIR}  (images without boxes: {n_empty})")


if __name__ == "__main__":
    main()
