"""Build a clean, de-duplicated NEU manifest from whatever folder layout was downloaded.

Works for NEU-DET (images + Pascal VOC XML) and NEU-CLS (.bmp, names like Cr_1.bmp),
in any sub-folder structure (e.g. Kaggle 'train/images/crazing/crazing_1.jpg').

Labels come from the file name prefix (and folder name as a cross-check). This is the
dataset's own labelling convention - it is used ONLY to build ground truth, never at
inference time (see decisions.md D-003).

Run:  python -m metalai.m1_surface.prepare_data [--raw data/raw/neu]
Out:  data/processed/neu/images/<image_id>.png
      data/processed/neu/manifest.csv, boxes.csv
      reports/m1/data_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd

from metalai.common.utils import PROC, RAW, REPORTS, save_json
from metalai.m1_surface import CLASSES, normalize_class, parse_stem

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def folder_class(path: Path) -> str | None:
    for part in reversed(path.parts[:-1]):
        c = normalize_class(part)
        if c:
            return c
    return None


def parse_xml(path: Path) -> dict:
    root = ET.parse(path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width"))) if size is not None else None
    h = int(float(size.findtext("height"))) if size is not None else None
    objs = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        objs.append({
            "label_raw": obj.findtext("name"),
            "label": normalize_class(obj.findtext("name")),
            "xmin": float(bb.findtext("xmin")), "ymin": float(bb.findtext("ymin")),
            "xmax": float(bb.findtext("xmax")), "ymax": float(bb.findtext("ymax")),
        })
    fname = root.findtext("filename") or path.stem
    return {"filename": Path(fname).stem, "width": w, "height": h, "objects": objs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW / "neu")
    ap.add_argument("--out", type=Path, default=PROC / "neu")
    args = ap.parse_args()

    if not args.raw.exists():
        raise SystemExit(f"Raw folder not found: {args.raw}")
    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in args.raw.rglob("*") if p.is_file()]
    images = [p for p in files if p.suffix.lower() in IMG_EXT]
    xmls = [p for p in files if p.suffix.lower() == ".xml"]
    print(f"Found {len(images)} image files, {len(xmls)} XML files under {args.raw}")

    summary = {"raw_dir": str(args.raw), "n_image_files": len(images), "n_xml_files": len(xmls),
               "unparsed_images": [], "folder_name_mismatch": [], "duplicates_dropped": 0,
               "xml_label_mismatch": [], "secondary_defect_boxes": [], "xml_without_image": [],
               "boxes_clipped": 0}

    # ---- images -> manifest (dedupe by (class, number) and by content hash) ----
    records: dict[str, dict] = {}
    seen_hash: dict[str, str] = {}
    for p in sorted(images):
        label, num = parse_stem(p.stem)
        fcls = folder_class(p)
        if label is None:
            label = fcls
        if label is None or num is None:
            summary["unparsed_images"].append(str(p.relative_to(args.raw)))
            continue
        if fcls and fcls != label:
            summary["folder_name_mismatch"].append(str(p.relative_to(args.raw)))
        image_id = f"{label}_{num}"
        h = md5(p)
        if image_id in records or h in seen_hash:
            summary["duplicates_dropped"] += 1
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            summary["unparsed_images"].append(str(p.relative_to(args.raw)) + " (unreadable)")
            continue
        out_path = img_dir / f"{image_id}.png"
        cv2.imwrite(str(out_path), img)
        seen_hash[h] = image_id
        records[image_id] = {"image_id": image_id, "label": label, "label_idx": CLASSES.index(label),
                             "number": num, "path": str(out_path.relative_to(args.out)),
                             "height": img.shape[0], "width": img.shape[1], "md5": h,
                             "source": str(p.relative_to(args.raw))}

    # ---- XML -> boxes ----
    boxes = []
    xml_done = set()
    for x in sorted(xmls):
        try:
            info = parse_xml(x)
        except Exception as e:  # noqa: BLE001
            summary["unparsed_images"].append(f"{x.name} (xml error {e})")
            continue
        label, num = parse_stem(info["filename"])
        if label is None or num is None:
            label, num = parse_stem(x.stem)
        image_id = f"{label}_{num}"
        if image_id in xml_done:
            continue
        if image_id not in records:
            summary["xml_without_image"].append(x.name)
            continue
        xml_done.add(image_id)
        rec = records[image_id]
        W, H = rec["width"], rec["height"]
        for o in info["objects"]:
            # NEU-DET images can contain SECONDARY defects of another class (e.g. inclusions inside a
            # 'patches' image). Box class = XML <name> (decisions.md D-020); image class stays the file class.
            box_label = o["label"] or rec["label"]
            if o["label"] is None:
                summary["xml_label_mismatch"].append(f"{x.name}: unparseable xml name '{o['label_raw']}'")
            elif o["label"] != rec["label"]:
                summary["secondary_defect_boxes"].append(f"{x.name}: box={o['label']} image={rec['label']}")
            xmin, ymin = max(0.0, o["xmin"]), max(0.0, o["ymin"])
            xmax, ymax = min(float(W), o["xmax"]), min(float(H), o["ymax"])
            if (xmin, ymin, xmax, ymax) != (o["xmin"], o["ymin"], o["xmax"], o["ymax"]):
                summary["boxes_clipped"] += 1
            if xmax - xmin < 1 or ymax - ymin < 1:
                continue
            boxes.append({"image_id": image_id, "label": box_label, "label_idx": CLASSES.index(box_label),
                          "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})

    man = pd.DataFrame(sorted(records.values(), key=lambda r: (r["label_idx"], r["number"])))
    bx = pd.DataFrame(boxes, columns=["image_id", "label", "label_idx", "xmin", "ymin", "xmax", "ymax"])
    nbox = bx.groupby("image_id").size() if len(bx) else pd.Series(dtype=int)
    man["n_boxes"] = man["image_id"].map(nbox).fillna(0).astype(int)
    man.to_csv(args.out / "manifest.csv", index=False)
    bx.to_csv(args.out / "boxes.csv", index=False)

    summary.update({
        "n_images_kept": len(man),
        "images_per_class": man["label"].value_counts().reindex(CLASSES, fill_value=0).to_dict(),
        "images_with_boxes": int((man["n_boxes"] > 0).sum()),
        "boxes_per_class": bx["label"].value_counts().reindex(CLASSES, fill_value=0).to_dict() if len(bx) else {},
        "image_sizes": {f"{w}x{h}": int(c) for (w, h), c in Counter(zip(man.width, man.height)).items()},
    })
    summary["images_with_secondary_defects"] = int(bx[bx.label_idx != bx.image_id.map(
        man.set_index("image_id").label_idx)].image_id.nunique()) if len(bx) else 0
    for k in ["unparsed_images", "folder_name_mismatch", "xml_label_mismatch", "secondary_defect_boxes",
              "xml_without_image"]:
        summary[f"n_{k}"] = len(summary[k])
        summary[k] = summary[k][:30]  # keep the report short
    save_json(summary, REPORTS / "m1" / "data_summary.json")

    print("\n=== DATA SUMMARY ===")
    for k, v in summary.items():
        if not isinstance(v, list):
            print(f"{k}: {v}")
    print(f"\nManifest: {args.out / 'manifest.csv'}  ({len(man)} images)")
    print(f"Boxes:    {args.out / 'boxes.csv'}  ({len(bx)} boxes)")
    expected = 300
    bad = {c: n for c, n in summary["images_per_class"].items() if n != expected}
    if bad:
        print(f"[WARN] classes without exactly {expected} images: {bad}  -> report this to Claude")


if __name__ == "__main__":
    main()
