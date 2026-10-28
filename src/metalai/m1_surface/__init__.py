"""Module 1 - Surface Defect Inspector (NEU steel surface defects)."""
import re

# Fixed class order. NEVER reorder: model heads, YOLO labels and splits depend on it.
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Aliases seen in NEU-CLS (Cr_1.bmp) and NEU-DET (crazing_1.jpg) naming / XML <name> tags.
_ALIASES = {
    "cr": "crazing", "crazing": "crazing",
    "in": "inclusion", "inclusion": "inclusion",
    "pa": "patches", "patches": "patches", "patch": "patches",
    "ps": "pitted_surface", "pittedsurface": "pitted_surface", "pitted": "pitted_surface",
    "rs": "rolled-in_scale", "rolledinscale": "rolled-in_scale", "rolledscale": "rolled-in_scale",
    "sc": "scratches", "scratches": "scratches", "scratch": "scratches",
}


def normalize_class(name: str) -> str | None:
    """Map any NEU naming variant to a canonical class name (or None)."""
    key = re.sub(r"[^a-z]", "", str(name).lower())
    return _ALIASES.get(key)


def parse_stem(stem: str) -> tuple[str | None, int | None]:
    """'rolled-in_scale_12' -> ('rolled-in_scale', 12); 'Cr_5' -> ('crazing', 5)."""
    m = re.match(r"^(.*?)[_\-\s]*(\d+)$", stem.strip())
    if not m:
        return normalize_class(stem), None
    return normalize_class(m.group(1)), int(m.group(2))


# Short reference text shown in the app (clearly labelled as reference, not model output).
DEFECT_INFO = {
    "crazing": "Network of fine surface cracks, typically from thermal or mechanical stress during rolling/cooling.",
    "inclusion": "Non-metallic particles (oxides, sulphides, slag) trapped in the steel surface; act as crack initiation sites.",
    "patches": "Irregular surface regions (scale/oxidation patches) with different reflectance from the matrix.",
    "pitted_surface": "Many small pits/holes, often from corrosion or scale detachment.",
    "rolled-in_scale": "Oxide scale pressed into the strip surface during hot rolling.",
    "scratches": "Long thin linear marks from handling, guides or rollers.",
}
