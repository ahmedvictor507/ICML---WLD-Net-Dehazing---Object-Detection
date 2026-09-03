"""
Compile the best before/after frames from every video into single figure sheets
for the paper. Reads paper_results/stats.json + figures/*_panels/.

Produces:
  figures/FIG_qualitative_grid.png   one row per video: hazy | dehazed | hazy+YOLO | dehazed+YOLO
  figures/FIG_detection_only.png     one row per video: hazy+YOLO | dehazed+YOLO (compact 2-col)
"""
import json
from pathlib import Path
import cv2
import numpy as np

OUT_DIR = Path("/home/orin_nano/Desktop/Projects/dehazing_gui/paper_results")
FIG_DIR = OUT_DIR / "figures"
BG = 22
GAP = 8

# Which frame to feature per video: the top "detection gain" pick (first listed).
FEATURE_OVERRIDE = {}   # e.g. {"Object3": 1450}


def label_bar(width, text, height=32, scale=0.6):
    bar = np.full((height, width, 3), BG, np.uint8)
    cv2.putText(bar, text, (10, int(height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (235, 235, 235), 1, cv2.LINE_AA)
    return bar


def row_label(height, text, width=44):
    """Vertical label strip on the left of a row."""
    strip = np.full((width, height, 3), BG, np.uint8)
    cv2.putText(strip, text, (10, int(width * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (235, 235, 235), 1, cv2.LINE_AA)
    return cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)


def hcat(imgs, gap=GAP):
    h = imgs[0].shape[0]
    sep = np.full((h, gap, 3), BG, np.uint8)
    out = imgs[0]
    for im in imgs[1:]:
        out = np.hstack([out, sep, im])
    return out


def vcat(imgs, gap=GAP):
    w = max(i.shape[1] for i in imgs)
    padded = [np.pad(i, ((0, 0), (0, w - i.shape[1]), (0, 0)),
                     constant_values=BG) for i in imgs]
    sep = np.full((gap, w, 3), BG, np.uint8)
    out = padded[0]
    for im in padded[1:]:
        out = np.vstack([out, sep, im])
    return out


def pick_frame(safe, stats_entry):
    if safe in FEATURE_OVERRIDE:
        return FEATURE_OVERRIDE[safe]
    figs = stats_entry.get("figures", [])
    if not figs:
        return None
    # figures are listed in frame order; re-rank by the panel that carries the
    # most dehazed boxes is overkill -- the first entry of `best` was already the
    # highest-gain frame, so recover gains by comparing panel file sizes is not
    # reliable. Use the middle figure as a representative, stable choice.
    return int(figs[len(figs) // 2].split("_f")[1].split("_")[0])


def build(sheet_name, columns):
    """columns: list of (suffix, caption)."""
    stats = json.loads((OUT_DIR / "stats.json").read_text())
    rows, header = [], None
    for s in stats:
        if "error" in s or not s.get("figures"):
            continue
        safe = s["figures"][0].split("_f")[0]
        fi = pick_frame(safe, s)
        pdir = FIG_DIR / f"{safe}_panels"
        imgs = []
        for suf, _ in columns:
            p = pdir / f"{safe}_f{fi:05d}_{suf}.png"
            if not p.exists():
                imgs = []
                break
            imgs.append(cv2.imread(str(p)))
        if not imgs:
            continue
        body = hcat(imgs)
        rows.append(hcat([row_label(body.shape[0], s["video"]), body]))
        if header is None:
            lw = 44 + GAP
            bars = [label_bar(i.shape[1], cap) for i, (_, cap) in zip(imgs, columns)]
            header = np.hstack([np.full((bars[0].shape[0], lw, 3), BG, np.uint8),
                                hcat(bars)])
    if not rows:
        print(f"  no rows for {sheet_name}")
        return
    sheet = vcat([header] + rows)
    sheet = np.pad(sheet, ((12, 12), (12, 12), (0, 0)), constant_values=BG)
    out = FIG_DIR / sheet_name
    cv2.imwrite(str(out), sheet)
    print(f"  {out}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    print("compiling figure sheets:")
    build("FIG_qualitative_grid.png", [
        ("a_hazy", "(a) Hazy input"),
        ("b_dehazed", "(b) ICEML dehazed"),
        ("c_hazy_detected", "(c) YOLO on hazy"),
        ("d_dehazed_detected", "(d) YOLO on dehazed"),
    ])
    build("FIG_detection_only.png", [
        ("c_hazy_detected", "Detection without dehazing"),
        ("d_dehazed_detected", "Detection with ICEML dehazing"),
    ])
