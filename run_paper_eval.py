"""
Batch dehazing + object-detection evaluation for the research paper.

For each input video, produces:
  <name>_original_detected.mp4   YOLO on the raw (hazy) frame
  <name>_dehazed.mp4             ICEML dehazed frame, no boxes
  <name>_dehazed_detected.mp4    YOLO on the ICEML dehazed frame
  figures/<name>_cmp_<frame>.png 2x2 before/after comparison plates
  figures/<name>_panels/*.png    the same panels as separate images
  stats.json / stats.md          per-video and per-class detection tables

Everything runs at RES x RES (matches the deployed GUI's square working resolution).
"""
import os, sys, json, time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch

BASE_DIR = Path("/home/orin_nano/Desktop/Projects/dehazing_gui")
BACKUP   = Path("/home/orin_nano/Desktop/Projects/real-time-wld-net-backup")
sys.path.insert(0, str(BASE_DIR / "iceml_src"))

ICEML_WEIGHTS = BASE_DIR / "models" / "iceml" / "final_dehazing_model.pth"
YOLO_WEIGHTS  = BACKUP / "detection_model" / "fyp_test5_best.pt"
VIDEO_DIR     = BACKUP / "videos"
OUT_DIR       = BASE_DIR / "paper_results"
FIG_DIR       = OUT_DIR / "figures"

VIDEOS = ["human(2)_test(1).mp4", "Human (3).mp4", "Object(3).mp4", "Object(6).mp4"]

RES        = 512      # square working resolution, same as the GUI
CONF       = 0.25     # ultralytics reporting default
TOP_K      = 6        # best "detection gain" frames kept per video
EVEN_SHOTS = 4        # evenly spaced frames kept per video regardless of gain

CYAN = (255, 212, 0)   # BGR


# ── models ───────────────────────────────────────────────────────────────────
def load_iceml(device):
    from retinexmodule import dehazing_module as ICEMLModule
    m = ICEMLModule(3)
    full_sd = torch.load(ICEML_WEIGHTS, map_location="cpu", weights_only=False)
    prefix = "retinex_model."
    m.load_state_dict({k[len(prefix):]: v for k, v in full_sd.items()
                       if k.startswith(prefix)})
    m.to(device).eval()
    if device.type == "cuda":
        m.half()
    return m


def dehaze(model, frame_bgr, device):
    """frame_bgr is already RES x RES. Returns dehazed BGR uint8."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
    if device.type == "cuda":
        t = t.half()
    t = t.to(device)
    with torch.no_grad():
        lr_inv  = model(t)
        dehazed = (t.float() * lr_inv.float()).clamp(0, 1)
    out = (dehazed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def detect(yolo, frame_bgr, conf=CONF):
    """Returns (annotated BGR copy, list of (label, score))."""
    res = yolo(frame_bgr, conf=conf, verbose=False, imgsz=RES)[0]
    ann = frame_bgr.copy()
    dets = []
    for box, score, cls in zip(res.boxes.xyxy.cpu().numpy(),
                               res.boxes.conf.cpu().numpy(),
                               res.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = box.astype(int)
        label = yolo.names[int(cls)]
        dets.append((label, float(score)))
        text = f"{label} {score:.2f}"
        cv2.rectangle(ann, (x1, y1), (x2, y2), CYAN, 2)
        cv2.rectangle(ann, (x1, y1), (x1 + len(text) * 9 + 6, y1 + 18), CYAN, -1)
        cv2.putText(ann, text, (x1 + 3, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return ann, dets


# ── figure helpers ───────────────────────────────────────────────────────────
def caption(img, text):
    """Add a caption bar under a panel."""
    h, w = img.shape[:2]
    bar = np.full((34, w, 3), 22, np.uint8)
    cv2.putText(bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def plate(panels, gap=8):
    """2x2 grid from four captioned panels."""
    h, w = panels[0].shape[:2]
    sep_v = np.full((h, gap, 3), 22, np.uint8)
    rows = [np.hstack([panels[0], sep_v, panels[1]]),
            np.hstack([panels[2], sep_v, panels[3]])]
    sep_h = np.full((gap, rows[0].shape[1], 3), 22, np.uint8)
    return np.vstack([rows[0], sep_h, rows[1]])


# ── per-video run ────────────────────────────────────────────────────────────
def process(video_name, yolo, iceml, device):
    src = VIDEO_DIR / video_name
    stem = Path(video_name).stem
    safe = stem.replace(" ", "_").replace("(", "").replace(")", "")
    print(f"\n=== {video_name} ===", flush=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_wh = (int(cap.get(3)), int(cap.get(4)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_orig = cv2.VideoWriter(str(OUT_DIR / f"{safe}_original_detected.mp4"), fourcc, fps, (RES, RES))
    vw_dhz  = cv2.VideoWriter(str(OUT_DIR / f"{safe}_dehazed.mp4"),           fourcc, fps, (RES, RES))
    vw_dhzd = cv2.VideoWriter(str(OUT_DIR / f"{safe}_dehazed_detected.mp4"),  fourcc, fps, (RES, RES))

    st = {
        "video": video_name, "source_resolution": f"{src_wh[0]}x{src_wh[1]}",
        "processed_resolution": f"{RES}x{RES}", "fps": round(fps, 2),
        "frames": 0, "conf_threshold": CONF,
        "original": {"total_det": 0, "frames_with_det": 0,
                     "per_class": defaultdict(int), "conf_sum": defaultdict(float)},
        "dehazed":  {"total_det": 0, "frames_with_det": 0,
                     "per_class": defaultdict(int), "conf_sum": defaultdict(float)},
    }
    t_dehaze = t_det = 0.0

    even_idx = set(np.linspace(0, max(total - 1, 0), EVEN_SHOTS + 2)[1:-1].astype(int).tolist()) \
               if total > 0 else set()
    best = []   # (gain, dehazed_conf_sum, idx, orig, dhz, orig_ann, dhz_ann)
    forced = []

    idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (RES, RES), interpolation=cv2.INTER_AREA)

        ta = time.time()
        dhz = dehaze(iceml, small, device)
        t_dehaze += time.time() - ta

        tb = time.time()
        ann_o, det_o = detect(yolo, small)
        ann_d, det_d = detect(yolo, dhz)
        t_det += time.time() - tb

        for key, dets in (("original", det_o), ("dehazed", det_d)):
            b = st[key]
            b["total_det"] += len(dets)
            b["frames_with_det"] += 1 if dets else 0
            for lbl, sc in dets:
                b["per_class"][lbl] += 1
                b["conf_sum"][lbl] += sc

        vw_orig.write(ann_o)
        vw_dhz.write(dhz)
        vw_dhzd.write(ann_d)

        gain = len(det_d) - len(det_o)
        conf_d = sum(s for _, s in det_d)
        if len(det_d) > 0:
            best.append((gain, conf_d, idx, small.copy(), dhz.copy(), ann_o.copy(), ann_d.copy()))
            best.sort(key=lambda r: (r[0], r[1]), reverse=True)
            best = best[:TOP_K]
        if idx in even_idx:
            forced.append((gain, conf_d, idx, small.copy(), dhz.copy(), ann_o.copy(), ann_d.copy()))

        st["frames"] += 1
        idx += 1
        if idx % 200 == 0:
            el = time.time() - t0
            print(f"  {idx}/{total} frames  ({idx/el:.1f} fps end-to-end)", flush=True)

    cap.release()
    for w in (vw_orig, vw_dhz, vw_dhzd):
        w.release()

    n = max(st["frames"], 1)
    st["timing_ms_per_frame"] = {
        "dehaze": round(t_dehaze / n * 1000, 2),
        "detect_two_passes": round(t_det / n * 1000, 2),
        "end_to_end": round((time.time() - t0) / n * 1000, 2),
    }
    for key in ("original", "dehazed"):
        b = st[key]
        b["mean_conf_per_class"] = {k: round(b["conf_sum"][k] / b["per_class"][k], 3)
                                    for k in b["per_class"]}
        b["mean_det_per_frame"] = round(b["total_det"] / n, 3)
        b["detection_rate"] = round(b["frames_with_det"] / n, 3)
        allc = sum(b["conf_sum"].values())
        b["mean_conf_overall"] = round(allc / b["total_det"], 3) if b["total_det"] else 0.0
        b["per_class"] = dict(b["per_class"])
        b.pop("conf_sum")

    # ── figures ──────────────────────────────────────────────────────────────
    picks, seen = [], set()
    for rec in best + forced:
        if rec[2] not in seen:
            seen.add(rec[2]); picks.append(rec)
    picks.sort(key=lambda r: r[2])

    pdir = FIG_DIR / f"{safe}_panels"
    pdir.mkdir(parents=True, exist_ok=True)
    for gain, _, fi, orig, dh, ao, ad in picks:
        tag = f"{safe}_f{fi:05d}"
        cv2.imwrite(str(pdir / f"{tag}_a_hazy.png"), orig)
        cv2.imwrite(str(pdir / f"{tag}_b_dehazed.png"), dh)
        cv2.imwrite(str(pdir / f"{tag}_c_hazy_detected.png"), ao)
        cv2.imwrite(str(pdir / f"{tag}_d_dehazed_detected.png"), ad)
        fig = plate([
            caption(orig, "(a) Hazy input"),
            caption(dh,   "(b) ICEML dehazed"),
            caption(ao,   "(c) YOLO on hazy input"),
            caption(ad,   "(d) YOLO on dehazed"),
        ])
        cv2.imwrite(str(FIG_DIR / f"{tag}_cmp.png"), fig)
    st["figures"] = [f"{safe}_f{r[2]:05d}_cmp.png" for r in picks]
    print(f"  done: {st['frames']} frames, "
          f"det {st['original']['total_det']} -> {st['dehazed']['total_det']}", flush=True)
    return st


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_WEIGHTS))
    iceml = load_iceml(device)
    print(f"classes: {yolo.names}", flush=True)

    all_stats = []
    for v in VIDEOS:
        try:
            all_stats.append(process(v, yolo, iceml, device))
        except Exception as e:
            print(f"  FAILED {v}: {type(e).__name__}: {e}", flush=True)
            all_stats.append({"video": v, "error": f"{type(e).__name__}: {e}"})
        (OUT_DIR / "stats.json").write_text(json.dumps(all_stats, indent=2))

    write_report(all_stats)
    print("\nAll done ->", OUT_DIR, flush=True)


def write_report(all_stats):
    L = ["# ICEML Dehazing + YOLO Detection — Results", "",
         f"- Detector: `fyp_test5_best.pt` (classes: car, exit, person, warning)",
         f"- Dehazer: ICEML / RtxICE-Net (`final_dehazing_model.pth`)",
         f"- Working resolution: {RES}x{RES}, confidence threshold: {CONF}", "",
         "## Summary", "",
         "| Video | Frames | Det. (hazy) | Det. (dehazed) | Δ | Det/frame hazy | Det/frame dehazed | Mean conf hazy | Mean conf dehazed |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in all_stats:
        if "error" in s:
            L.append(f"| {s['video']} | — | — | — | — | — | — | — | — |")
            continue
        o, d = s["original"], s["dehazed"]
        delta = d["total_det"] - o["total_det"]
        pct = f"{delta:+d} ({delta/o['total_det']*100:+.1f}%)" if o["total_det"] else f"{delta:+d}"
        L.append(f"| {s['video']} | {s['frames']} | {o['total_det']} | {d['total_det']} | {pct} | "
                 f"{o['mean_det_per_frame']} | {d['mean_det_per_frame']} | "
                 f"{o['mean_conf_overall']} | {d['mean_conf_overall']} |")

    L += ["", "## Per-class detections", "",
          "| Video | Class | Hazy count | Dehazed count | Hazy mean conf | Dehazed mean conf |",
          "|---|---|---|---|---|---|"]
    for s in all_stats:
        if "error" in s:
            continue
        o, d = s["original"], s["dehazed"]
        for c in sorted(set(o["per_class"]) | set(d["per_class"])):
            L.append(f"| {s['video']} | {c} | {o['per_class'].get(c,0)} | {d['per_class'].get(c,0)} | "
                     f"{o['mean_conf_per_class'].get(c,'—')} | {d['mean_conf_per_class'].get(c,'—')} |")

    L += ["", "## Timing (ms/frame, Jetson Orin Nano, PyTorch FP16)", "",
          "| Video | Dehaze | Detection (2 passes) | End-to-end |", "|---|---|---|---|"]
    for s in all_stats:
        if "error" in s:
            continue
        t = s["timing_ms_per_frame"]
        L.append(f"| {s['video']} | {t['dehaze']} | {t['detect_two_passes']} | {t['end_to_end']} |")

    (OUT_DIR / "stats.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
