"""
Clean deployment-latency benchmark for the ICEML + YOLO pipeline.

Measures the DEPLOYED path only: decode -> preprocess -> dehaze -> detect.
No video writing, no display, no second detection pass.

  python benchmark_deployment.py --backend trt      # TensorRT FP16
  python benchmark_deployment.py --backend pytorch  # PyTorch FP16 (for the ratio)

Protocol (see paper_results/fps_optimization.md section 4):
  * warm-up frames discarded before timing
  * explicit torch.cuda.synchronize() at every stage boundary
  * output verified non-empty each frame (guards the silent-failure pitfall)
  * clock state recorded in the output
  * per-stage + end-to-end, mean/median/p95/p99
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import cv2, numpy as np, torch

BASE   = Path(__file__).resolve().parent
BACKUP = Path("/home/orin_nano/Desktop/Projects/real-time-wld-net-backup")
sys.path.insert(0, str(BASE / "iceml_src"))
sys.path.insert(0, str(BASE))

RES        = 512          # set from --res at run time
CONF       = 0.25
WARMUP     = 30
YOLO_PT    = BACKUP / "detection_model" / "fyp_test5_best.pt"


def engine_paths(res):
    return (BASE / "detection_model_TRT" / f"yolo5_fp16_{res}.engine",
            BASE / "models_TRT" / "iceml" / f"iceml_fp16_{res}.engine")


def clock_state():
    """Record governor + GPU clock so runs are never compared across clock states."""
    st = {}
    try:
        st["cpu_governor"] = Path(
            "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor").read_text().strip()
    except Exception:
        st["cpu_governor"] = "unknown"
    for node in ("/sys/class/devfreq/17000000.gpu/cur_freq",
                 "/sys/devices/gpu.0/devfreq/17000000.ga10b/cur_freq"):
        try:
            st["gpu_cur_hz"] = int(Path(node).read_text().strip()); break
        except Exception:
            continue
    for node in ("/sys/class/devfreq/17000000.gpu/max_freq",
                 "/sys/devices/gpu.0/devfreq/17000000.ga10b/max_freq"):
        try:
            st["gpu_max_hz"] = int(Path(node).read_text().strip()); break
        except Exception:
            continue
    if "gpu_cur_hz" in st and "gpu_max_hz" in st:
        st["gpu_at_max"] = st["gpu_cur_hz"] >= st["gpu_max_hz"] * 0.95
    try:
        free = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.splitlines()[1]
        st["mem_available_mb"] = int(free.split()[-1])
    except Exception:
        pass
    return st


def pct(a):
    a = np.asarray(a) * 1000.0  # ms
    return {"mean": round(float(a.mean()), 2), "median": round(float(np.median(a)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2),
            "min": round(float(a.min()), 2), "max": round(float(a.max()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["trt", "pytorch"], default="trt")
    ap.add_argument("--video", default="Object(3).mp4")
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--res", type=int, default=512, choices=[256, 512, 1024])
    args = ap.parse_args()

    global RES
    RES = args.res
    YOLO_ENG, ICEML_ENG = engine_paths(RES)

    dev = torch.device("cuda")
    clocks = clock_state()
    print(f"backend={args.backend}  res={RES}  clip={args.video}  frames={args.frames}")
    print(f"clocks: {clocks}")
    if not clocks.get("gpu_at_max", False):
        print("  !! GPU NOT AT MAX CLOCK -- run `sudo jetson_clocks` first; "
              "this run is measuring the governor.")

    # ── models ───────────────────────────────────────────────────────────────
    from ultralytics import YOLO
    if args.backend == "trt":
        if not YOLO_ENG.exists():
            sys.exit(f"missing YOLO engine {YOLO_ENG} -- export fyp_test5_best.pt first")
        if not ICEML_ENG.exists():
            sys.exit(f"missing ICEML engine {ICEML_ENG}")
        from trt_model import TRTModel, HAS_TRT
        if not HAS_TRT:
            sys.exit("tensorrt not importable")
        iceml = TRTModel(str(ICEML_ENG), device="cuda")
        yolo  = YOLO(str(YOLO_ENG))
    else:
        from run_paper_eval import load_iceml
        iceml = load_iceml(dev)
        yolo  = YOLO(str(YOLO_PT))

    cap = cv2.VideoCapture(str(BACKUP / "videos" / args.video))
    t_dec, t_pre, t_deh, t_det, t_e2e = [], [], [], [], []
    empty_out = 0
    n = 0

    while n < args.frames + WARMUP:
        w0 = time.perf_counter()

        a = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        b = time.perf_counter()

        # preprocess: resize + BGR->RGB + scale, folded into as few passes as possible
        small = cv2.resize(frame, (RES, RES), interpolation=cv2.INTER_AREA)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).to(dev).permute(2, 0, 1).unsqueeze(0)
        t = t.half().div_(255.0)
        torch.cuda.synchronize()
        c = time.perf_counter()

        with torch.no_grad():
            lr_inv = iceml(t)
            dhz_t  = (t.float() * lr_inv.float()).clamp(0, 1)
        torch.cuda.synchronize()
        d = time.perf_counter()

        dhz = (dhz_t.squeeze(0).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        dhz = cv2.cvtColor(dhz, cv2.COLOR_RGB2BGR)
        res = yolo(dhz, conf=CONF, verbose=False, imgsz=RES)[0]
        _ = res.boxes.xyxy.cpu().numpy()      # forces completion
        torch.cuda.synchronize()
        e = time.perf_counter()

        # guard the silent-failure pitfall: a run with no detections at all on a
        # clip known to contain them means inference is failing, not accelerating
        if res.boxes is None or len(res.boxes) == 0:
            empty_out += 1

        n += 1
        if n > WARMUP:                        # discard warm-up
            t_dec.append(b - w0); t_pre.append(c - b)
            t_deh.append(d - c);  t_det.append(e - d)
            t_e2e.append(e - w0)
    cap.release()

    timed = len(t_e2e)
    out = {
        "backend": args.backend, "clip": args.video, "resolution": f"{RES}x{RES}",
        "conf": CONF, "frames_timed": timed, "warmup_discarded": WARMUP,
        "clock_state": clocks,
        "frames_with_no_detection": empty_out,
        "stages_ms": {"decode": pct(t_dec), "preprocess": pct(t_pre),
                      "dehaze": pct(t_deh), "detect": pct(t_det),
                      "end_to_end": pct(t_e2e)},
        "fps_mean": round(1.0 / np.mean(t_e2e), 2),
        "fps_p99_worst_case": round(1.0 / np.percentile(t_e2e, 99), 2),
    }
    dest = BASE / "paper_results" / f"benchmark_{args.backend}_{RES}.json"
    dest.write_text(json.dumps(out, indent=2))

    print(f"\n--- {args.backend.upper()} @ {RES}x{RES} | {timed} frames timed ---")
    for k, v in out["stages_ms"].items():
        print(f"  {k:12s} mean {v['mean']:7.2f}  median {v['median']:7.2f}  "
              f"p95 {v['p95']:7.2f}  p99 {v['p99']:7.2f}")
    print(f"  FPS mean {out['fps_mean']}   FPS at p99 {out['fps_p99_worst_case']}")
    print(f"  frames with zero detections: {empty_out}/{timed}"
          + ("   <-- SUSPECT: verify inference is not silently failing" if empty_out > timed * 0.5 else ""))
    print(f"  -> {dest}")


if __name__ == "__main__":
    main()
