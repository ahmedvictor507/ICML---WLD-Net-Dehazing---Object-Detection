"""
Batch-converts every dehazing model (ICEML + all 4 WLD-Net datasets) and the
YOLO detection model to TensorRT engines at 256/512/1024, using a single
consistent naming scheme that main.py can look up deterministically:

    models_TRT/iceml/iceml_fp16_{res}.engine
    models_TRT/wldnet/{PREFIX}_fp16_{res}.engine      (PREFIX: OH/RD/NH/DH)
    detection_model_TRT/yolo_fp16_{res}.engine

Run: /path/to/dehaze_env/bin/python convert_all_trt.py
"""
import sys
import time
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from convert_to_trt import (
    export_iceml_to_onnx,
    export_wldnet_to_onnx,
    compile_onnx_to_engine,
)

RESOLUTIONS = [256, 512, 1024]
WORKSPACE_MB = {256: 1024, 512: 2048, 1024: 4096}

WLD_MODELS = {
    "OH": BASE_DIR / "models" / "wldnet" / "OH-dehazing_model_final.pth",
    "RD": BASE_DIR / "models" / "wldnet" / "RD_dehazing_model_final.pth",
    "NH": BASE_DIR / "models" / "wldnet" / "NH_dehazing_model_final.pth",
    "DH": BASE_DIR / "models" / "wldnet" / "DH_dehazing_model_final.pth",
}
ICEML_MODEL = BASE_DIR / "models" / "iceml" / "final_dehazing_model.pth"
YOLO_PT = BASE_DIR / "detection_model" / "fyp_test3_best.pt"

results = []


def do_dehazing(model_type, src_path, dest_dir, name):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for res in RESOLUTIONS:
        onnx_path = dest_dir / f"{name}_fp16_{res}.onnx"
        engine_path = dest_dir / f"{name}_fp16_{res}.engine"
        tag = f"{name} @ {res}"
        if engine_path.exists():
            print(f"[=] Skipping {tag} — engine already exists")
            results.append((tag, "skipped"))
            continue
        try:
            t0 = time.time()
            if model_type == "iceml":
                export_iceml_to_onnx(str(src_path), str(onnx_path), res, res)
            else:
                export_wldnet_to_onnx(str(src_path), str(onnx_path), res, res)
            ok = compile_onnx_to_engine(
                str(onnx_path), str(engine_path),
                dynamic=False, fp16=True, workspace_mb=WORKSPACE_MB[res],
            )
            if ok:
                print(f"[+] {tag} done in {time.time()-t0:.1f}s")
                results.append((tag, "ok"))
            else:
                results.append((tag, "FAILED (compile)"))
        except Exception:
            print(f"[-] {tag} raised an exception:")
            traceback.print_exc()
            results.append((tag, "FAILED (exception)"))


def do_yolo():
    from ultralytics import YOLO
    dest_dir = BASE_DIR / "detection_model_TRT"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for res in RESOLUTIONS:
        engine_path = dest_dir / f"yolo_fp16_{res}.engine"
        tag = f"yolo @ {res}"
        if engine_path.exists():
            print(f"[=] Skipping {tag} — engine already exists")
            results.append((tag, "skipped"))
            continue
        try:
            t0 = time.time()
            model = YOLO(str(YOLO_PT))
            exported = model.export(format="engine", imgsz=res, half=True, device=0,
                                     workspace=WORKSPACE_MB[res] / 1024, batch=1)
            exported_path = Path(exported)
            exported_path.rename(engine_path)
            # ultralytics also drops a sibling .onnx in the source dir; move it alongside for reference
            sibling_onnx = YOLO_PT.with_suffix(".onnx")
            if sibling_onnx.exists():
                sibling_onnx.rename(dest_dir / f"yolo_fp16_{res}.onnx")
            print(f"[+] {tag} done in {time.time()-t0:.1f}s -> {engine_path}")
            results.append((tag, "ok"))
        except Exception:
            print(f"[-] {tag} raised an exception:")
            traceback.print_exc()
            results.append((tag, "FAILED (exception)"))


def main():
    print("=" * 60)
    print("ICEML")
    print("=" * 60)
    do_dehazing("iceml", ICEML_MODEL, BASE_DIR / "models_TRT" / "iceml", "iceml")

    for prefix, src in WLD_MODELS.items():
        print("=" * 60)
        print(f"WLD-Net [{prefix}]")
        print("=" * 60)
        do_dehazing("wldnet", src, BASE_DIR / "models_TRT" / "wldnet", prefix)

    print("=" * 60)
    print("YOLO Detection")
    print("=" * 60)
    do_yolo()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for tag, status in results:
        print(f"  {status:>20} : {tag}")
    n_fail = sum(1 for _, s in results if s.startswith("FAILED"))
    print(f"\n{len(results)} total, {n_fail} failed.")


if __name__ == "__main__":
    main()
