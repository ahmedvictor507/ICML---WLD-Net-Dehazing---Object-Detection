# Dehazing + Object Detection on NVIDIA Jetson Orin Nano

Real-time pipeline that **dehazes each video frame with a small Retinex-based network (ICEML) and then runs YOLOv8s object detection**, deployed on an NVIDIA Jetson Orin Nano with TensorRT FP16. Includes a Tkinter GUI, a TensorRT conversion toolchain, a reproducible latency benchmark and the batch evaluation used for the accompanying paper.

**Result:** 17.4 FPS mean (15.3 FPS at p99) end-to-end at 512x512 on a Jetson Orin Nano, measured with clocks pinned and 500 timed frames per run (see [Benchmarks](#benchmarks)).

> Research done at UTM MJIIT (Kuala Lumpur) under the supervision of Dr. Uswah Khairudin.

## What is in the pipeline

```
video frame -> decode -> preprocess -> dehaze (ICEML, TensorRT FP16) -> detect (YOLOv8s, TensorRT FP16) -> display / save
```

| Stage | Purpose |
|---|---|
| **Dehazing** | ICEML (RtxICE-Net), a Retinex illumination-map dehazer. **415,036 executing parameters.** WLD-Net variants (O-HAZE, RESIDE, NH-HAZE, D-HAZE) are also supported for comparison. |
| **Detection** | YOLOv8s trained on 4 classes: `car`, `exit`, `person`, `warning`. |
| **Deployment** | PyTorch -> ONNX -> TensorRT FP16 engines at 256, 512 and 1024 px. The GUI falls back to PyTorch when no engine is present. |

## Benchmarks

Measured on a Jetson Orin Nano (GPU pinned at 1020 MHz, `jetson_clocks`), one clip, **500 timed frames after 30 warm-up frames**, with `torch.cuda.synchronize()` at every stage boundary. Each run records the CPU governor and GPU clock so runs are never compared across clock states.

| Backend | Resolution | Decode | Preprocess | Dehaze | Detect | End-to-end (ms) | FPS mean | FPS p99 (worst case) |
|---|---|---|---|---|---|---|---|---|
| **TensorRT FP16** | 512x512 | 5.8 | 15.3 | 24.9 | 11.6 | 57.5 | **17.4** | **15.3** |
| PyTorch FP16 | 512x512 | 5.9 | 15.3 | 30.2 | 24.5 | 75.8 | 13.2 | 11.5 |
| TensorRT FP16 | 1024x1024 | 5.8 | 25.4 | 78.4 | 27.2 | 136.7 | 7.3 | 6.4 |

All values are mean milliseconds per stage. Raw numbers, including median, p95 and p99 for every stage, are in `paper_results/benchmark_*.json`.

Reproduce:

```bash
python benchmark_deployment.py --backend trt --res 512       # TensorRT FP16
python benchmark_deployment.py --backend pytorch --res 512   # PyTorch FP16, for the ratio
```

### What made it faster

Profiling came first; the optimisation order was not guessed. Full method and the measurement protocol are in [`paper_results/fps_optimization.md`](paper_results/fps_optimization.md). The main findings:

| Change | Effect |
|---|---|
| Pin GPU clocks (`sudo jetson_clocks`) | 2.4x GPU time (default governor idled the GPU at 306 of 1020 MHz) |
| TensorRT execution provider instead of CUDA EP | 1.8x |
| FP16 build precision | 1.27x (modest because the workload is memory-bandwidth-bound) |
| Rewrite preprocessing (one pass instead of four) | 16.5x on that stage |

Two lessons recorded there: on this hardware, memory traffic rather than arithmetic sets the frame rate; and INT8 on the CUDA execution provider was **2.3x slower** than FP32 because of fallback conversions, so precision modes must be checked per execution provider.

## Does dehazing help detection?

Mixed. The batch evaluation (`run_paper_eval.py`) runs YOLO on the raw hazy frame and on the dehazed frame for the same video and compares **detection counts** (not accuracy: the videos are unlabelled). Confidence threshold 0.25, 512x512.

| Video | Frames | Detections (hazy) | Detections (dehazed) | Change |
|---|---|---|---|---|
| human(2)_test(1) | 1803 | 986 | 1018 | +3.2% |
| Human (3) | 603 | 2009 | 1884 | -6.2% |
| Object(3) | 2911 | 2902 | 2905 | +0.1% |
| Object(6) | 3277 | 2349 | 2032 | -13.5% |

These four clips are a subset of the sequences studied in the paper. [**Add here, in your own words and matching the paper: how many sequences were evaluated in total, how many improved, and how the four shown were chosen.**] Per-class counts and mean confidence are in `paper_results/stats.md`. Before/after video is in `paper_results/*.mp4`.

## Quick start

Requires a Jetson (JetPack 6 with TensorRT) or any CUDA machine, plus Python 3.10.

```bash
python3 -m venv ../dehaze_env && source ../dehaze_env/bin/activate
pip install torch torchvision opencv-python numpy pillow ultralytics   # use NVIDIA's Jetson wheels for torch on-device
./run.sh                                                               # launches the GUI
```

The GUI lets you pick a dehazing model (ICEML or one of the WLD-Net variants), a working resolution, and a video or camera source, and shows the hazy and dehazed streams with detection boxes side by side.

### Convert models to TensorRT

```bash
# ICEML
python convert_to_trt.py --model-type iceml --src models/iceml/final_dehazing_model.pth --width 512 --height 512 --workspace 512

# a WLD-Net variant (e.g. O-HAZE)
python convert_to_trt.py --model-type wldnet --src models/wldnet/OH-dehazing_model_final.pth --width 512 --height 512 --workspace 512

# everything at once
python convert_all_trt.py
```

Engines are hardware-specific: build them on the device you deploy to. To verify a build, check engine size: an FP16 engine should be about 55% the size of the FP32 one; a full-size engine means the flag was silently ignored.

## Repository layout

| Path | Contents |
|---|---|
| `main.py` | Tkinter GUI (heavy imports are lazy to avoid ARM/Jetson segfaults before Tk starts) |
| `iceml_src/`, `wldnet_src/` | Model definitions |
| `trt_model.py`, `convert_to_trt.py`, `convert_all_trt.py`, `test_trt.py` | TensorRT wrappers and conversion |
| `benchmark_deployment.py` | Stage-by-stage latency benchmark |
| `run_paper_eval.py`, `compile_figures.py` | Batch evaluation and figure generation for the paper |
| `paper_results/` | Benchmarks, statistics, figures and result videos |

## Known limitations

- Some scripts still contain absolute paths from the development Jetson (`/home/orin_nano/...`). Edit `BASE_DIR` / `BACKUP` at the top of `run_paper_eval.py` and `benchmark_deployment.py`, or pass your own paths.
- Detection results are counts on unlabelled video, not accuracy, precision, recall or mAP.
- Benchmarks are for a single clip and a single device.
- The detector's classes (`car`, `exit`, `person`, `warning`) come from a small custom dataset and will not generalise without retraining.

## Credits

- ICEML and WLD-Net dehazing models: [**add authors, paper titles and links, and the licence of the released weights**].
- Detector: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- Author of this pipeline, deployment and evaluation: Ahmed Montasser ([LinkedIn](https://www.linkedin.com/in/)) . Contact: ahmontasser507@gmail.com

## License

[**Choose a licence that is compatible with the model weights and Ultralytics (AGPL-3.0).**]
