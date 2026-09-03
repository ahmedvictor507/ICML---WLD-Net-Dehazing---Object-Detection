# ICEML Dehazing + YOLO Detection — Results

- Detector: `fyp_test5_best.pt` (classes: car, exit, person, warning)
- Dehazer: ICEML / RtxICE-Net (`final_dehazing_model.pth`)
- Working resolution: 512x512, confidence threshold: 0.25

## Summary

| Video | Frames | Det. (hazy) | Det. (dehazed) | Δ | Det/frame hazy | Det/frame dehazed | Mean conf hazy | Mean conf dehazed |
|---|---|---|---|---|---|---|---|---|
| human(2)_test(1).mp4 | 1803 | 986 | 1018 | +32 (+3.2%) | 0.547 | 0.565 | 0.681 | 0.647 |
| Human (3).mp4 | 603 | 2009 | 1884 | -125 (-6.2%) | 3.332 | 3.124 | 0.589 | 0.631 |
| Object(3).mp4 | 2911 | 2902 | 2905 | +3 (+0.1%) | 0.997 | 0.998 | 0.884 | 0.875 |
| Object(6).mp4 | 3277 | 2349 | 2032 | -317 (-13.5%) | 0.717 | 0.62 | 0.83 | 0.845 |

## Per-class detections

| Video | Class | Hazy count | Dehazed count | Hazy mean conf | Dehazed mean conf |
|---|---|---|---|---|---|
| human(2)_test(1).mp4 | exit | 0 | 4 | — | 0.464 |
| human(2)_test(1).mp4 | person | 963 | 994 | 0.688 | 0.652 |
| human(2)_test(1).mp4 | warning | 23 | 20 | 0.392 | 0.414 |
| Human (3).mp4 | car | 363 | 434 | 0.644 | 0.604 |
| Human (3).mp4 | person | 1211 | 1152 | 0.615 | 0.695 |
| Human (3).mp4 | warning | 435 | 298 | 0.469 | 0.426 |
| Object(3).mp4 | warning | 2902 | 2905 | 0.884 | 0.875 |
| Object(6).mp4 | warning | 2349 | 2032 | 0.83 | 0.845 |

## Timing (ms/frame, Jetson Orin Nano, PyTorch FP16)

| Video | Dehaze | Detection (2 passes) | End-to-end |
|---|---|---|---|
| human(2)_test(1).mp4 | 31.77 | 43.12 | 93.84 |
| Human (3).mp4 | 32.04 | 45.47 | 96.39 |
| Object(3).mp4 | 36.31 | 48.13 | 117.67 |
| Object(6).mp4 | 35.74 | 45.98 | 114.62 |
