# Real-Time Performance Optimization — Methods

Companion to `stats.md`. Documents how the dehazing + detection pipeline was
profiled and accelerated on the Jetson Orin Nano, what worked, what did not,
and the measurement protocol required to trust any of the numbers.

---

## 1. Diagnosis method

The optimization order was not guessed. Two cheap measurements determined it,
and both are worth reproducing before optimizing any new stage.

**Test 1 — is the GPU saturated?** Sum the measured times of stages that run
concurrently and compare that sum against the wall-clock window they occupy.
When the summed stage time approaches the wall window, the stages are not
overlapping: they are serialized behind a saturated GPU. This ruled out
scheduling and pipelining fixes, which only help when the device is idle.

**Test 2 — compute-bound or bandwidth-bound?** Run small models against large
inputs. If throughput tracks input size rather than model FLOPs, the pipeline is
memory-bandwidth-bound, not compute-bound.

The second test explains the single most counterintuitive result in the table
below: **FP16 delivered 1.27×, not the ~2× a compute-bound workload would
predict.** Halving arithmetic precision does little when the bottleneck is
moving bytes. The same finding is what redirected attention to *preprocessing*
(pure memory traffic) and to *clock rates* (which raise memory clocks as well as
core clocks) — the two largest wins. Neither would have been an obvious target
from a FLOP-counting view of the pipeline.

**Takeaway for the write-up:** on this class of embedded accelerator, with small
models and large frames, memory traffic — not arithmetic — sets the frame rate.

---

## 2. Optimizations that worked

Ordered by payoff.

| # | Optimization | Speedup | Type | Scope |
|---|---|---|---|---|
| 1 | `sudo jetson_clocks` | **2.4×** GPU time | Deployment | Whole pipeline |
| 2 | TensorRT execution provider (over CUDA EP) | **1.8×** | Configuration | Inference |
| 3 | FP16 build precision | **1.27×** | Configuration | Inference |
| 4 | Preprocessing rewrite | **16.5×** | Code | Preprocessing stage |
| 5 | Cross-frame tensor reuse | allocation removed | Code | Per-frame overhead |

### 1. Clock governor — 2.4× on GPU time
The default governor idles the GPU at **306 MHz of an available 1020 MHz** and
does not reliably ramp under this workload's bursty, short-kernel profile: the
governor's ramp decision lags the kernel that needed the clocks. `jetson_clocks`
pins all clocks to maximum, and it is not a code change at all.

Two practical notes: **it does not survive a reboot**, so it belongs in the
service start-up path rather than in a one-off setup step; and any benchmark run
without it is measuring the governor, not the pipeline.

### 2. TensorRT EP instead of CUDA EP — 1.8×
Configuration only, no model or code change. The CUDA EP dispatches to generic
cuDNN kernels; the TensorRT EP builds a fused, hardware-specific engine.

Corollary worth recording: **INT8 on the CUDA EP was 2.3× *slower* than FP32.**
The CUDA EP has no INT8 kernels for these layers, so quantization forced
fallback conversions on every layer boundary — a quantization "optimization"
that costs more than it saves. Precision modes are properties of the execution
provider, not of the model.

### 3. FP16 build precision — 1.27×, accuracy-neutral
Modest for the reason given in §1 (bandwidth-bound), but free in accuracy terms
on this pipeline.

**Verification step — do not assume the flag applied.** A precision flag can be
silently ignored when a layer or EP does not support it. Confirm by checking the
cached engine files: an FP16 engine is **~55% the size of its FP32 counterpart**.
A cached engine at full size means the build fell back and the flag did nothing.

### 4. Preprocessing rewrite — 16.5× on that stage
The original preprocessing made **four separate passes over 6 MB per frame**
(channel split, type conversion, mean subtraction, standard-deviation division).
Each pass is a full read and write of the frame through memory — precisely the
resource §1 identified as the bottleneck.

Collapsed into **one pass** by two changes:
- splitting `uint8` data directly, rather than converting to float first and then splitting;
- folding the mean and standard-deviation normalization into `convertTo`'s
  `alpha` and `beta` parameters, so scaling and offset happen inside the single
  conversion that was already required.

The arithmetic is identical; only the number of memory traversals changed. This
is the clearest demonstration of the §1 finding, and the largest single-stage
win in the project.

### 5. Cross-frame tensor reuse
Allocating input tensors once and reusing them across frames removes one type
conversion and roughly **19 MB of allocation per frame**. Beyond the raw
allocator cost, this matters for stability: sustained per-frame allocation on a
unified-memory device drives the memory pressure implicated in the silent
inference failures described in §4.

### On composing these numbers
The factors above are **stage-level and partly overlapping, and must not be
multiplied into a headline figure.** Clocks and FP16 both act on GPU time;
preprocessing's 16.5× applies to one stage's share of the frame budget, so its
end-to-end effect is bounded by that share (Amdahl). Any total speedup quoted in
the paper should be a measured end-to-end number, not a product of this column.

---

## 3. What failed

Recorded deliberately — the negative results carry as much method as the wins.

| Attempt | Outcome | Cause |
|---|---|---|
| INT8 quantization | Abandoned | Bias export bug |
| Illumination map applied at full resolution | Rejected | Worse output *and* more expensive |
| CPU offload | Rejected | No improvement |

**INT8** failed on a bias export bug, not on a fundamental limit — worth
revisiting if the export path is fixed, and distinct from the separate CUDA EP
INT8 slowdown in §2.2.

**Full-resolution illumination map** is the most useful negative result: applying
the retinex illumination map at full resolution cost more *and* produced worse
output. The map is low-frequency by construction, so computing it at reduced
resolution and upsampling loses nothing it actually encodes, while the
full-resolution path adds both memory traffic and noise. A rare case where the
cheaper option is also the better one.

**CPU offload** gave no improvement, consistent with §1: moving work to the CPU
does not help when the bottleneck is shared memory bandwidth, since both
processors contend for the same bus on a unified-memory device.

---

## 4. Benchmarking pitfalls

**Four of six measured results were reported and then had to be retracted.**
Each cause below produced numbers that looked plausible. This section is the one
to carry into any future measurement session; treat it as a pre-flight checklist
rather than a narrative.

| # | Pitfall | Effect on results |
|---|---|---|
| 1 | Unseeded particle filter | Non-deterministic outputs; runs not comparable |
| 2 | Silent inference failures under memory pressure | **Flattered results** — skipped work timed as fast work |
| 3 | Shared config left pointing at the wrong clip | Wrong workload measured entirely |
| 4 | Video-recording overhead inside the timing loop | Inflated per-frame times |

Pitfall 2 is the dangerous one: it biases results in the *favourable* direction,
so it does not trigger the suspicion that an implausibly slow number would. A
failing inference produces no output to check and no error to catch, and the
saved time is indistinguishable from a genuine optimization. Any measured
speedup that arrives without a corresponding mechanism should be treated as
suspected pitfall 2 until output correctness is verified frame by frame.

**Checklist before trusting a measurement:**
1. Seed every stochastic component; confirm two runs produce identical output.
2. Verify inference actually produced valid output on every frame — do not infer success from the absence of an error.
3. Confirm the config points at the intended clip and model; do not inherit a shared config across runs.
4. Move I/O — video encode, disk writes, display — outside the timing window.
5. Time each stage separately as well as end-to-end; a stage that improves while the total does not indicates the bottleneck moved elsewhere.
6. Confirm `jetson_clocks` state, and report it. It is a 2.4× factor on GPU time and silently invalidates cross-session comparison.

---

## 5. Applicability to the detection results in `stats.md`

The detection evaluation was run **before** this optimization work was applied.
Its timing column should be read accordingly:

- **Clocks were not pinned.** The governor was `schedutil` during the run, so
  the reported GPU times reflect the 306 MHz idle-clock behaviour of §2.1 and
  are pessimistic by up to the 2.4× factor.
- **TensorRT was not used.** The run used PyTorch FP16 with the CUDA path;
  the TensorRT engines were not in the loop, forgoing §2.2's 1.8×.
- **The `end_to_end` column contains recording overhead** — three
  `VideoWriter.write` calls, frame decode, and resize sit inside the timing
  window. This is pitfall 4 in the pipeline's own measurement, and that column
  should not be quoted as pipeline latency. The `dehaze` and
  `detect_two_passes` columns are clean: each brackets only its own stage, and
  each ends in a device-to-host copy that forces CUDA synchronization, so they
  are not measuring asynchronous kernel launch.
- **`detect_two_passes` is two detector invocations,** one on the hazy frame and
  one on the dehazed frame, because each frame is scored both ways for the
  comparison. A deployed single-path system runs one.

Deployment latency should therefore be measured in a dedicated run with clocks
pinned, TensorRT enabled, recording disabled, and a single detection pass —
**not** derived from the comparison run's numbers.
