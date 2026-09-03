import os
import sys
import time
import torch
from pathlib import Path

# Add source paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "iceml_src"))

def test_inference():
    print("[*] Running speed test between PyTorch and TensorRT...")
    
    # 1. Load PyTorch model
    from retinexmodule import dehazing_module as ICEMLModule
    pth_path = BASE_DIR / "models" / "iceml" / "final_dehazing_model.pth"
    
    print("[*] Loading PyTorch model...")
    model_pytorch = ICEMLModule(in_channels=3)
    full_sd = torch.load(str(pth_path), map_location="cpu")
    prefix = "retinex_model."
    new_sd = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}
    model_pytorch.load_state_dict(new_sd)
    model_pytorch = model_pytorch.cuda().eval().half()
    
    # 2. Load TensorRT Engine
    engine_path = BASE_DIR / "models" / "iceml" / "final_dehazing_model_512x512.engine"
    if not engine_path.exists():
        print(f"[-] Engine file {engine_path} not found. Please wait for compilation to complete.")
        return
        
    from trt_model import TRTModel
    print("[*] Loading TensorRT model...")
    model_trt = TRTModel(str(engine_path), device="cuda")
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 512, 512, device="cuda", dtype=torch.float16)
    
    # Warmup
    print("[*] Warming up...")
    for _ in range(10):
        _ = model_pytorch(dummy_input)
        _ = model_trt(dummy_input.float())
        
    # PyTorch Benchmark
    print("[*] Benchmarking PyTorch (.pth) model...")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = model_pytorch(dummy_input)
    torch.cuda.synchronize()
    t1 = time.time()
    pytorch_time = (t1 - t0) / 100 * 1000  # ms
    pytorch_fps = 1000 / pytorch_time
    
    # TensorRT Benchmark
    print("[*] Benchmarking TensorRT (.engine) model...")
    # Convert input to float32 if the ONNX was exported as float32
    dummy_input_fp32 = dummy_input.float()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = model_trt(dummy_input_fp32)
    torch.cuda.synchronize()
    t1 = time.time()
    trt_time = (t1 - t0) / 100 * 1000  # ms
    trt_fps = 1000 / trt_time
    
    print("\n" + "="*50)
    print(f"Results for 512x512 input on Jetson Orin Nano:")
    print(f"PyTorch (.pth)  : {pytorch_time:.2f} ms per frame ({pytorch_fps:.1f} FPS)")
    print(f"TensorRT (.engine): {trt_time:.2f} ms per frame ({trt_fps:.1f} FPS)")
    print(f"Speedup Factor  : {pytorch_time / trt_time:.2f}x faster!")
    print("="*50)

if __name__ == "__main__":
    test_inference()
