import os
import sys
import argparse
import subprocess
import torch
from pathlib import Path

# Add source directories to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "iceml_src"))
sys.path.insert(0, str(BASE_DIR / "wldnet_src"))

def export_iceml_to_onnx(model_path, onnx_path, width, height, dynamic=False):
    print(f"[*] Loading ICEML (RtxICE-Net) weights from {model_path}...")
    from retinexmodule import dehazing_module as ICEMLModule
    model = ICEMLModule(in_channels=3)
    
    # Load state dict (mapping keys to match class definition)
    full_sd = torch.load(model_path, map_location="cpu")
    prefix = "retinex_model."
    new_sd = {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}
    model.load_state_dict(new_sd)
    model.eval()

    print("[*] Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, height, width)
    
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"}
        }
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )
    print(f"[+] Successfully exported ONNX model to {onnx_path}")

def export_wldnet_to_onnx(model_path, onnx_path, width, height, dynamic=False):
    print(f"[*] Loading WLD-Net weights from {model_path}...")
    import dehazing_model as wld_mod
    model = wld_mod.Dehazing_Model()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print("[*] Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, height, width)
    
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"}
        }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )
    print(f"[+] Successfully exported ONNX model to {onnx_path}")

def compile_onnx_to_engine(onnx_path, engine_path, dynamic=False, fp16=True, workspace_mb=1024):
    print(f"[*] Compiling ONNX to TensorRT engine: {onnx_path} -> {engine_path}...")
    trtexec_path = "/usr/src/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec_path):
        print(f"[-] Error: trtexec not found at {trtexec_path}")
        return False
    
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mb}M"
    ]
    
    if fp16:
        cmd.append("--fp16")
        
    if dynamic:
        # Define shape profiles for dynamic axes
        cmd.append("--minShapes=input:1x3x256x256")
        cmd.append("--optShapes=input:1x3x512x512")
        cmd.append("--maxShapes=input:1x3x1024x1024")
        
    print(f"[*] Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode == 0:
        print(f"[+] Successfully compiled TensorRT engine to {engine_path}")
        return True
    else:
        print("[-] Compilation failed with the following error:")
        print(result.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch .pth models to TensorRT .engine")
    parser.add_argument("--model-type", type=str, choices=["iceml", "wldnet"], required=True, help="Model type: iceml or wldnet")
    parser.add_argument("--src", type=str, required=True, help="Path to input .pth weights file")
    parser.add_argument("--dest-dir", type=str, default=None, help="Directory to save the exported files (defaults to model's directory)")
    parser.add_argument("--width", type=int, default=512, help="Input width for static engine")
    parser.add_argument("--height", type=int, default=512, help="Input height for static engine")
    parser.add_argument("--dynamic", action="store_true", help="Compile with dynamic input shapes (256x256 to 1024x1024)")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 precision (runs in FP32 instead)")
    parser.add_argument("--workspace", type=int, default=1024, help="Workspace size limit in MiB (default: 1024)")
    
    args = parser.parse_args()
    
    src_path = Path(args.src).resolve()
    if not src_path.exists():
        print(f"[-] Error: Source file {src_path} does not exist.")
        sys.exit(1)
        
    dest_dir = Path(args.dest_dir) if args.dest_dir else src_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    stem = src_path.stem
    shape_suffix = "_dynamic" if args.dynamic else f"_{args.width}x{args.height}"
    onnx_path = dest_dir / f"{stem}{shape_suffix}.onnx"
    engine_path = dest_dir / f"{stem}{shape_suffix}.engine"
    
    # Export ONNX
    if args.model_type == "iceml":
        export_iceml_to_onnx(str(src_path), str(onnx_path), args.width, args.height, args.dynamic)
    elif args.model_type == "wldnet":
        export_wldnet_to_onnx(str(src_path), str(onnx_path), args.width, args.height, args.dynamic)
        
    # Compile Engine
    success = compile_onnx_to_engine(str(onnx_path), str(engine_path), args.dynamic, not args.no_fp16, args.workspace)
    if success:
        print(f"\n[SUCCESS] Your engine is ready at: {engine_path}")
    else:
        print("\n[FAILURE] Engine compilation failed.")

if __name__ == "__main__":
    main()
