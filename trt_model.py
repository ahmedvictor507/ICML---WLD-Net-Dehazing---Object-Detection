import os
import torch

try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

class TRTModel:
    def __init__(self, engine_path, device="cuda"):
        if not HAS_TRT:
            raise RuntimeError("TensorRT library is not installed or import failed.")
        
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        print(f"[*] Loading TensorRT engine from {engine_path}...")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context. This is usually due to out-of-memory (OOM) on the GPU.")
            
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

    def forward(self, input_tensor):
        """
        Runs inference on the provided PyTorch tensor.
        input_tensor: PyTorch tensor (CPU or CUDA).
        """
        # Ensure input is a CUDA tensor
        if not input_tensor.is_cuda:
            input_tensor = input_tensor.to(self.device)
            
        # Convert dtype if necessary to match the engine expectations
        expected_dtype = self.engine.get_tensor_dtype(self.input_name)
        if expected_dtype == trt.float32 and input_tensor.dtype == torch.float16:
            input_tensor = input_tensor.float()
        elif expected_dtype == trt.float16 and input_tensor.dtype == torch.float32:
            input_tensor = input_tensor.half()

        # TensorRT expects contiguous memory
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()
            
        # Set input shape dynamically based on input tensor dimensions
        self.context.set_input_shape(self.input_name, input_tensor.shape)
        
        # Allocate output tensor on the same device with correct torch dtype
        output_shape = self.context.get_tensor_shape(self.output_name)
        output_trt_dtype = self.engine.get_tensor_dtype(self.output_name)
        output_torch_dtype = torch.float16 if output_trt_dtype == trt.float16 else torch.float32
        
        output_tensor = torch.empty(tuple(output_shape), dtype=output_torch_dtype, device=self.device)
        
        # Bind PyTorch CUDA tensor memory addresses directly to TensorRT execution context
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, output_tensor.data_ptr())
        
        # Execute asynchronously (using default stream 0)
        self.context.execute_async_v3(0)
        
        return output_tensor

    def __call__(self, x):
        return self.forward(x)
