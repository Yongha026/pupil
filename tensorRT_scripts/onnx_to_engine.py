"""
Script to build and verify NVIDIA TensorRT engine (.engine) from an ONNX model.
Supports custom architectures like GBC (archs_GBC.py) in nnU-Net v2.

Designed to be executed on target GPU servers:
    1) Server 1 (RTX A6000, sm_86, CUDA 13.x):
       python onnx_to_engine.py \
           --onnx_file nnunet_gbc_backbone.onnx \
           --output_engine nnunet_gbc_backbone_a6000.engine \
           --precision fp16 \
           --min_batch 1 --opt_batch 1 --max_batch 8 \
           --workspace_gb 8 \
           --device 0

    2) Server 2 (RTX 4090, sm_89, CUDA 12.2):
       python onnx_to_engine.py \
           --onnx_file nnunet_gbc_backbone.onnx \
           --output_engine nnunet_gbc_backbone_4090.engine \
           --precision fp16 \
           --min_batch 1 --opt_batch 1 --max_batch 4 \
           --workspace_gb 4 \
           --device 0
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

# Official documentation citations:
# [source](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html#python-api-docs)
# [source](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dynamic-shapes.html)
# [source](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-python-api.html)

try:
    import tensorrt as trt
except ImportError as e:
    print(f"[Error] Failed to import tensorrt: {e}")
    print("Please install TensorRT or run inside an environment with TensorRT enabled.")
    print("Example: pip install tensorrt or activate your server's conda environment.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX model to NVIDIA TensorRT engine.")
    parser.add_argument(
        "--onnx_file",
        type=str,
        default="nnunet_gbc_backbone.onnx",
        help="Path to input ONNX model file. Default: nnunet_gbc_backbone.onnx"
    )
    parser.add_argument(
        "--output_engine",
        type=str,
        default=None,
        help="Path to output TensorRT engine file. If omitted, named after ONNX file."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="Target inference precision: fp16 (default), fp32, or bf16."
    )
    parser.add_argument(
        "--min_batch",
        type=int,
        default=1,
        help="Minimum batch size for dynamic shape optimization profile. Default: 1"
    )
    parser.add_argument(
        "--opt_batch",
        type=int,
        default=1,
        help="Optimal batch size for dynamic shape optimization profile. Default: 1"
    )
    parser.add_argument(
        "--max_batch",
        type=int,
        default=4,
        help="Maximum batch size for dynamic shape optimization profile. Default: 4"
    )
    parser.add_argument(
        "--workspace_gb",
        type=float,
        default=4.0,
        help="Max GPU memory pool limit for workspace in GiB. Default: 4.0"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index. Default: 0"
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip post-build engine verification pass."
    )
    return parser.parse_args()


def build_engine_from_onnx(
    onnx_file: str,
    output_engine: str = None,
    precision: str = "fp16",
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 4,
    workspace_gb: float = 4.0,
    device_id: int = 0,
    verify: bool = True
):
    if not os.path.isfile(onnx_file):
        raise FileNotFoundError(f"Input ONNX file not found: {onnx_file}")

    if output_engine is None:
        base_name = os.path.splitext(onnx_file)[0]
        output_engine = f"{base_name}.engine"

    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device_name = torch.cuda.get_device_name(device_id)
        print(f"[*] Target GPU: [{device_id}] {device_name}")
    else:
        print("[!] Warning: CUDA is not detected via PyTorch. TensorRT compilation requires an active NVIDIA GPU.")

    print(f"[*] TensorRT version: {trt.__version__}")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    # 1. Create Network Definition
    network_flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # 2. Parse ONNX model
    print(f"[*] Parsing ONNX model: {onnx_file}...")
    parser = trt.OnnxParser(network, logger)
    with open(onnx_file, "rb") as model_file:
        parsed_success = parser.parse(model_file.read())
        if not parsed_success:
            print("[Error] Failed to parse the ONNX file:")
            for error_idx in range(parser.num_errors):
                err = parser.get_error(error_idx)
                print(f"    - Error #{error_idx}: {err.desc()} (Line: {err.line()}, Node: {err.node()})")
            raise RuntimeError(f"ONNX parsing failed for: {onnx_file}")
    print("[+] ONNX parsing completed successfully.")

    # 3. Create Builder Configuration
    config = builder.create_builder_config()

    # Configure Workspace Memory Pool Limit
    workspace_bytes = int(workspace_gb * (1024 ** 3))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        print(f"[*] Workspace memory pool limit set to: {workspace_gb:.1f} GiB ({workspace_bytes} bytes)")
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_bytes
        print(f"[*] Max workspace size set to: {workspace_gb:.1f} GiB")

    # Configure Precision Mode
    prec = precision.lower()
    if prec == "fp16":
        if hasattr(trt.BuilderFlag, "FP16"):
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("[*] Enabled FP16 Precision Mode (Hardware Tensor Core acceleration supported).")
            else:
                config.set_flag(trt.BuilderFlag.FP16)
                print("[!] Enabled FP16 Precision Mode (Hardware lacks native fast FP16, but flag accepted).")
        else:
            print("[*] Note: BuilderFlag.FP16 not present in this TensorRT version (strongly typed mode active).")
    elif prec == "bf16":
        if hasattr(trt.BuilderFlag, "BF16"):
            config.set_flag(trt.BuilderFlag.BF16)
            print("[*] Enabled BF16 Precision Mode.")
        else:
            print("[!] BF16 requested but BuilderFlag.BF16 is not supported in this TensorRT version.")
    elif prec == "fp32":
        print("[*] Using FP32 Precision Mode.")

    # 4. Configure Optimization Profile for Dynamic Batch Axis
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    raw_shape = input_tensor.shape
    print(f"[*] Input tensor detected: '{input_name}', shape={raw_shape}")

    spatial_shape = tuple(raw_shape[1:])
    min_shape = (min_batch,) + spatial_shape
    opt_shape = (opt_batch,) + spatial_shape
    max_shape = (max_batch,) + spatial_shape

    print(f"[*] Configuring Optimization Profile for '{input_name}':")
    print(f"    - Min shape: {min_shape}")
    print(f"    - Opt shape: {opt_shape}")
    print(f"    - Max shape: {max_shape}")

    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # 5. Build Serialized Engine
    print(f"\n[*] Starting TensorRT engine compilation for: {output_engine}...")
    start_time = time.time()

    if hasattr(builder, "build_serialized_network"):
        # Modern API (TensorRT 8.5+, 10.x, 11.x)
        serialized_engine = builder.build_serialized_network(network, config)
    else:
        # Legacy API fallback (TensorRT 8.x and older)
        engine = builder.build_engine(network, config)
        serialized_engine = engine.serialize() if engine else None

    build_duration = time.time() - start_time
    if serialized_engine is None:
        raise RuntimeError("[Error] Failed to build serialized TensorRT engine.")

    # Save to disk
    output_dir = os.path.dirname(os.path.abspath(output_engine))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_engine, "wb") as f:
        f.write(serialized_engine)

    engine_size_mb = os.path.getsize(output_engine) / (1024 * 1024)
    print(f"[+] Engine compilation succeeded in {build_duration:.1f}s!")
    print(f"[+] Engine saved to: {output_engine} (Size: {engine_size_mb:.2f} MB)")

    # 6. Verification Pass
    if verify:
        output_name = network.get_output(0).name
        verify_engine(
            engine_file=output_engine,
            input_name=input_name,
            test_shape=opt_shape,
            output_name=output_name,
            device_id=device_id
        )

    return output_engine


def verify_engine(
    engine_file: str,
    input_name: str,
    test_shape: tuple,
    output_name: str,
    device_id: int = 0
):
    print("\n" + "=" * 60)
    print(" [*] Starting TensorRT Engine Execution Verification")
    print("=" * 60)

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_file, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError("[Error] Failed to deserialize compiled TensorRT engine.")

    context = engine.create_execution_context()
    device = torch.device(f"cuda:{device_id}")

    # Set dynamic input shape
    if hasattr(context, "set_input_shape"):
        context.set_input_shape(input_name, test_shape)

    # Determine output shape
    if hasattr(context, "get_tensor_shape"):
        out_shape = tuple(context.get_tensor_shape(output_name))
    else:
        # Fallback for legacy bindings
        out_shape = tuple(engine.get_binding_shape(1))

    print(f"[*] Verification test shapes:")
    print(f"    - Input  ({input_name}):  {test_shape}")
    print(f"    - Output ({output_name}): {out_shape}")

    # Allocate GPU buffers using PyTorch
    d_input = torch.randn(test_shape, device=device, dtype=torch.float32)
    d_output = torch.empty(out_shape, device=device, dtype=torch.float32)

    stream = torch.cuda.Stream(device=device)

    # Execute inference
    if hasattr(context, "execute_async_v3"):
        # Modern TensorRT 10.x / 11.x API
        context.set_tensor_address(input_name, d_input.data_ptr())
        context.set_tensor_address(output_name, d_output.data_ptr())
        context.execute_async_v3(stream.cuda_stream)
    elif hasattr(context, "execute_async_v2"):
        # Legacy TensorRT 8.x API
        bindings = [d_input.data_ptr(), d_output.data_ptr()]
        context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
    else:
        bindings = [d_input.data_ptr(), d_output.data_ptr()]
        context.execute_v2(bindings=bindings)

    stream.synchronize()

    # Validate output
    output_np = d_output.cpu().numpy()
    has_nan = np.isnan(output_np).any()
    has_inf = np.isinf(output_np).any()
    if has_nan or has_inf:
        raise ValueError(f"[Error] Engine output contains NaN ({has_nan}) or Inf ({has_inf}) values!")

    print(f"[+] Engine execution successful!")
    print(f"    - Output min value: {float(d_output.min()):.4f}")
    print(f"    - Output max value: {float(d_output.max()):.4f}")
    print(f"    - Output mean value: {float(d_output.mean()):.4f}")
    print("=" * 60 + "\n")


class TRTInferenceWrapper:
    """
    High-level Python inference helper for deployed TensorRT engines.
    """
    def __init__(self, engine_path: str, device_id: int = 0):
        self.device = torch.device(f"cuda:{device_id}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=self.device)

        # Detect input and output tensor names
        if hasattr(self.engine, "num_io_tensors"):
            self.input_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                                if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
            self.output_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                                 if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        else:
            self.input_names = [self.engine.get_binding_name(0)]
            self.output_names = [self.engine.get_binding_name(1)]

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """
        Runs inference on input numpy array.
        input_array: shape (B, C, H, W) or (C, H, W)
        """
        if input_array.ndim == 3:
            input_array = np.expand_dims(input_array, axis=0)

        in_tensor = torch.from_numpy(input_array.astype(np.float32)).to(self.device).contiguous()
        in_name = self.input_names[0]
        out_name = self.output_names[0]

        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(in_name, tuple(in_tensor.shape))

        if hasattr(self.context, "get_tensor_shape"):
            out_shape = tuple(self.context.get_tensor_shape(out_name))
        else:
            out_shape = tuple(self.engine.get_binding_shape(1))

        out_tensor = torch.empty(out_shape, device=self.device, dtype=torch.float32)

        if hasattr(self.context, "execute_async_v3"):
            self.context.set_tensor_address(in_name, in_tensor.data_ptr())
            self.context.set_tensor_address(out_name, out_tensor.data_ptr())
            self.context.execute_async_v3(self.stream.cuda_stream)
        else:
            bindings = [in_tensor.data_ptr(), out_tensor.data_ptr()]
            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.cuda_stream)

        self.stream.synchronize()
        return out_tensor.cpu().numpy()


if __name__ == "__main__":
    args = parse_args()
    build_engine_from_onnx(
        onnx_file=args.onnx_file,
        output_engine=args.output_engine,
        precision=args.precision,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        workspace_gb=args.workspace_gb,
        device_id=args.device,
        verify=not args.no_verify
    )
