import logging
import os
from typing import Optional, Tuple
import torch

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
except ImportError:
    trt = None


class TRTDetectorModule:
    """
    Zero-copy TensorRT inference wrapper compatible with PyTorch CUDA tensors.
    Supports TensorRT 8.x, 10.x, and 11.x execution APIs without CPU round-trips.
    """

    def __init__(self, engine_path: str, device: torch.device):
        if trt is None:
            raise ImportError(
                "tensorrt Python package is not installed. Please run: pip install tensorrt"
            )

        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found at: {engine_path}")

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=self.device)

        # Discover tensor bindings and expected spatial shapes
        if hasattr(self.engine, "num_io_tensors"):
            # Modern TensorRT 8.5+ / 10.x / 11.x API
            self.input_names = [
                self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)
                if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
                == trt.TensorIOMode.INPUT
            ]
            self.output_names = [
                self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)
                if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
                == trt.TensorIOMode.OUTPUT
            ]
            in_shape = tuple(self.engine.get_tensor_shape(self.input_names[0]))
            out_shape = tuple(self.engine.get_tensor_shape(self.output_names[0]))
        else:
            # Legacy TensorRT 8.x binding API
            self.input_names = [self.engine.get_binding_name(0)]
            self.output_names = [self.engine.get_binding_name(1)]
            in_shape = tuple(self.engine.get_binding_shape(0))
            out_shape = tuple(self.engine.get_binding_shape(1))

        self.input_name = self.input_names[0]
        self.output_name = self.output_names[0]

        # Extract expected spatial shape (e.g. 192, 192)
        self.expected_h = in_shape[-2] if in_shape[-2] > 0 else 192
        self.expected_w = in_shape[-1] if in_shape[-1] > 0 else 192
        self.num_classes = (
            out_shape[1] if len(out_shape) >= 2 and out_shape[1] > 0 else 4
        )

        logger.info(
            f"Initialized TRTDetectorModule: input='{self.input_name}' {in_shape}, "
            f"target_hw=({self.expected_h}, {self.expected_w}), classes={self.num_classes}"
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes inference directly on GPU tensors without host round-trips.

        Args:
            x: torch.Tensor of shape (B, 1, H, W) on self.device.

        Returns:
            out_tensor: torch.Tensor of shape (B, num_classes, H, W) on self.device.
        """
        if not x.is_cuda:
            x = x.to(self.device)
        x = x.contiguous()

        B = x.shape[0]
        actual_input_shape = (B, 1, self.expected_h, self.expected_w)

        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(self.input_name, actual_input_shape)

        out_shape = (B, self.num_classes, self.expected_h, self.expected_w)
        out_tensor = torch.empty(out_shape, device=self.device, dtype=torch.float32)

        # Execute using direct CUDA memory pointers
        if hasattr(self.context, "execute_async_v3"):
            self.context.set_tensor_address(self.input_name, x.data_ptr())
            self.context.set_tensor_address(self.output_name, out_tensor.data_ptr())
            self.context.execute_async_v3(self.stream.cuda_stream)
        else:
            bindings = [x.data_ptr(), out_tensor.data_ptr()]
            self.context.execute_async_v2(
                bindings=bindings, stream_handle=self.stream.cuda_stream
            )

        self.stream.synchronize()
        return out_tensor
