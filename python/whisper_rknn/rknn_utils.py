"""Utilities for loading and inspecting ONNX and RKNN models."""

import onnxruntime
from rknn.api import RKNN


def load_model(model_path, target=None, device_id=None):
    """Load an ONNX model or initialise an RKNN model for inference."""
    if model_path.endswith(".onnx"):
        return onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
    if not model_path.endswith(".rknn"):
        raise ValueError(f"Unsupported model file: {model_path}")

    model = RKNN()

    print("--> Loading model")
    result = model.load_rknn(model_path)
    if result != 0:
        model.release()
        raise RuntimeError(
            f'Loading RKNN model "{model_path}" failed with code {result}'
        )
    print("done")

    print("--> Init runtime environment")
    result = model.init_runtime(target=target, device_id=device_id)
    if result != 0:
        model.release()
        raise RuntimeError(
            f"Initialising RKNN runtime failed with code {result}"
        )
    print("done")
    return model


def model_input_shapes(model):
    """Return a mapping from model input names to their fixed shapes."""
    if isinstance(model, onnxruntime.InferenceSession):
        return {input_.name: input_.shape for input_ in model.get_inputs()}

    if isinstance(model, RKNN):
        metadata = model.rknn_base.inputs_meta["attrs"]
        return {
            name: attributes["shape"]
            for name, attributes in metadata.items()
            if not attributes["is_output"]
        }

    raise TypeError(f"Unsupported model: {type(model)}")


def release_model(model):
    """Release resources owned by a loaded model where required."""
    if isinstance(model, RKNN):
        model.release()
    elif not isinstance(model, onnxruntime.InferenceSession):
        raise TypeError(f"Unsupported model: {type(model)}")
