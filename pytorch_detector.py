"""Ultralytics adapter for PyTorch and LiteRT models."""

import importlib
from pathlib import Path
from types import SimpleNamespace


def _set_edgetpu_fallback_threads(threads):
    """Pass the requested CPU thread count to TFLite's Edge TPU backend."""
    interpreter_module = importlib.import_module("tflite_runtime.interpreter")
    original = getattr(
        interpreter_module,
        "_video_analyzer_original_interpreter",
        interpreter_module.Interpreter,
    )
    interpreter_module._video_analyzer_original_interpreter = original

    def interpreter_with_threads(*args, **kwargs):
        kwargs.setdefault("num_threads", threads)
        return original(*args, **kwargs)

    # Ultralytics' Edge TPU backend imports Interpreter from this module when
    # it loads the model, but currently doesn't pass num_threads itself.
    interpreter_module.Interpreter = interpreter_with_threads


class PyTorchDetector:
    def __init__(self, model_path, threads=1, imgsz=None, iou=0.7, data_path=None):
        # Fail locally rather than letting Ultralytics download a missing model.
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if threads < 1:
            raise ValueError("threads must be positive")
        import torch
        torch.set_num_threads(threads)
        if "edgetpu" in Path(model_path).name.lower():
            _set_edgetpu_fallback_threads(threads)
        from ultralytics import YOLO
        self.model = YOLO(str(model_path), task="detect")
        self.names = self.model.names
        self.height = self.width = imgsz or 640
        self.iou = iou
        self.data_path = data_path

    def detect(self, frame, confidence=0.45):
        options = {}
        if self.data_path:
            options["data"] = self.data_path
        results = self.model.predict(frame, conf=confidence, iou=self.iou,
                                     imgsz=self.height, device="cpu", verbose=False,
                                     save=False, **options)
        return [SimpleNamespace(class_id=int(box.cls.item()),
                                confidence=float(box.conf.item()),
                                xyxy=box.xyxy[0].cpu().numpy())
                for box in results[0].boxes]
