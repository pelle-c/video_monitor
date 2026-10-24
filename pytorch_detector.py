"""CPU Ultralytics adapter for the video analyzer's detection interface."""

from pathlib import Path
from types import SimpleNamespace


class PyTorchDetector:
    def __init__(self, model_path, threads=1, imgsz=None, iou=0.7):
        # Fail locally rather than letting Ultralytics download a missing model.
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if threads < 1:
            raise ValueError("threads must be positive")
        import torch
        torch.set_num_threads(threads)
        from ultralytics import YOLO
        self.model = YOLO(str(model_path), task="detect")
        self.names = self.model.names
        self.height = self.width = imgsz or 640
        self.iou = iou

    def detect(self, frame, confidence=0.45):
        results = self.model.predict(frame, conf=confidence, iou=self.iou,
                                     imgsz=self.height, device="cpu", verbose=False,
                                     save=False)
        return [SimpleNamespace(class_id=int(box.cls.item()),
                                confidence=float(box.conf.item()),
                                xyxy=box.xyxy[0].cpu().numpy())
                for box in results[0].boxes]
