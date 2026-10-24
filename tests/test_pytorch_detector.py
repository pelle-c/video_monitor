"""Check the adapter contract without loading PyTorch in the unit suite."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pytorch_detector import PyTorchDetector


class PyTorchDetectorTests(unittest.TestCase):
    def test_predictions_match_analyzer_interface(self):
        coordinates = object()
        vector = Mock()
        vector.cpu.return_value.numpy.return_value = coordinates
        box = SimpleNamespace(cls=Mock(), conf=Mock(), xyxy=[vector])
        box.cls.item.return_value = 3
        box.conf.item.return_value = 0.91
        model = Mock(names={3: "person"})
        model.predict.return_value = [SimpleNamespace(boxes=[box])]
        torch = Mock()
        yolo = Mock(return_value=model)
        with patch.dict("sys.modules", torch=torch, ultralytics=SimpleNamespace(YOLO=yolo)):
            detector = PyTorchDetector(__file__, threads=1, imgsz=320, iou=0.6)
            frame = object()
            result = detector.detect(frame, confidence=0.5)
        torch.set_num_threads.assert_called_once_with(1)
        model.predict.assert_called_once_with(frame, conf=0.5, iou=0.6,
                                               imgsz=320, device="cpu", verbose=False, save=False)
        self.assertEqual(detector.names[3], "person")
        self.assertEqual((detector.height, detector.width), (320, 320))
        self.assertEqual(result[0].class_id, 3)
        self.assertEqual(result[0].confidence, 0.91)
        self.assertIs(result[0].xyxy, coordinates)

    def test_missing_model_fails_before_backend_can_download(self):
        yolo = Mock()
        with patch.dict("sys.modules", ultralytics=SimpleNamespace(YOLO=yolo)):
            with self.assertRaises(FileNotFoundError):
                PyTorchDetector(Path(__file__).with_name("missing-model.pt"))
        yolo.assert_not_called()
