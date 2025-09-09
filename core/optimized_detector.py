import logging
import time
from typing import List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config.config import settings

logger = logging.getLogger(__name__)


class OptimizedVehicleDetector:
    """YOLO-based vehicle detector used by the optimized processor for counting.

    Returns detections as [x1, y1, x2, y2, conf, cls] matching the legacy detector.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        self.device = self._select_device(device)
        self.model_path = model_path or settings.get('vehicle_model_path')
        self.enabled_classes = settings.get('vehicle_classes', [2, 3, 5, 7])

        self.model = self._load_model()

        # Performance tracking
        self.inference_times: List[float] = []
        self.max_inference_history = 100

        logger.info(f"OptimizedVehicleDetector initialized on {self.device}")

    def _select_device(self, device: str) -> str:
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            else:
                device = 'cpu'
        return device

    def _load_model(self) -> YOLO:
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            model.model.eval()
            if self.device == 'cuda':
                model.model.half()
            logger.info(f"YOLO model loaded from {self.model_path} on {self.device}")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect_vehicles(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        max_detections: int = 50,
    ) -> List[List[float]]:
        """Run YOLO inference and return [x1, y1, x2, y2, conf, cls] detections."""
        try:
            # Timing
            if self.device == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                cpu_start = time.perf_counter()

            results = self.model(
                frame,
                conf=conf_threshold,
                iou=iou_threshold,
                classes=self.enabled_classes,
                max_det=max_detections,
                verbose=False,
            )

            # Timing end
            if self.device == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                inference_ms = start_event.elapsed_time(end_event)
            else:
                inference_ms = (time.perf_counter() - cpu_start) * 1000.0
            self._update_inference_times(inference_ms)

            detections: List[List[float]] = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    conf = boxes.conf.detach().cpu().numpy()
                    cls = boxes.cls.detach().cpu().numpy().astype(int)
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = map(float, xyxy[i])
                        detections.append([x1, y1, x2, y2, float(conf[i]), int(cls[i])])

            return detections
        except Exception as e:
            logger.error(f"Error in vehicle detection: {e}")
            return []

    def _update_inference_times(self, inference_time_ms: float) -> None:
        self.inference_times.append(inference_time_ms)
        if len(self.inference_times) > self.max_inference_history:
            self.inference_times.pop(0)

    def get_average_inference_time(self) -> float:
        if not self.inference_times:
            return 0.0
        return float(sum(self.inference_times) / len(self.inference_times))

    def get_performance_stats(self) -> dict:
        avg_time = self.get_average_inference_time()
        return {
            'average_inference_time_ms': avg_time,
            'fps': (1000.0 / avg_time) if avg_time > 0 else 0.0,
            'total_inferences': len(self.inference_times),
            'device': self.device,
        }

    def cleanup(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        logger.info("OptimizedVehicleDetector cleanup completed")
