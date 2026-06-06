"""
detector.py
-----------
Core YOLOv8-based PCB defect detection engine.
Handles model loading, inference, result parsing, and performance profiling.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

from .utils import setup_logger, validate_model_path, DetectionResult

logger = setup_logger(__name__)

# PCB defect class labels (matches dataset annotation schema)
PCB_DEFECT_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

# Visualization color map (BGR) per defect class
CLASS_COLORS = {
    "missing_hole":    (0,   165, 255),   # Orange
    "mouse_bite":      (0,   0,   255),   # Red
    "open_circuit":    (255, 0,   0),     # Blue
    "short":           (0,   255, 0),     # Green
    "spur":            (255, 0,   255),   # Magenta
    "spurious_copper": (0,   255, 255),   # Yellow
}
DEFAULT_COLOR = (128, 128, 128)


class PCBDefectDetector:
    """
    YOLOv8n inference engine for PCB defect detection.

    Designed for CPU-only deployment in manufacturing QC pipelines.
    Supports single-image, batch, video-frame, and webcam inference modes.

    Args:
        model_path (str | Path): Path to trained YOLOv8 .pt weights file.
        confidence_threshold (float): Minimum confidence score to retain detections.
        iou_threshold (float): IoU threshold for NMS suppression.
        device (str): Inference device — 'cpu' for laptop/edge deployment.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Performance counters
        self._inference_times: list[float] = []
        self._frame_count: int = 0

        self.model = self._load_model()
        self.class_names = self._resolve_class_names()
        logger.info(
            f"PCBDefectDetector initialized | model={self.model_path.name} "
            f"| conf={confidence_threshold} | device={device}"
        )

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------

    def _load_model(self) -> YOLO:
        """Load YOLOv8 weights with validation and fallback handling."""
        if not validate_model_path(self.model_path):
            logger.warning(
                f"Model not found at {self.model_path}. "
                "Falling back to pretrained YOLOv8n for architecture validation."
            )
            return YOLO("yolov8n.pt")

        logger.info(f"Loading model weights from: {self.model_path}")
        model = YOLO(str(self.model_path))
        model.to(self.device)
        return model

    def _resolve_class_names(self) -> dict[int, str]:
        """Extract class index→name mapping from loaded model."""
        if hasattr(self.model, "names") and self.model.names:
            return self.model.names
        # Fallback to known PCB dataset schema
        return {i: name for i, name in enumerate(PCB_DEFECT_CLASSES)}

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        return_annotated: bool = True,
    ) -> tuple[list[DetectionResult], Optional[np.ndarray], float]:
        """
        Run defect detection on a single BGR frame.

        Args:
            frame: Input image as NumPy BGR array (H, W, 3).
            return_annotated: Whether to render bounding boxes onto a copy.

        Returns:
            Tuple of:
                - List of DetectionResult objects
                - Annotated frame (or None if return_annotated=False)
                - Inference latency in milliseconds
        """
        t_start = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        inference_ms = (time.perf_counter() - t_start) * 1000
        self._inference_times.append(inference_ms)
        self._frame_count += 1

        detections = self._parse_results(results)
        annotated = self._annotate_frame(frame.copy(), detections) if return_annotated else None

        return detections, annotated, inference_ms

    def _parse_results(self, results) -> list[DetectionResult]:
        """Convert Ultralytics Results objects to structured DetectionResult list."""
        detections: list[DetectionResult] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes  = result.boxes.xyxy.cpu().numpy()    # [x1, y1, x2, y2]
            confs  = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                class_name = self.class_names.get(cls_id, f"class_{cls_id}")
                detections.append(
                    DetectionResult(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=float(conf),
                        bbox=tuple(box.tolist()),  # (x1, y1, x2, y2)
                    )
                )

        return detections

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[DetectionResult],
    ) -> np.ndarray:
        """
        Render bounding boxes, class labels, and confidence scores onto frame.

        Uses per-class color coding for rapid visual triage.
        """
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)
            label = f"{det.class_name} {det.confidence:.2f}"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

            # Label background
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)

            # Label text
            cv2.putText(
                frame, label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), thickness=1, lineType=cv2.LINE_AA,
            )

        return frame

    def overlay_stats(
        self,
        frame: np.ndarray,
        fps: float,
        defect_count: int,
        inference_ms: float,
    ) -> np.ndarray:
        """
        Overlay HUD with real-time performance metrics on the frame.

        Mimics industrial machine-vision monitoring displays.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent HUD background
        cv2.rectangle(overlay, (0, 0), (260, 85), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        lines = [
            f"FPS: {fps:.1f}",
            f"Inference: {inference_ms:.1f} ms",
            f"Defects: {defect_count}",
            f"Threshold: {self.confidence_threshold:.2f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                frame, line,
                (8, 20 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 255, 180), 1, cv2.LINE_AA,
            )

        return frame

    # ------------------------------------------------------------------
    # Performance profiling
    # ------------------------------------------------------------------

    def get_performance_stats(self) -> dict:
        """Return aggregated inference performance metrics."""
        if not self._inference_times:
            return {}

        times = np.array(self._inference_times)
        return {
            "total_frames":        self._frame_count,
            "avg_inference_ms":    float(np.mean(times)),
            "p95_inference_ms":    float(np.percentile(times, 95)),
            "max_inference_ms":    float(np.max(times)),
            "min_inference_ms":    float(np.min(times)),
            "avg_fps":             1000.0 / float(np.mean(times)),
            "device":              self.device,
            "model":               self.model_path.name,
        }

    def reset_stats(self):
        """Reset performance counters (call between benchmark runs)."""
        self._inference_times.clear()
        self._frame_count = 0

    def set_confidence_threshold(self, threshold: float):
        """Dynamically update confidence threshold (useful for live tuning)."""
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Confidence threshold must be in (0, 1), got {threshold}")
        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold updated → {threshold}")
