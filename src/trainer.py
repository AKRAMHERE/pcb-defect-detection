"""
trainer.py
----------
YOLOv8n training orchestrator for PCB defect detection.

Handles:
- Dataset validation and YAML config generation
- Training with CPU-optimized hyperparameters
- Post-training evaluation (mAP, precision, recall)
- Model export and weight management
"""

import shutil
from pathlib import Path
from typing import Optional, Union

import yaml
from ultralytics import YOLO

from .utils import setup_logger, ensure_dir

logger = setup_logger(__name__)

# Default CPU-optimized training configuration
DEFAULT_TRAIN_CONFIG = {
    "epochs":       50,
    "imgsz":        640,
    "batch":        8,           # Reduced for CPU memory constraints
    "workers":      4,
    "optimizer":    "AdamW",
    "lr0":          0.001,
    "lrf":          0.01,
    "momentum":     0.937,
    "weight_decay": 0.0005,
    "warmup_epochs":3,
    "box":          7.5,         # Box loss gain
    "cls":          0.5,         # Classification loss gain
    "dfl":          1.5,         # DFL loss gain
    "mosaic":       1.0,         # Mosaic augmentation
    "mixup":        0.0,         # Disabled for small datasets
    "copy_paste":   0.0,
    "degrees":      10.0,        # Rotation augmentation (±10°)
    "flipud":       0.1,
    "fliplr":       0.5,
    "hsv_h":        0.015,
    "hsv_s":        0.7,
    "hsv_v":        0.4,
    "cache":        False,       # Disable caching on CPU
    "amp":          False,       # Disable AMP (CPU-only)
    "device":       "cpu",
    "patience":     15,          # Early stopping patience
    "exist_ok":     True,
    "pretrained":   True,
    "verbose":      True,
}

PCB_CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


class PCBTrainer:
    """
    Training pipeline for YOLOv8n on PCB defect datasets.

    Designed to run end-to-end on CPU hardware while achieving
    production-grade mAP for 6-class defect classification.

    Args:
        dataset_dir:   Root directory containing train/val/test splits.
        output_dir:    Directory to store training runs and exported weights.
        model_variant: YOLOv8 variant — 'yolov8n' recommended for CPU.
        config:        Optional dict to override default training hyperparameters.
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        output_dir: Union[str, Path] = "models",
        model_variant: str = "yolov8n.pt",
        config: Optional[dict] = None,
    ):
        self.dataset_dir  = Path(dataset_dir).resolve()
        self.output_dir   = ensure_dir(output_dir)
        self.model_variant = model_variant
        self.config = {**DEFAULT_TRAIN_CONFIG, **(config or {})}

        self._validate_dataset_structure()

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    def _validate_dataset_structure(self):
        """Assert expected YOLO dataset directory layout."""
        required = ["train/images", "valid/images"]
        missing = [d for d in required if not (self.dataset_dir / d).exists()]
        if missing:
            raise FileNotFoundError(
                f"Dataset missing required directories: {missing}\n"
                f"Expected layout under: {self.dataset_dir}\n"
                "  train/images/  train/labels/\n"
                "  valid/images/  valid/labels/\n"
                "  test/images/   test/labels/   (optional)"
            )
        logger.info(f"Dataset structure validated at: {self.dataset_dir}")

    def generate_dataset_yaml(self, output_path: Optional[Path] = None) -> Path:
        """
        Auto-generate dataset.yaml from directory structure.

        Resolves class names from existing labels or falls back to
        canonical PCB defect schema. Writes YAML to dataset root.
        """
        yaml_path = output_path or (self.dataset_dir / "dataset.yaml")

        # Count detected classes from label files
        class_names = self._infer_class_names()

        config = {
            "path": str(self.dataset_dir),
            "train": "train/images",
            "val":   "valid/images",
            "nc":    len(class_names),
            "names": class_names,
        }

        # Include test split if present
        test_dir = self.dataset_dir / "test" / "images"
        if test_dir.exists():
            config["test"] = "test/images"

        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Dataset YAML written: {yaml_path}")
        logger.info(f"  Classes ({len(class_names)}): {class_names}")
        return yaml_path

    def _infer_class_names(self) -> list[str]:
        """
        Scan label files to determine class count, then map to known names.

        Falls back to canonical PCB_CLASS_NAMES if class count matches.
        """
        label_dirs = [
            self.dataset_dir / "train" / "labels",
            self.dataset_dir / "valid" / "labels",
        ]
        max_class_id = -1
        for label_dir in label_dirs:
            if not label_dir.exists():
                continue
            for label_file in label_dir.glob("*.txt"):
                try:
                    content = label_file.read_text().strip()
                    for line in content.splitlines():
                        parts = line.split()
                        if parts:
                            max_class_id = max(max_class_id, int(parts[0]))
                except (ValueError, IOError):
                    continue

        num_classes = max_class_id + 1 if max_class_id >= 0 else len(PCB_CLASS_NAMES)

        if num_classes == len(PCB_CLASS_NAMES):
            return PCB_CLASS_NAMES

        logger.warning(
            f"Detected {num_classes} classes, expected {len(PCB_CLASS_NAMES)}. "
            "Using generic class names."
        )
        return [f"defect_{i}" for i in range(num_classes)]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, run_name: str = "pcb_defect_v1") -> Path:
        """
        Execute YOLOv8n training run.

        Args:
            run_name: Identifier for this training experiment.

        Returns:
            Path to best weights file (best.pt).
        """
        yaml_path = self.generate_dataset_yaml()
        model = YOLO(self.model_variant)

        logger.info("=" * 60)
        logger.info("Starting PCB Defect Detection Training")
        logger.info(f"  Model:   {self.model_variant}")
        logger.info(f"  Dataset: {yaml_path}")
        logger.info(f"  Epochs:  {self.config['epochs']}")
        logger.info(f"  Device:  {self.config['device']}")
        logger.info(f"  ImgSz:   {self.config['imgsz']}")
        logger.info("=" * 60)

        results = model.train(
            data=str(yaml_path),
            name=run_name,
            project=str(self.output_dir / "runs"),
            **{k: v for k, v in self.config.items()},
        )

        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        if best_weights.exists():
            # Copy best weights to top-level models directory for easy access
            dest = self.output_dir / f"{run_name}_best.pt"
            shutil.copy2(best_weights, dest)
            logger.info(f"Best weights exported → {dest}")
        else:
            logger.warning("best.pt not found in run directory.")
            best_weights = Path(results.save_dir) / "weights" / "last.pt"

        return best_weights

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        weights_path: Union[str, Path],
        split: str = "val",
    ) -> dict:
        """
        Run YOLOv8 evaluation on val or test split.

        Args:
            weights_path: Path to .pt weights to evaluate.
            split:        Dataset split — 'val' or 'test'.

        Returns:
            Dict containing mAP50, mAP50-95, precision, recall per class.
        """
        yaml_path = self.dataset_dir / "dataset.yaml"
        if not yaml_path.exists():
            yaml_path = self.generate_dataset_yaml()

        model = YOLO(str(weights_path))
        metrics = model.val(
            data=str(yaml_path),
            split=split,
            device=self.config["device"],
            imgsz=self.config["imgsz"],
            verbose=True,
        )

        results = {
            "mAP50":      float(metrics.box.map50),
            "mAP50_95":   float(metrics.box.map),
            "precision":  float(metrics.box.mp),
            "recall":     float(metrics.box.mr),
        }

        logger.info("Evaluation Results:")
        for k, v in results.items():
            logger.info(f"  {k}: {v:.4f}")

        return results

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_model(
        self,
        weights_path: Union[str, Path],
        format: str = "onnx",
    ) -> Path:
        """
        Export trained model to deployment format.

        Supported: 'onnx', 'torchscript', 'openvino'
        ONNX recommended for CPU edge deployment.
        """
        model = YOLO(str(weights_path))
        export_path = model.export(
            format=format,
            imgsz=self.config["imgsz"],
            device=self.config["device"],
        )
        logger.info(f"Model exported to {format.upper()}: {export_path}")
        return Path(export_path)
