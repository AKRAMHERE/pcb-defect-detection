#!/usr/bin/env python3
"""
scripts/download_dataset.py
---------------------------
Downloads and prepares the PCB Defect Dataset from Roboflow Universe.

Requires:
    pip install roboflow

Usage:
    python scripts/download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

Dataset: PCB Defect Detection (DeepPCB-based dataset)
Source:  https://universe.roboflow.com
"""

import argparse
import sys
from pathlib import Path


def download_roboflow(api_key: str, output_dir: str = "data/pcb_dataset"):
    """Download dataset using Roboflow API."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed.")
        print("       Run: pip install roboflow")
        sys.exit(1)

    print(f"Connecting to Roboflow with API key: {api_key[:8]}***")

    rf = Roboflow(api_key=api_key)

    # Primary dataset option: PCB Defect Detection by Roboflow
    # https://universe.roboflow.com/roboflow-100/pcb-defect-detection
    project = rf.workspace("roboflow-100").project("pcb-defect-detection")
    dataset = project.version(1).download("yolov8", location=output_dir)

    print(f"\nDataset downloaded to: {output_dir}")
    print("Directory structure:")
    for p in sorted(Path(output_dir).rglob("*")):
        if p.is_dir():
            print(f"  {p.relative_to(output_dir)}/")

    return dataset


def create_sample_structure(output_dir: str = "data/pcb_dataset"):
    """
    Create empty dataset directory structure for manual dataset placement.
    
    Use this if you downloaded the dataset manually from Roboflow Universe.
    """
    base = Path(output_dir)
    dirs = [
        "train/images", "train/labels",
        "valid/images", "valid/labels",
        "test/images",  "test/labels",
    ]
    for d in dirs:
        (base / d).mkdir(parents=True, exist_ok=True)

    # Write placeholder dataset.yaml
    yaml_content = """# PCB Defect Detection Dataset
# Place images in train/images/ and labels in train/labels/
# Label format: YOLO (class_id cx cy w h) normalized

path: {path}
train: train/images
val: valid/images
test: test/images

nc: 6
names:
  - missing_hole
  - mouse_bite
  - open_circuit
  - short
  - spur
  - spurious_copper
""".format(path=str(base.resolve()))

    (base / "dataset.yaml").write_text(yaml_content)
    print(f"Dataset directory structure created at: {output_dir}")
    print("Place your dataset images and YOLO-format labels in the appropriate folders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download PCB Defect Dataset from Roboflow"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Roboflow API key (get from https://app.roboflow.com/settings/api)"
    )
    parser.add_argument(
        "--output", type=str, default="data/pcb_dataset",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--scaffold-only", action="store_true",
        help="Only create directory structure (for manual dataset placement)"
    )
    args = parser.parse_args()

    if args.scaffold_only or not args.api_key:
        create_sample_structure(args.output)
    else:
        download_roboflow(args.api_key, args.output)
