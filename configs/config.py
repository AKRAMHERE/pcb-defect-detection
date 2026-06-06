"""
config.py
---------
Centralized configuration management for the PCB Defect Detection system.

Supports YAML config file loading with environment variable override.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Default configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "detector": {
        "confidence_threshold": 0.25,
        "iou_threshold":        0.45,
        "device":               "cpu",
        "model_path":           "models/pcb_defect_v1_best.pt",
    },
    "training": {
        "model_variant": "yolov8n.pt",
        "epochs":        50,
        "batch_size":    8,
        "imgsz":         640,
        "workers":       4,
        "optimizer":     "AdamW",
        "lr0":           0.001,
        "patience":      15,
        "amp":           False,
        "cache":         False,
    },
    "video": {
        "inference_every_n":  1,
        "target_display_fps": 30,
        "webcam_resolution":  [1280, 720],
    },
    "output": {
        "images_dir":  "outputs/images",
        "videos_dir":  "outputs/videos",
        "reports_dir": "outputs/reports",
        "models_dir":  "models",
    },
    "reporting": {
        "fail_threshold": 0,
        "report_prefix":  "inspection",
    },
}


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration with priority: YAML file > env vars > defaults.

    Args:
        config_path: Optional path to a YAML config file.

    Returns:
        Merged configuration dictionary.
    """
    config = _deep_copy(DEFAULT_CONFIG)

    if config_path:
        file_config = _load_yaml(config_path)
        config = _deep_merge(config, file_config)

    # Environment variable overrides
    _apply_env_overrides(config)

    return config


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = {**base}
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _deep_copy(d: dict) -> dict:
    import copy
    return copy.deepcopy(d)


def _apply_env_overrides(config: dict):
    """Apply PCB_* environment variables to override config values."""
    env_map = {
        "PCB_MODEL_PATH":       ("detector", "model_path"),
        "PCB_CONF_THRESHOLD":   ("detector", "confidence_threshold"),
        "PCB_IOU_THRESHOLD":    ("detector", "iou_threshold"),
        "PCB_DEVICE":           ("detector", "device"),
        "PCB_EPOCHS":           ("training", "epochs"),
        "PCB_BATCH_SIZE":       ("training", "batch_size"),
        "PCB_FAIL_THRESHOLD":   ("reporting", "fail_threshold"),
    }
    for env_var, (section, key) in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            # Type coercion
            try:
                existing = config[section][key]
                if isinstance(existing, float):
                    config[section][key] = float(val)
                elif isinstance(existing, int):
                    config[section][key] = int(val)
                else:
                    config[section][key] = val
            except (ValueError, KeyError):
                pass


def save_config(config: dict, output_path: str):
    """Persist config to YAML for reproducibility."""
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
