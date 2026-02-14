from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PROJECT_ROOT / "weights"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
INPUT_IMAGES_DIR = PROJECT_ROOT / "data" / "images"

# Phase 1 defaults
TOP1_ONLY = True
MIN_CROP_AREA = 5_000
DETECT_CONF_THRESHOLD = 0.01

# Class mapping defaults (can be overridden via CLI)
TRUCK_CLASS_IDS: list[int] = []
BED_CLASS_IDS: list[int] = []
BED_CLASS_NAMES = [
    "box",
    "bed",
    "truck bed",
    "truck_bed",
    "cargo bed",
    "cargo_area",
]
TRUCK_FALLBACK_CLASS_NAMES = [
    "truck",
    "lorry",
    "pickup",
    "vehicle",
]

# Phase 2 defaults
PHASE1_CROPS_DIR = OUTPUT_DIR / "phase1_crops"
PHASE2_OUTPUT_DIR = OUTPUT_DIR / "phase2_classification"
PHASE2_SUMMARY_CSV = OUTPUT_DIR / "phase2_summary.csv"
PHASE2_IMAGE_GATE_CSV = OUTPUT_DIR / "phase2_image_gate.csv"

# Phase 3 defaults
PHASE3_OUTPUT_DIR = OUTPUT_DIR / "phase3_classification"
PHASE3_SUMMARY_CSV = OUTPUT_DIR / "phase3_summary.csv"
PHASE3_IMAGE_GATE_CSV = OUTPUT_DIR / "phase3_image_gate.csv"

# Phase 4 defaults
PHASE4_OUTPUT_DIR = OUTPUT_DIR / "phase4_classification"
PHASE4_SUMMARY_CSV = OUTPUT_DIR / "phase4_summary.csv"
PHASE4_IMAGE_GATE_CSV = OUTPUT_DIR / "phase4_image_gate.csv"

# Phase 5 defaults
PHASE5_OUTPUT_DIR = OUTPUT_DIR / "phase5_segmentation"
PHASE5_MASKS_DIR = OUTPUT_DIR / "phase5_masks"
PHASE5_SUMMARY_CSV = OUTPUT_DIR / "phase5_summary.csv"
PHASE5_IMAGE_GATE_CSV = OUTPUT_DIR / "phase5_image_gate.csv"
PHASE5_SEG_CONF_THRESHOLD = 0.25

# Stream-event defaults
STREAM_EVENTS_JSONL = OUTPUT_DIR / "events.jsonl"
STREAM_EVENTS_IMAGES_DIR = OUTPUT_DIR / "events_images"
MISSED_M = 15
IOU_THRESHOLD = 0.25
MERGE_WINDOW_FRAMES = 10
MERGE_IOU_THRESHOLD = 0.2
MERGE_MAX_CENTER_DIST_RATIO = 0.15
EDGE_GUARD = True
EDGE_MARGIN = 80
EXPAND_BED_TO_TRUCK_X = 1.25
EXPAND_BED_TO_TRUCK_Y = 1.60

STREAM_IOU_THRESHOLD = IOU_THRESHOLD
STREAM_MISSED_M = MISSED_M
STREAM_MERGE_WINDOW = MERGE_WINDOW_FRAMES
STREAM_MERGE_IOU = MERGE_IOU_THRESHOLD
STREAM_MERGE_CENTER_RATIO = MERGE_MAX_CENTER_DIST_RATIO
STREAM_EDGE_GUARD = EDGE_GUARD
STREAM_EDGE_MARGIN = EDGE_MARGIN
STREAM_EVERY_N = 2
STREAM_EVENT_INFER_MODE = "finalize"
STREAM_TOP2 = False
STREAM_MIN_BEST_AREA = 10_000
STREAM_STABLE_FRAMES = 3
STREAM_SCORE_AREA_WEIGHT = 0.7
STREAM_SCORE_CONF_WEIGHT = 0.3
STREAM_SCORE_CENTER_WEIGHT = 0.05
STREAM_MIN_EVENT_HITS = 3
STREAM_MIN_BED_PERSIST_FRAMES = 10
STREAM_VOTE_ENABLE = False
STREAM_VOTE_EVERY_N_FRAMES = 5
STREAM_VOTE_MAX_SAMPLES = 80
STREAM_TRACK_SMOOTH_ALPHA = 0.20
STREAM_TRACK_DEADBAND_PX = 4.0
STREAM_TRACK_MAX_STEP_PX = 12.0
STREAM_MAX_DETECT_FPS = 0.0

# Class mapping for coverage decision.
PHASE2_COVERED_CLASS_NAMES = [
    "covered",
    "fully covered",
    "fully_covered",
    "full_cover",
]
PHASE2_UNCOVERED_CLASS_NAMES = [
    "uncovered_or_partial",
    "uncovered",
    "partially covered",
    "partially_covered",
    "partial",
]

# Class mapping for regularity decision.
PHASE3_IRREGULAR_CLASS_NAMES = [
    "irregular",
    "not_regular",
]
PHASE3_REGULAR_CLASS_NAMES = [
    "regular",
]

# Class mapping for irregular subtype (Phase 4).
PHASE4_SANDLIKE_CLASS_NAMES = [
    "sand_like",
    "sand like",
    "covered_sand_like",
]
PHASE4_IRONBARS_CLASS_NAMES = [
    "iron_bars_like",
    "iron bars like",
    "covered_iron_bars_like",
]
PHASE4_UNKNOWN_CLASS_NAMES = [
    "unknown",
    "covered_irregular",
    "irregular",
    "other",
]

# Heuristic bed crop from a truck box, if no direct bed class is detected.
# Coordinates are relative to the detected truck box.
HEURISTIC_X_START_RATIO = 0.30
HEURISTIC_X_END_RATIO = 0.98
HEURISTIC_Y_START_RATIO = 0.15
HEURISTIC_Y_END_RATIO = 0.85


def _score_detect_candidate(path: Path) -> int:
    lower = str(path).lower()
    score = 0
    if "bed_detection" in lower:
        score += 12
    if "bed" in lower:
        score += 6
    if "detect" in lower or "detection" in lower:
        score += 4
    if path.name.lower() == "best.pt":
        score += 2
    if "class" in lower:
        score -= 6
    if "seg" in lower:
        score -= 6
    return score


def _auto_discover_detect_model_path() -> Path | None:
    if not WEIGHTS_DIR.exists():
        return None

    pt_files = sorted(WEIGHTS_DIR.rglob("*.pt"))
    if not pt_files:
        return None

    ranked = sorted(
        pt_files,
        key=lambda p: (_score_detect_candidate(p), str(p).lower()),
        reverse=True,
    )
    return ranked[0]


def _score_cls1_candidate(path: Path) -> int:
    lower = str(path).lower()
    score = 0
    if "classification#1" in lower or "classification1" in lower:
        score += 12
    if "cls1" in lower:
        score += 8
    if "cover" in lower or "coverage" in lower:
        score += 6
    if "class" in lower:
        score += 3
    if path.name.lower() == "best.pt":
        score += 2
    if "seg" in lower:
        score -= 5
    if "detect" in lower or "detection" in lower:
        score -= 5
    return score


def _auto_discover_cls1_model_path() -> Path | None:
    if not WEIGHTS_DIR.exists():
        return None

    pt_files = sorted(WEIGHTS_DIR.rglob("*.pt"))
    if not pt_files:
        return None

    ranked = sorted(
        pt_files,
        key=lambda p: (_score_cls1_candidate(p), str(p).lower()),
        reverse=True,
    )
    return ranked[0]


def _score_cls2_candidate(path: Path) -> int:
    lower = str(path).lower()
    score = 0
    if "classification#2" in lower or "classification2" in lower:
        score += 12
    if "cls2" in lower:
        score += 8
    if "shape" in lower:
        score += 6
    if "regular" in lower or "irregular" in lower:
        score += 6
    if "class" in lower:
        score += 3
    if path.name.lower() == "best.pt":
        score += 2
    if "seg" in lower:
        score -= 5
    if "detect" in lower or "detection" in lower:
        score -= 5
    return score


def _auto_discover_cls2_model_path() -> Path | None:
    if not WEIGHTS_DIR.exists():
        return None

    pt_files = sorted(WEIGHTS_DIR.rglob("*.pt"))
    if not pt_files:
        return None

    ranked = sorted(
        pt_files,
        key=lambda p: (_score_cls2_candidate(p), str(p).lower()),
        reverse=True,
    )
    return ranked[0]


def _score_cls3_candidate(path: Path) -> int:
    lower = str(path).lower()
    score = 0
    if "classification#3" in lower or "classification3" in lower:
        score += 12
    if "cls3" in lower:
        score += 8
    if "irregular_type" in lower or "irregulartype" in lower:
        score += 6
    if "sand" in lower or "iron" in lower or "bars" in lower:
        score += 5
    if "class" in lower:
        score += 3
    if path.name.lower() == "best.pt":
        score += 2
    if "seg" in lower:
        score -= 5
    if "detect" in lower or "detection" in lower:
        score -= 5
    return score


def _auto_discover_cls3_model_path() -> Path | None:
    if not WEIGHTS_DIR.exists():
        return None

    pt_files = sorted(WEIGHTS_DIR.rglob("*.pt"))
    if not pt_files:
        return None

    ranked = sorted(
        pt_files,
        key=lambda p: (_score_cls3_candidate(p), str(p).lower()),
        reverse=True,
    )
    return ranked[0]


def _score_seg_candidate(path: Path) -> int:
    lower = str(path).lower()
    score = 0
    if "segmentation" in lower or "seg" in lower:
        score += 12
    if "yolo_segmentation" in lower:
        score += 10
    if "load" in lower:
        score += 6
    if path.name.lower() == "best.pt":
        score += 2
    if "class" in lower:
        score -= 5
    if "detect" in lower or "detection" in lower:
        score -= 5
    return score


def _auto_discover_seg_model_path() -> Path | None:
    if not WEIGHTS_DIR.exists():
        return None

    pt_files = sorted(WEIGHTS_DIR.rglob("*.pt"))
    if not pt_files:
        return None

    ranked = sorted(
        pt_files,
        key=lambda p: (_score_seg_candidate(p), str(p).lower()),
        reverse=True,
    )
    return ranked[0]


_env_detect_model = os.getenv("DETECT_MODEL_PATH")
if _env_detect_model:
    _candidate = Path(_env_detect_model).expanduser()
    if not _candidate.is_absolute():
        _candidate = PROJECT_ROOT / _candidate
    DETECT_MODEL_PATH = _candidate.resolve()
else:
    DETECT_MODEL_PATH = _auto_discover_detect_model_path()

_env_cls1_model = os.getenv("CLS1_MODEL_PATH")
if _env_cls1_model:
    _candidate = Path(_env_cls1_model).expanduser()
    if not _candidate.is_absolute():
        _candidate = PROJECT_ROOT / _candidate
    CLS1_MODEL_PATH = _candidate.resolve()
else:
    CLS1_MODEL_PATH = _auto_discover_cls1_model_path()

_env_cls2_model = os.getenv("CLS2_MODEL_PATH")
if _env_cls2_model:
    _candidate = Path(_env_cls2_model).expanduser()
    if not _candidate.is_absolute():
        _candidate = PROJECT_ROOT / _candidate
    CLS2_MODEL_PATH = _candidate.resolve()
else:
    CLS2_MODEL_PATH = _auto_discover_cls2_model_path()

_env_cls3_model = os.getenv("CLS3_MODEL_PATH")
if _env_cls3_model:
    _candidate = Path(_env_cls3_model).expanduser()
    if not _candidate.is_absolute():
        _candidate = PROJECT_ROOT / _candidate
    CLS3_MODEL_PATH = _candidate.resolve()
else:
    CLS3_MODEL_PATH = _auto_discover_cls3_model_path()

_env_seg_model = os.getenv("SEG_MODEL_PATH")
if _env_seg_model:
    _candidate = Path(_env_seg_model).expanduser()
    if not _candidate.is_absolute():
        _candidate = PROJECT_ROOT / _candidate
    SEG_MODEL_PATH = _candidate.resolve()
else:
    SEG_MODEL_PATH = _auto_discover_seg_model_path()
