from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from src import config
from src.utils import ensure_dirs, load_image_bgr, setup_logger


@dataclass
class Phase5Prediction:
    crop_path: Path
    detections_count: int
    labels: list[str]
    confs: list[float]
    areas_px: list[int]
    dominant_label: str
    dominant_area_px: int
    violation: bool
    output_overlay_path: Path


def _normalize_label(label: str) -> str:
    return " ".join(label.replace("_", " ").replace("-", " ").lower().split())


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


def discover_pt_models(weights_dir: Path) -> list[Path]:
    if not weights_dir.exists():
        return []
    return sorted(weights_dir.rglob("*.pt"))


def choose_seg_model(
    seg_model_path: Path | None,
    weights_dir: Path,
    interactive: bool,
    logger,
) -> Path:
    if seg_model_path is not None:
        candidate = seg_model_path if seg_model_path.is_absolute() else (config.PROJECT_ROOT / seg_model_path)
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"SEG_MODEL_PATH not found: {candidate}")
        logger.info("Using explicit Phase 5 model: %s", candidate)
        return candidate

    pt_candidates = discover_pt_models(weights_dir)
    if not pt_candidates:
        raise FileNotFoundError(f"No .pt files found under weights directory: {weights_dir}")

    if len(pt_candidates) == 1:
        logger.info("Only one model found; selected: %s", pt_candidates[0])
        return pt_candidates[0]

    ranked = sorted(pt_candidates, key=lambda p: (_score_seg_candidate(p), str(p).lower()), reverse=True)
    top_score = _score_seg_candidate(ranked[0])
    second_score = _score_seg_candidate(ranked[1]) if len(ranked) > 1 else -999

    if top_score > second_score and top_score > 0:
        logger.info("Auto-selected Phase 5 model: %s (score=%s)", ranked[0], top_score)
        return ranked[0]

    logger.info("Multiple model files found. Select Phase 5 model index:")
    for i, path in enumerate(ranked):
        logger.info("[%s] %s (score=%s)", i, path, _score_seg_candidate(path))

    if interactive and sys.stdin.isatty():
        default_idx = 0
        while True:
            raw = input(f"Select Phase 5 model index [default {default_idx}]: ").strip()
            if raw == "":
                selected = ranked[default_idx]
                break
            if raw.isdigit() and 0 <= int(raw) < len(ranked):
                selected = ranked[int(raw)]
                break
            print("Invalid selection. Try again.")
        logger.info("User selected Phase 5 model: %s", selected)
        return selected

    logger.warning("Non-interactive mode with multiple models; defaulting to top candidate: %s", ranked[0])
    return ranked[0]


def _names_to_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _image_key_from_crop_name(crop_path: Path) -> str:
    return re.sub(r"_crop_\d+$", "", crop_path.stem)


def _load_phase2_uncovered_crops(phase2_summary_csv: Path, logger) -> tuple[list[Path], int, int]:
    if not phase2_summary_csv.exists():
        return [], 0, 0

    uncovered_crops: list[Path] = []
    uncovered_rows = 0
    missing_rows = 0
    with phase2_summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _parse_bool(row.get("is_fully_covered", "")):
                continue
            uncovered_rows += 1
            raw = str(row.get("crop_path", "")).strip()
            if raw == "":
                missing_rows += 1
                continue
            crop = Path(raw)
            if not crop.is_absolute():
                crop = (config.PROJECT_ROOT / crop).resolve()
            if crop.exists():
                uncovered_crops.append(crop)
            else:
                missing_rows += 1
                logger.warning("Phase 2 uncovered crop listed but missing on disk: %s", crop)

    return sorted(set(uncovered_crops)), uncovered_rows, missing_rows


def _save_mask(mask_2d: np.ndarray, out_path: Path) -> None:
    mask_u8 = (mask_2d > 0.5).astype(np.uint8) * 255
    cv2.imwrite(str(out_path), mask_u8)


def run_phase5(
    phase2_summary_csv: Path | str | None = None,
    seg_model_path: Path | str | None = None,
    interactive_model_select: bool = True,
    sample_size: int = 5,
    seg_conf_threshold: float | None = None,
) -> bool:
    phase5_output_dir = config.PHASE5_OUTPUT_DIR
    phase5_masks_dir = config.PHASE5_MASKS_DIR
    summary_csv = config.PHASE5_SUMMARY_CSV
    image_gate_csv = config.PHASE5_IMAGE_GATE_CSV
    logs_dir = config.PROJECT_ROOT / "logs"
    ensure_dirs([phase5_output_dir, phase5_masks_dir, logs_dir])

    logger = setup_logger("phase5", logs_dir / "phase5.log")

    if phase2_summary_csv is None:
        phase2_csv = config.PHASE2_SUMMARY_CSV
    else:
        phase2_csv = Path(phase2_summary_csv)
        if not phase2_csv.is_absolute():
            phase2_csv = (config.PROJECT_ROOT / phase2_csv).resolve()

    model_path: Path | None
    if seg_model_path is None:
        model_path = config.SEG_MODEL_PATH
    else:
        model_path = Path(seg_model_path)

    model_path = choose_seg_model(
        seg_model_path=model_path,
        weights_dir=config.WEIGHTS_DIR,
        interactive=interactive_model_select,
        logger=logger,
    )

    conf_thr = config.PHASE5_SEG_CONF_THRESHOLD if seg_conf_threshold is None else float(seg_conf_threshold)
    logger.info("Loading Phase 5 segmentation model on CPU: %s", model_path)
    logger.info("Segmentation confidence threshold: %.3f", conf_thr)
    model = YOLO(str(model_path))
    names = _names_to_dict(model.names)

    logger.info("Phase 5 model class names (index -> name):")
    for idx in sorted(names.keys()):
        logger.info("  %s -> %s", idx, names[idx])

    uncovered_crops, uncovered_rows, missing_rows = _load_phase2_uncovered_crops(phase2_csv, logger)

    if uncovered_rows == 0:
        logger.info("No uncovered/partial crops found in Phase 2 summary. Phase 5 skipped.")
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "crop_path",
                    "detections_count",
                    "labels",
                    "confs",
                    "areas_px",
                    "dominant_label",
                    "dominant_area_px",
                    "violation",
                    "output_overlay_path",
                ],
            )
            writer.writeheader()
        with image_gate_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_key",
                    "crops_count",
                    "best_crop_path",
                    "load_types",
                    "dominant_label",
                    "dominant_area_px",
                    "violation",
                ],
            )
            writer.writeheader()
        logger.info("PHASE 5 PASSED (skipped: no uncovered branch inputs)")
        return True

    if not uncovered_crops:
        logger.error(
            "Uncovered rows were found in Phase 2 summary, but no valid crop files were available. missing_rows=%s",
            missing_rows,
        )
        return False

    n = min(sample_size, len(uncovered_crops))
    sampled = uncovered_crops[:n]
    logger.info(
        "Processing %s uncovered crop(s) from Phase 2 summary (total uncovered rows=%s, missing rows=%s)",
        n,
        uncovered_rows,
        missing_rows,
    )

    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    predictions: list[Phase5Prediction] = []

    for crop_path in sampled:
        image_bgr = load_image_bgr(crop_path)
        result = model.predict(source=str(crop_path), device="cpu", conf=conf_thr, verbose=False)[0]

        overlay = result.plot()
        overlay_path = phase5_output_dir / f"{crop_path.stem}_seg.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        if not overlay_path.exists():
            failures.append(f"{crop_path.name}: overlay output missing at {overlay_path}")

        labels: list[str] = []
        confs: list[float] = []
        areas_px: list[int] = []

        boxes = result.boxes
        masks = result.masks
        detections_count = 0 if boxes is None else len(boxes)

        if detections_count == 0:
            labels = ["no_load_detected"]
            dominant_label = "no_load_detected"
            dominant_area_px = 0
        else:
            mask_data = masks.data.cpu().numpy() if masks is not None and masks.data is not None else None
            for i in range(detections_count):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                label = names.get(cls_id, f"class_{cls_id}")

                if label.strip() == "":
                    failures.append(f"{crop_path.name}: empty label for detection {i + 1}")
                    label = "unknown"
                if conf < 0.0 or conf > 1.0:
                    failures.append(f"{crop_path.name}: confidence out of range for detection {i + 1}: {conf}")

                area_px = 0
                if mask_data is not None and i < mask_data.shape[0]:
                    mask_2d = mask_data[i]
                    area_px = int((mask_2d > 0.5).sum())
                    safe_label = re.sub(r"[^a-zA-Z0-9_]+", "_", _normalize_label(label))
                    mask_path = phase5_masks_dir / f"{crop_path.stem}_mask_{i + 1}_{safe_label}.png"
                    _save_mask(mask_2d, mask_path)
                else:
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    area_px = max(0, int(round((x2 - x1) * (y2 - y1))))

                labels.append(_normalize_label(label).replace(" ", "_"))
                confs.append(conf)
                areas_px.append(area_px)

            if not labels:
                failures.append(f"{crop_path.name}: segmentation returned detections but no valid labels.")
                labels = ["no_load_detected"]
                dominant_label = "no_load_detected"
                dominant_area_px = 0
            else:
                best_idx = max(range(len(labels)), key=lambda idx: areas_px[idx])
                dominant_label = labels[best_idx]
                dominant_area_px = areas_px[best_idx]

        violation = True
        predictions.append(
            Phase5Prediction(
                crop_path=crop_path,
                detections_count=detections_count,
                labels=labels,
                confs=confs,
                areas_px=areas_px,
                dominant_label=dominant_label,
                dominant_area_px=dominant_area_px,
                violation=violation,
                output_overlay_path=overlay_path,
            )
        )

        rows.append(
            {
                "crop_path": str(crop_path),
                "detections_count": detections_count,
                "labels": "|".join(labels),
                "confs": "|".join(f"{c:.6f}" for c in confs),
                "areas_px": "|".join(str(a) for a in areas_px),
                "dominant_label": dominant_label,
                "dominant_area_px": dominant_area_px,
                "violation": violation,
                "output_overlay_path": str(overlay_path),
            }
        )
        logger.info(
            "Crop=%s detections=%s labels=%s dominant=%s violation=%s",
            crop_path.name,
            detections_count,
            labels,
            dominant_label,
            violation,
        )

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "crop_path",
                "detections_count",
                "labels",
                "confs",
                "areas_px",
                "dominant_label",
                "dominant_area_px",
                "violation",
                "output_overlay_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote Phase 5 summary CSV: %s", summary_csv)

    image_groups: dict[str, list[Phase5Prediction]] = {}
    for pred in predictions:
        key = _image_key_from_crop_name(pred.crop_path)
        image_groups.setdefault(key, []).append(pred)

    with image_gate_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_key",
                "crops_count",
                "best_crop_path",
                "load_types",
                "dominant_label",
                "dominant_area_px",
                "violation",
            ],
        )
        writer.writeheader()
        for key, preds in sorted(image_groups.items()):
            preds_sorted = sorted(preds, key=lambda p: p.dominant_area_px, reverse=True)
            best = preds_sorted[0]
            load_types = sorted({label for p in preds for label in p.labels if label != "no_load_detected"})
            writer.writerow(
                {
                    "image_key": key,
                    "crops_count": len(preds),
                    "best_crop_path": str(best.crop_path),
                    "load_types": "|".join(load_types) if load_types else "no_load_detected",
                    "dominant_label": best.dominant_label,
                    "dominant_area_px": best.dominant_area_px,
                    "violation": best.violation,
                }
            )
    logger.info("Wrote Phase 5 image gate CSV: %s", image_gate_csv)

    if failures:
        logger.error("PHASE 5 FAILED with %s failure(s):", len(failures))
        for msg in failures:
            logger.error("  - %s", msg)
        return False

    logger.info("PHASE 5 PASSED")
    return True

