from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from src import config
from src.utils import clamp_xyxy, ensure_dirs, list_image_files, load_image_bgr, setup_logger


@dataclass
class BedDetection:
    xyxy: tuple[int, int, int, int]
    conf: float
    cls_id: int
    cls_name: str
    source: str


def _normalize_label(label: str) -> str:
    return " ".join(label.replace("_", " ").replace("-", " ").lower().split())


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


def discover_pt_models(weights_dir: Path) -> list[Path]:
    if not weights_dir.exists():
        return []
    return sorted(weights_dir.rglob("*.pt"))


def choose_detection_model(
    detect_model_path: Path | None,
    weights_dir: Path,
    interactive: bool,
    logger,
) -> Path:
    if detect_model_path is not None:
        candidate = detect_model_path if detect_model_path.is_absolute() else (config.PROJECT_ROOT / detect_model_path)
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"DETECT_MODEL_PATH not found: {candidate}")
        logger.info("Using explicit detection model: %s", candidate)
        return candidate

    pt_candidates = discover_pt_models(weights_dir)
    if not pt_candidates:
        raise FileNotFoundError(f"No .pt files found under weights directory: {weights_dir}")

    if len(pt_candidates) == 1:
        logger.info("Only one model found; selected: %s", pt_candidates[0])
        return pt_candidates[0]

    ranked = sorted(pt_candidates, key=lambda p: (_score_detect_candidate(p), str(p).lower()), reverse=True)
    top_score = _score_detect_candidate(ranked[0])
    second_score = _score_detect_candidate(ranked[1]) if len(ranked) > 1 else -999

    if top_score > second_score and top_score > 0:
        logger.info("Auto-selected detection model: %s (score=%s)", ranked[0], top_score)
        return ranked[0]

    logger.info("Multiple model files found. Select detection model index:")
    for i, path in enumerate(ranked):
        logger.info("[%s] %s (score=%s)", i, path, _score_detect_candidate(path))

    if interactive and sys.stdin.isatty():
        default_idx = 0
        while True:
            raw = input(f"Select detection model index [default {default_idx}]: ").strip()
            if raw == "":
                selected = ranked[default_idx]
                break
            if raw.isdigit() and 0 <= int(raw) < len(ranked):
                selected = ranked[int(raw)]
                break
            print("Invalid selection. Try again.")
        logger.info("User selected detection model: %s", selected)
        return selected

    logger.warning("Non-interactive mode with multiple models; defaulting to top candidate: %s", ranked[0])
    return ranked[0]


def _names_to_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def resolve_class_ids(
    names: dict[int, str],
    bed_class_ids: list[int] | None,
    bed_class_names: list[str] | None,
    truck_class_names: list[str] | None,
) -> tuple[list[int], list[int]]:
    max_cls = set(names.keys())

    resolved_bed_ids: list[int] = []
    if bed_class_ids:
        resolved_bed_ids = sorted([int(i) for i in bed_class_ids if int(i) in max_cls])

    if not resolved_bed_ids and bed_class_names:
        targets = {_normalize_label(x) for x in bed_class_names}
        for cls_id, cls_name in names.items():
            n = _normalize_label(cls_name)
            if any(t == n or t in n for t in targets):
                resolved_bed_ids.append(cls_id)
        resolved_bed_ids = sorted(set(resolved_bed_ids))

    resolved_truck_ids: list[int] = []
    if truck_class_names:
        truck_targets = {_normalize_label(x) for x in truck_class_names}
        for cls_id, cls_name in names.items():
            n = _normalize_label(cls_name)
            if any(t == n or t in n for t in truck_targets):
                resolved_truck_ids.append(cls_id)
        resolved_truck_ids = sorted(set(resolved_truck_ids))

    return resolved_bed_ids, resolved_truck_ids


def heuristic_bed_from_truck_bbox(xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    width = x2 - x1
    height = y2 - y1
    hx1 = x1 + config.HEURISTIC_X_START_RATIO * width
    hx2 = x1 + config.HEURISTIC_X_END_RATIO * width
    hy1 = y1 + config.HEURISTIC_Y_START_RATIO * height
    hy2 = y1 + config.HEURISTIC_Y_END_RATIO * height
    return hx1, hy1, hx2, hy2


def extract_bed_detections(
    result,
    names: dict[int, str],
    bed_ids: list[int],
    truck_ids: list[int],
    logger,
) -> list[BedDetection]:
    orig = result.orig_img
    h, w = orig.shape[:2]
    boxes = result.boxes
    detections: list[BedDetection] = []

    if boxes is None or len(boxes) == 0:
        return detections

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        cls_name = names.get(cls_id, f"class_{cls_id}")
        if bed_ids and cls_id in bed_ids:
            cx1, cy1, cx2, cy2 = clamp_xyxy(x1, y1, x2, y2, w, h)
            detections.append(
                BedDetection(
                    xyxy=(cx1, cy1, cx2, cy2),
                    conf=conf,
                    cls_id=cls_id,
                    cls_name=cls_name,
                    source="direct",
                )
            )

    if detections:
        return detections

    logger.warning("No direct bed-class detections found. Falling back to truck->bed heuristic.")

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        cls_name = names.get(cls_id, f"class_{cls_id}")
        if truck_ids and cls_id not in truck_ids:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        hx1, hy1, hx2, hy2 = heuristic_bed_from_truck_bbox((x1, y1, x2, y2))
        cx1, cy1, cx2, cy2 = clamp_xyxy(hx1, hy1, hx2, hy2, w, h)
        detections.append(
            BedDetection(
                xyxy=(cx1, cy1, cx2, cy2),
                conf=max(0.0, conf * 0.80),
                cls_id=cls_id,
                cls_name=f"{cls_name}_heuristic",
                source="heuristic",
            )
        )

    return detections


def draw_detections(image_bgr, detections: list[BedDetection]):
    canvas = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        color = (0, 200, 0) if det.source == "direct" else (0, 165, 255)
        label = f"{det.cls_name} {det.conf:.2f} [{det.source}]"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def run_phase1(
    input_images_dir: Path | str | None = None,
    detect_model_path: Path | str | None = None,
    top1_only: bool = config.TOP1_ONLY,
    bed_class_ids: list[int] | None = None,
    bed_class_names: list[str] | None = None,
    detect_conf_threshold: float | None = None,
    interactive_model_select: bool = True,
    sample_size: int = 5,
) -> bool:
    output_det_dir = config.OUTPUT_DIR / "phase1_detection"
    output_crops_dir = config.OUTPUT_DIR / "phase1_crops"
    logs_dir = config.PROJECT_ROOT / "logs"
    ensure_dirs([config.OUTPUT_DIR, output_det_dir, output_crops_dir, logs_dir])

    logger = setup_logger("phase1", logs_dir / "phase1.log")

    if input_images_dir is None:
        input_dir = config.INPUT_IMAGES_DIR
    else:
        input_dir = Path(input_images_dir)
        if not input_dir.is_absolute():
            input_dir = (config.PROJECT_ROOT / input_dir).resolve()

    model_path: Path | None
    if detect_model_path is None:
        model_path = config.DETECT_MODEL_PATH
    else:
        model_path = Path(detect_model_path)

    model_path = choose_detection_model(
        detect_model_path=model_path,
        weights_dir=config.WEIGHTS_DIR,
        interactive=interactive_model_select,
        logger=logger,
    )

    logger.info("Loading YOLO model on CPU: %s", model_path)
    model = YOLO(str(model_path))

    names = _names_to_dict(model.names)
    logger.info("Model class names (index -> name):")
    for idx in sorted(names.keys()):
        logger.info("  %s -> %s", idx, names[idx])

    resolved_bed_ids, resolved_truck_ids = resolve_class_ids(
        names=names,
        bed_class_ids=bed_class_ids if bed_class_ids is not None else config.BED_CLASS_IDS,
        bed_class_names=bed_class_names if bed_class_names is not None else config.BED_CLASS_NAMES,
        truck_class_names=config.TRUCK_FALLBACK_CLASS_NAMES,
    )

    conf_thr = config.DETECT_CONF_THRESHOLD if detect_conf_threshold is None else float(detect_conf_threshold)
    logger.info("Configured bed class ids: %s", resolved_bed_ids if resolved_bed_ids else "None (heuristic fallback)")
    logger.info("Configured truck fallback class ids: %s", resolved_truck_ids if resolved_truck_ids else "None (all boxes)")
    logger.info("Detection confidence threshold: %.4f", conf_thr)

    image_files = list_image_files(input_dir)
    if not image_files:
        logger.error("No input images found in: %s", input_dir)
        logger.error("Phase 1 cannot run acceptance tests without input images.")
        return False

    n = min(sample_size, len(image_files))
    sampled = image_files[:n]
    logger.info("Processing %s image(s) from %s", n, input_dir)

    summary_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for image_path in sampled:
        image_bgr = load_image_bgr(image_path)
        result = model.predict(
            source=str(image_path),
            device="cpu",
            conf=conf_thr,
            verbose=False,
        )[0]

        bed_dets = extract_bed_detections(
            result=result,
            names=names,
            bed_ids=resolved_bed_ids,
            truck_ids=resolved_truck_ids,
            logger=logger,
        )

        bed_dets = sorted(bed_dets, key=lambda d: d.conf, reverse=True)
        detections_count = len(bed_dets)

        if top1_only and bed_dets:
            bed_dets = bed_dets[:1]

        det_annotated = draw_detections(image_bgr, bed_dets)
        det_out_path = output_det_dir / f"{image_path.stem}_det.jpg"
        cv2.imwrite(str(det_out_path), det_annotated)

        crop_paths: list[Path] = []
        crop_sizes: list[tuple[int, int]] = []
        best_conf = None
        for k, det in enumerate(bed_dets, start=1):
            x1, y1, x2, y2 = det.xyxy
            crop = image_bgr[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2] if crop is not None else (0, 0)
            crop_area = crop_h * crop_w
            crop_path = output_crops_dir / f"{image_path.stem}_crop_{k}.jpg"

            if crop_h <= 0 or crop_w <= 0:
                failures.append(f"{image_path.name}: empty crop generated for detection {k}")
                continue

            cv2.imwrite(str(crop_path), crop)
            crop_paths.append(crop_path)
            crop_sizes.append((crop_w, crop_h))

            if best_conf is None:
                best_conf = det.conf

            if not crop_path.exists():
                failures.append(f"{image_path.name}: crop file missing after write: {crop_path}")

            if crop_area <= config.MIN_CROP_AREA:
                failures.append(
                    f"{image_path.name}: crop area too small ({crop_area}) <= threshold ({config.MIN_CROP_AREA})"
                )

        if not crop_paths:
            failures.append(f"{image_path.name}: no crop saved (detections_count={detections_count})")

        logger.info(
            "Image=%s detections=%s crops=%s crop_sizes=%s",
            image_path.name,
            detections_count,
            len(crop_paths),
            crop_sizes,
        )

        best_crop_path = str(crop_paths[0]) if crop_paths else ""
        best_w = crop_sizes[0][0] if crop_sizes else ""
        best_h = crop_sizes[0][1] if crop_sizes else ""

        summary_rows.append(
            {
                "image_path": str(image_path),
                "detections_count": detections_count,
                "crops_count": len(crop_paths),
                "best_crop_path": best_crop_path,
                "best_crop_w": best_w,
                "best_crop_h": best_h,
                "best_conf": f"{best_conf:.6f}" if best_conf is not None else "",
            }
        )

    summary_csv = config.OUTPUT_DIR / "phase1_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "detections_count",
                "crops_count",
                "best_crop_path",
                "best_crop_w",
                "best_crop_h",
                "best_conf",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    logger.info("Wrote summary CSV: %s", summary_csv)

    if failures:
        logger.error("PHASE 1 FAILED with %s failure(s):", len(failures))
        for msg in failures:
            logger.error("  - %s", msg)
        return False

    logger.info("PHASE 1 PASSED")
    return True
