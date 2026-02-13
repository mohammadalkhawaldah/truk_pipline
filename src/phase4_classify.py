from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from src import config
from src.utils import ensure_dirs, load_image_bgr, setup_logger


@dataclass
class Phase4Prediction:
    crop_path: Path
    pred_idx: int
    pred_label: str
    pred_conf: float
    irregular_type: str
    violation: bool
    decision_source: str
    output_path: Path


def _normalize_label(label: str) -> str:
    return " ".join(label.replace("_", " ").replace("-", " ").lower().split())


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


def discover_pt_models(weights_dir: Path) -> list[Path]:
    if not weights_dir.exists():
        return []
    return sorted(weights_dir.rglob("*.pt"))


def choose_cls3_model(
    cls3_model_path: Path | None,
    weights_dir: Path,
    interactive: bool,
    logger,
) -> Path:
    if cls3_model_path is not None:
        candidate = cls3_model_path if cls3_model_path.is_absolute() else (config.PROJECT_ROOT / cls3_model_path)
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"CLS3_MODEL_PATH not found: {candidate}")
        logger.info("Using explicit Phase 4 model: %s", candidate)
        return candidate

    pt_candidates = discover_pt_models(weights_dir)
    if not pt_candidates:
        raise FileNotFoundError(f"No .pt files found under weights directory: {weights_dir}")

    if len(pt_candidates) == 1:
        logger.info("Only one model found; selected: %s", pt_candidates[0])
        return pt_candidates[0]

    ranked = sorted(pt_candidates, key=lambda p: (_score_cls3_candidate(p), str(p).lower()), reverse=True)
    top_score = _score_cls3_candidate(ranked[0])
    second_score = _score_cls3_candidate(ranked[1]) if len(ranked) > 1 else -999

    if top_score > second_score and top_score > 0:
        logger.info("Auto-selected Phase 4 model: %s (score=%s)", ranked[0], top_score)
        return ranked[0]

    logger.info("Multiple model files found. Select Phase 4 model index:")
    for i, path in enumerate(ranked):
        logger.info("[%s] %s (score=%s)", i, path, _score_cls3_candidate(path))

    if interactive and sys.stdin.isatty():
        default_idx = 0
        while True:
            raw = input(f"Select Phase 4 model index [default {default_idx}]: ").strip()
            if raw == "":
                selected = ranked[default_idx]
                break
            if raw.isdigit() and 0 <= int(raw) < len(ranked):
                selected = ranked[int(raw)]
                break
            print("Invalid selection. Try again.")
        logger.info("User selected Phase 4 model: %s", selected)
        return selected

    logger.warning("Non-interactive mode with multiple models; defaulting to top candidate: %s", ranked[0])
    return ranked[0]


def _names_to_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def _matches_target(normalized_name: str, normalized_target: str) -> bool:
    if normalized_name == normalized_target:
        return True
    pattern = r"\b" + r"\s+".join(re.escape(part) for part in normalized_target.split()) + r"\b"
    return re.search(pattern, normalized_name) is not None


def _resolve_subtype_class_ids(
    names: dict[int, str],
    sandlike_names: list[str],
    ironbars_names: list[str],
    unknown_names: list[str],
) -> tuple[list[int], list[int], list[int]]:
    sand_targets = {_normalize_label(x) for x in sandlike_names}
    iron_targets = {_normalize_label(x) for x in ironbars_names}
    unknown_targets = {_normalize_label(x) for x in unknown_names}

    sand_ids: list[int] = []
    iron_ids: list[int] = []
    unknown_ids: list[int] = []
    for cls_id, cls_name in names.items():
        n = _normalize_label(cls_name)
        if any(_matches_target(n, t) for t in sand_targets):
            sand_ids.append(cls_id)
        if any(_matches_target(n, t) for t in iron_targets):
            iron_ids.append(cls_id)
        if any(_matches_target(n, t) for t in unknown_targets):
            unknown_ids.append(cls_id)

    return sorted(set(sand_ids)), sorted(set(iron_ids)), sorted(set(unknown_ids))


def _infer_irregular_type_from_label(label: str) -> str | None:
    normalized = _normalize_label(label)
    if "sand" in normalized:
        return "sand_like"
    if "iron" in normalized or "bar" in normalized:
        return "ironbars_like"
    if "irregular" in normalized or "unknown" in normalized or "other" in normalized:
        return "unknown"
    return None


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _image_key_from_crop_name(crop_path: Path) -> str:
    return re.sub(r"_crop_\d+$", "", crop_path.stem)


def _save_overlay(image_bgr, text: str, output_path: Path) -> None:
    canvas = image_bgr.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), canvas)


def _load_phase3_irregular_crops(phase3_summary_csv: Path, logger) -> list[Path]:
    if not phase3_summary_csv.exists():
        return []

    irregular_crops: list[Path] = []
    with phase3_summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _parse_bool(row.get("is_irregular", "")):
                continue
            raw = str(row.get("crop_path", "")).strip()
            if raw == "":
                continue
            crop = Path(raw)
            if not crop.is_absolute():
                crop = (config.PROJECT_ROOT / crop).resolve()
            if crop.exists():
                irregular_crops.append(crop)
            else:
                logger.warning("Phase 3 crop listed but missing on disk: %s", crop)

    return sorted(set(irregular_crops))


def run_phase4(
    phase3_summary_csv: Path | str | None = None,
    cls3_model_path: Path | str | None = None,
    interactive_model_select: bool = True,
    sample_size: int = 5,
    sandlike_class_names: list[str] | None = None,
    ironbars_class_names: list[str] | None = None,
    unknown_class_names: list[str] | None = None,
) -> bool:
    phase4_output_dir = config.PHASE4_OUTPUT_DIR
    summary_csv = config.PHASE4_SUMMARY_CSV
    image_gate_csv = config.PHASE4_IMAGE_GATE_CSV
    logs_dir = config.PROJECT_ROOT / "logs"
    ensure_dirs([phase4_output_dir, logs_dir])

    logger = setup_logger("phase4", logs_dir / "phase4.log")

    if phase3_summary_csv is None:
        phase3_csv = config.PHASE3_SUMMARY_CSV
    else:
        phase3_csv = Path(phase3_summary_csv)
        if not phase3_csv.is_absolute():
            phase3_csv = (config.PROJECT_ROOT / phase3_csv).resolve()

    model_path: Path | None
    if cls3_model_path is None:
        model_path = config.CLS3_MODEL_PATH
    else:
        model_path = Path(cls3_model_path)

    model_path = choose_cls3_model(
        cls3_model_path=model_path,
        weights_dir=config.WEIGHTS_DIR,
        interactive=interactive_model_select,
        logger=logger,
    )

    logger.info("Loading Phase 4 classifier on CPU: %s", model_path)
    model = YOLO(str(model_path))
    names = _names_to_dict(model.names)

    logger.info("Phase 4 model class names (index -> name):")
    for idx in sorted(names.keys()):
        logger.info("  %s -> %s", idx, names[idx])

    sand_ids, iron_ids, unknown_ids = _resolve_subtype_class_ids(
        names=names,
        sandlike_names=sandlike_class_names if sandlike_class_names is not None else config.PHASE4_SANDLIKE_CLASS_NAMES,
        ironbars_names=(
            ironbars_class_names if ironbars_class_names is not None else config.PHASE4_IRONBARS_CLASS_NAMES
        ),
        unknown_names=unknown_class_names if unknown_class_names is not None else config.PHASE4_UNKNOWN_CLASS_NAMES,
    )
    logger.info("Mapped sand-like class ids: %s", sand_ids)
    logger.info("Mapped ironbars-like class ids: %s", iron_ids)
    logger.info("Mapped unknown class ids: %s", unknown_ids)

    irregular_crops = _load_phase3_irregular_crops(phase3_csv, logger)
    if not irregular_crops:
        logger.error("No irregular crops found via Phase 3 summary CSV: %s", phase3_csv)
        logger.error("Run Phase 3 first or verify irregular predictions.")
        return False

    n = min(sample_size, len(irregular_crops))
    sampled = irregular_crops[:n]
    logger.info("Processing %s irregular crop(s) from Phase 3 summary", n)

    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    predictions: list[Phase4Prediction] = []

    for crop_path in sampled:
        image_bgr = load_image_bgr(crop_path)
        result = model.predict(source=str(crop_path), device="cpu", verbose=False)[0]
        if result.probs is None:
            failures.append(f"{crop_path.name}: classification probabilities are missing.")
            continue

        pred_idx = int(result.probs.top1)
        pred_conf = float(result.probs.top1conf.item())
        pred_label = names.get(pred_idx, f"class_{pred_idx}")

        irregular_type: str | None = None
        decision_source = ""
        if sand_ids and pred_idx in sand_ids:
            irregular_type = "sand_like"
            decision_source = "mapped_class_id"
        elif iron_ids and pred_idx in iron_ids:
            irregular_type = "ironbars_like"
            decision_source = "mapped_class_id"
        elif unknown_ids and pred_idx in unknown_ids:
            irregular_type = "unknown"
            decision_source = "mapped_class_id"
        else:
            irregular_type = _infer_irregular_type_from_label(pred_label)
            decision_source = "label_heuristic"

        if irregular_type is None:
            failures.append(f"{crop_path.name}: unable to map label '{pred_label}' to irregular subtype.")
            continue

        violation = True
        text = f"{pred_label} ({pred_conf:.3f}) | type={irregular_type} | violation={violation}"
        output_path = phase4_output_dir / f"{crop_path.stem}_cls.jpg"
        _save_overlay(image_bgr, text, output_path)

        if not output_path.exists():
            failures.append(f"{crop_path.name}: output overlay missing at {output_path}")
        if pred_label.strip() == "":
            failures.append(f"{crop_path.name}: predicted label is empty.")
        if pred_conf < 0.0 or pred_conf > 1.0:
            failures.append(f"{crop_path.name}: prediction confidence out of range [0,1]: {pred_conf}")

        predictions.append(
            Phase4Prediction(
                crop_path=crop_path,
                pred_idx=pred_idx,
                pred_label=pred_label,
                pred_conf=pred_conf,
                irregular_type=irregular_type,
                violation=violation,
                decision_source=decision_source,
                output_path=output_path,
            )
        )

        rows.append(
            {
                "crop_path": str(crop_path),
                "pred_idx": pred_idx,
                "pred_label": pred_label,
                "pred_conf": f"{pred_conf:.6f}",
                "irregular_type": irregular_type,
                "violation": violation,
                "decision_source": decision_source,
                "output_path": str(output_path),
            }
        )
        logger.info(
            "Crop=%s pred=%s conf=%.4f irregular_type=%s violation=%s",
            crop_path.name,
            pred_label,
            pred_conf,
            irregular_type,
            violation,
        )

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "crop_path",
                "pred_idx",
                "pred_label",
                "pred_conf",
                "irregular_type",
                "violation",
                "decision_source",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote Phase 4 summary CSV: %s", summary_csv)

    image_groups: dict[str, list[Phase4Prediction]] = {}
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
                "best_pred_label",
                "best_pred_conf",
                "irregular_type",
                "violation",
            ],
        )
        writer.writeheader()
        for key, preds in sorted(image_groups.items()):
            preds_sorted = sorted(preds, key=lambda p: p.pred_conf, reverse=True)
            best = preds_sorted[0]
            writer.writerow(
                {
                    "image_key": key,
                    "crops_count": len(preds),
                    "best_crop_path": str(best.crop_path),
                    "best_pred_label": best.pred_label,
                    "best_pred_conf": f"{best.pred_conf:.6f}",
                    "irregular_type": best.irregular_type,
                    "violation": best.violation,
                }
            )
    logger.info("Wrote Phase 4 image gate CSV: %s", image_gate_csv)

    if failures:
        logger.error("PHASE 4 FAILED with %s failure(s):", len(failures))
        for msg in failures:
            logger.error("  - %s", msg)
        return False

    logger.info("PHASE 4 PASSED")
    return True

