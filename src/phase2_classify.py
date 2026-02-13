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
from src.utils import ensure_dirs, list_image_files, load_image_bgr, setup_logger


@dataclass
class Phase2Prediction:
    crop_path: Path
    pred_idx: int
    pred_label: str
    pred_conf: float
    is_fully_covered: bool
    decision_source: str
    output_path: Path


def _normalize_label(label: str) -> str:
    return " ".join(label.replace("_", " ").replace("-", " ").lower().split())


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


def discover_pt_models(weights_dir: Path) -> list[Path]:
    if not weights_dir.exists():
        return []
    return sorted(weights_dir.rglob("*.pt"))


def choose_cls1_model(
    cls1_model_path: Path | None,
    weights_dir: Path,
    interactive: bool,
    logger,
) -> Path:
    if cls1_model_path is not None:
        candidate = cls1_model_path if cls1_model_path.is_absolute() else (config.PROJECT_ROOT / cls1_model_path)
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"CLS1_MODEL_PATH not found: {candidate}")
        logger.info("Using explicit Phase 2 model: %s", candidate)
        return candidate

    pt_candidates = discover_pt_models(weights_dir)
    if not pt_candidates:
        raise FileNotFoundError(f"No .pt files found under weights directory: {weights_dir}")

    if len(pt_candidates) == 1:
        logger.info("Only one model found; selected: %s", pt_candidates[0])
        return pt_candidates[0]

    ranked = sorted(pt_candidates, key=lambda p: (_score_cls1_candidate(p), str(p).lower()), reverse=True)
    top_score = _score_cls1_candidate(ranked[0])
    second_score = _score_cls1_candidate(ranked[1]) if len(ranked) > 1 else -999

    if top_score > second_score and top_score > 0:
        logger.info("Auto-selected Phase 2 model: %s (score=%s)", ranked[0], top_score)
        return ranked[0]

    logger.info("Multiple model files found. Select Phase 2 model index:")
    for i, path in enumerate(ranked):
        logger.info("[%s] %s (score=%s)", i, path, _score_cls1_candidate(path))

    if interactive and sys.stdin.isatty():
        default_idx = 0
        while True:
            raw = input(f"Select Phase 2 model index [default {default_idx}]: ").strip()
            if raw == "":
                selected = ranked[default_idx]
                break
            if raw.isdigit() and 0 <= int(raw) < len(ranked):
                selected = ranked[int(raw)]
                break
            print("Invalid selection. Try again.")
        logger.info("User selected Phase 2 model: %s", selected)
        return selected

    logger.warning("Non-interactive mode with multiple models; defaulting to top candidate: %s", ranked[0])
    return ranked[0]


def _names_to_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def _resolve_class_groups(
    names: dict[int, str],
    covered_names: list[str],
    uncovered_names: list[str],
) -> tuple[list[int], list[int]]:
    def _matches_target(normalized_name: str, normalized_target: str) -> bool:
        if normalized_name == normalized_target:
            return True
        pattern = r"\b" + r"\s+".join(re.escape(part) for part in normalized_target.split()) + r"\b"
        return re.search(pattern, normalized_name) is not None

    covered_targets = {_normalize_label(x) for x in covered_names}
    uncovered_targets = {_normalize_label(x) for x in uncovered_names}

    covered_ids: list[int] = []
    uncovered_ids: list[int] = []
    for cls_id, cls_name in names.items():
        n = _normalize_label(cls_name)
        if any(_matches_target(n, t) for t in covered_targets):
            covered_ids.append(cls_id)
        if any(_matches_target(n, t) for t in uncovered_targets):
            uncovered_ids.append(cls_id)
    return sorted(set(covered_ids)), sorted(set(uncovered_ids))


def _infer_coverage_from_label(label: str) -> bool | None:
    normalized = _normalize_label(label)
    if "uncovered" in normalized or "partial" in normalized:
        return False
    if "fully covered" in normalized or normalized == "covered" or "covered" in normalized:
        return True
    return None


def _image_key_from_crop_name(crop_path: Path) -> str:
    return re.sub(r"_crop_\d+$", "", crop_path.stem)


def _save_classification_overlay(image_bgr, text: str, output_path: Path) -> None:
    canvas = image_bgr.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), canvas)


def run_phase2(
    phase1_crops_dir: Path | str | None = None,
    cls1_model_path: Path | str | None = None,
    interactive_model_select: bool = True,
    sample_size: int = 5,
    covered_class_names: list[str] | None = None,
    uncovered_class_names: list[str] | None = None,
) -> bool:
    phase2_output_dir = config.PHASE2_OUTPUT_DIR
    summary_csv = config.PHASE2_SUMMARY_CSV
    image_gate_csv = config.OUTPUT_DIR / "phase2_image_gate.csv"
    logs_dir = config.PROJECT_ROOT / "logs"
    ensure_dirs([phase2_output_dir, logs_dir])

    logger = setup_logger("phase2", logs_dir / "phase2.log")

    if phase1_crops_dir is None:
        crops_dir = config.PHASE1_CROPS_DIR
    else:
        crops_dir = Path(phase1_crops_dir)
        if not crops_dir.is_absolute():
            crops_dir = (config.PROJECT_ROOT / crops_dir).resolve()

    model_path: Path | None
    if cls1_model_path is None:
        model_path = config.CLS1_MODEL_PATH
    else:
        model_path = Path(cls1_model_path)

    model_path = choose_cls1_model(
        cls1_model_path=model_path,
        weights_dir=config.WEIGHTS_DIR,
        interactive=interactive_model_select,
        logger=logger,
    )

    logger.info("Loading Phase 2 classifier on CPU: %s", model_path)
    model = YOLO(str(model_path))
    names = _names_to_dict(model.names)

    logger.info("Phase 2 model class names (index -> name):")
    for idx in sorted(names.keys()):
        logger.info("  %s -> %s", idx, names[idx])

    covered_ids, uncovered_ids = _resolve_class_groups(
        names=names,
        covered_names=covered_class_names if covered_class_names is not None else config.PHASE2_COVERED_CLASS_NAMES,
        uncovered_names=(
            uncovered_class_names if uncovered_class_names is not None else config.PHASE2_UNCOVERED_CLASS_NAMES
        ),
    )
    logger.info("Mapped covered class ids: %s", covered_ids)
    logger.info("Mapped uncovered/partial class ids: %s", uncovered_ids)

    crop_files = list_image_files(crops_dir)
    if not crop_files:
        logger.error("No Phase 1 crops found in: %s", crops_dir)
        logger.error("Run Phase 1 first or provide --phase1-crops-dir.")
        return False

    n = min(sample_size, len(crop_files))
    sampled = crop_files[:n]
    logger.info("Processing %s crop image(s) from %s", n, crops_dir)

    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    predictions: list[Phase2Prediction] = []

    for crop_path in sampled:
        image_bgr = load_image_bgr(crop_path)
        result = model.predict(source=str(crop_path), device="cpu", verbose=False)[0]
        if result.probs is None:
            failures.append(f"{crop_path.name}: classification probabilities are missing.")
            continue

        pred_idx = int(result.probs.top1)
        pred_conf = float(result.probs.top1conf.item())
        pred_label = names.get(pred_idx, f"class_{pred_idx}")

        is_covered: bool | None = None
        decision_source = ""
        if covered_ids and pred_idx in covered_ids:
            is_covered = True
            decision_source = "mapped_class_id"
        elif uncovered_ids and pred_idx in uncovered_ids:
            is_covered = False
            decision_source = "mapped_class_id"
        else:
            is_covered = _infer_coverage_from_label(pred_label)
            decision_source = "label_heuristic"

        if is_covered is None:
            failures.append(f"{crop_path.name}: unable to map label '{pred_label}' to covered/uncovered decision.")
            continue

        text = f"{pred_label} ({pred_conf:.3f}) | fully_covered={is_covered}"
        output_path = phase2_output_dir / f"{crop_path.stem}_cls.jpg"
        _save_classification_overlay(image_bgr, text, output_path)

        if not output_path.exists():
            failures.append(f"{crop_path.name}: output overlay missing at {output_path}")

        if pred_label.strip() == "":
            failures.append(f"{crop_path.name}: predicted label is empty.")

        if pred_conf < 0.0 or pred_conf > 1.0:
            failures.append(f"{crop_path.name}: prediction confidence out of range [0,1]: {pred_conf}")

        predictions.append(
            Phase2Prediction(
                crop_path=crop_path,
                pred_idx=pred_idx,
                pred_label=pred_label,
                pred_conf=pred_conf,
                is_fully_covered=is_covered,
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
                "is_fully_covered": is_covered,
                "decision_source": decision_source,
                "output_path": str(output_path),
            }
        )
        logger.info(
            "Crop=%s pred=%s conf=%.4f fully_covered=%s",
            crop_path.name,
            pred_label,
            pred_conf,
            is_covered,
        )

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "crop_path",
                "pred_idx",
                "pred_label",
                "pred_conf",
                "is_fully_covered",
                "decision_source",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote Phase 2 summary CSV: %s", summary_csv)

    image_groups: dict[str, list[Phase2Prediction]] = {}
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
                "is_fully_covered",
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
                    "is_fully_covered": best.is_fully_covered,
                }
            )
    logger.info("Wrote Phase 2 image gate CSV: %s", image_gate_csv)

    if failures:
        logger.error("PHASE 2 FAILED with %s failure(s):", len(failures))
        for msg in failures:
            logger.error("  - %s", msg)
        return False

    logger.info("PHASE 2 PASSED")
    return True
