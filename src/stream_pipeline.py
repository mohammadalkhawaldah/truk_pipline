from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, deque
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from src import config
from src.size_event_pipeline import OnlineSizeEventPipeline, SizeEvent
from src.tracker import CropCandidate, Detection, IoUTracker, TrackState, expand_box_xyxy, iou_xyxy
from src.utils import clamp_xyxy, ensure_dirs, setup_logger


@dataclass
class ModelBundle:
    detect: YOLO
    cls1: YOLO
    cls2: YOLO
    cls3: YOLO
    seg: YOLO

    detect_names: dict[int, str]
    cls1_names: dict[int, str]
    cls2_names: dict[int, str]
    cls3_names: dict[int, str]
    seg_names: dict[int, str]

    bed_ids: list[int]
    truck_ids: list[int]
    covered_ids: list[int]
    uncovered_ids: list[int]
    irregular_ids: list[int]
    regular_ids: list[int]
    sand_ids: list[int]
    iron_ids: list[int]
    unknown_ids: list[int]


def _normalize_label(label: str) -> str:
    return " ".join(label.replace("_", " ").replace("-", " ").lower().split())


def _format_irregular_label(label: str) -> str:
    text = str(label).strip()
    if text == "":
        return ""
    return text.replace(" ", "_").lower()


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


def _ensure_model_path(path_like: Path | str | None, default_path: Path | None, name: str) -> Path:
    path = None
    if path_like is not None:
        path = Path(path_like)
    elif default_path is not None:
        path = Path(default_path)

    if path is None:
        raise FileNotFoundError(f"{name} path is not configured.")
    if not path.is_absolute():
        path = (config.PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _resolve_ids_by_names(names: dict[int, str], target_names: list[str]) -> list[int]:
    targets = {_normalize_label(x) for x in target_names}
    resolved: list[int] = []
    for cls_id, cls_name in names.items():
        n = _normalize_label(cls_name)
        if any(_matches_target(n, t) for t in targets):
            resolved.append(cls_id)
    return sorted(set(resolved))


def _resolve_bed_and_truck_ids(
    names: dict[int, str],
    bed_class_ids: list[int] | None,
    truck_class_ids: list[int] | None,
) -> tuple[list[int], list[int]]:
    available_ids = set(names.keys())
    resolved_bed_ids: list[int] = []
    resolved_truck_ids: list[int] = []
    if truck_class_ids:
        resolved_truck_ids = sorted([int(i) for i in truck_class_ids if int(i) in available_ids])
    if not resolved_truck_ids:
        resolved_truck_ids = _resolve_ids_by_names(names, config.TRUCK_FALLBACK_CLASS_NAMES)

    if bed_class_ids:
        resolved_bed_ids = sorted([int(i) for i in bed_class_ids if int(i) in available_ids])
    if not resolved_bed_ids:
        resolved_bed_ids = _resolve_ids_by_names(names, config.BED_CLASS_NAMES)
    return resolved_bed_ids, resolved_truck_ids


def _bbox_center_xy(box_xyxy: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _clamp_xyxy_floor(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1_i = max(0, min(int(x1), max(0, width - 1)))
    y1_i = max(0, min(int(y1), max(0, height - 1)))
    x2_i = max(0, min(int(x2), width))
    y2_i = max(0, min(int(y2), height))
    if x2_i <= x1_i:
        x2_i = min(width, x1_i + 1)
    if y2_i <= y1_i:
        y2_i = min(height, y1_i + 1)
    return (x1_i, y1_i, x2_i, y2_i)


def _point_in_box(px: float, py: float, box_xyxy: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box_xyxy
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def _dedupe_anchor_detections(detections: list[Detection], iou_threshold: float = 0.7) -> list[Detection]:
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d.conf, reverse=True)
    kept: list[Detection] = []
    for det in sorted_dets:
        if any(iou_xyxy(det.xyxy, k.xyxy) >= iou_threshold for k in kept):
            continue
        kept.append(det)
    return kept


def _extract_anchor_detections(
    result,
    names: dict[int, str],
    truck_ids: list[int],
    bed_ids: list[int],
) -> list[Detection]:
    orig = result.orig_img
    h, w = orig.shape[:2]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    truck_dets: list[Detection] = []
    bed_dets: list[Detection] = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        box_xyxy = clamp_xyxy(x1, y1, x2, y2, w, h)
        cls_name = names.get(cls_id, f"class_{cls_id}")

        if truck_ids and cls_id in truck_ids:
            truck_box_xyxy = _clamp_xyxy_floor(x1, y1, x2, y2, w, h)
            truck_dets.append(
                Detection(
                    xyxy=truck_box_xyxy,
                    conf=conf,
                    cls_id=cls_id,
                    cls_name=cls_name,
                    source="truck_direct",
                )
            )
        if bed_ids and cls_id in bed_ids:
            bed_box_xyxy = _clamp_xyxy_floor(x1, y1, x2, y2, w, h)
            bed_dets.append(
                Detection(
                    xyxy=bed_box_xyxy,
                    conf=conf,
                    cls_id=cls_id,
                    cls_name=cls_name,
                    source="bed_direct",
                )
            )

    bed_dets = _dedupe_anchor_detections(bed_dets, iou_threshold=0.6)

    # If no truck class exists, derive pseudo-truck anchors from bed boxes.
    if not truck_dets and bed_dets:
        pseudo_trucks: list[Detection] = []
        for bed in bed_dets:
            pseudo = expand_box_xyxy(
                bed.xyxy,
                x_scale=config.EXPAND_BED_TO_TRUCK_X,
                y_scale=config.EXPAND_BED_TO_TRUCK_Y,
                frame_w=w,
                frame_h=h,
            )
            pseudo_trucks.append(
                Detection(
                    xyxy=pseudo,
                    conf=max(0.0, bed.conf * 0.85),
                    cls_id=bed.cls_id,
                    cls_name=f"{bed.cls_name}_pseudo_truck",
                    source="pseudo_from_bed",
                    bed_box_xyxy=bed.xyxy,
                    bed_conf=bed.conf,
                )
            )
        return _dedupe_anchor_detections(pseudo_trucks, iou_threshold=0.7)

    truck_dets = _dedupe_anchor_detections(truck_dets, iou_threshold=0.7)
    if not truck_dets:
        return []

    # Attach best bed child to each truck anchor.
    for truck in truck_dets:
        best_bed = None
        best_key = (-1.0, -1.0)
        for bed in bed_dets:
            iou = iou_xyxy(truck.xyxy, bed.xyxy)
            if iou <= 0.0:
                bx, by = _bbox_center_xy(bed.xyxy)
                if not _point_in_box(bx, by, truck.xyxy):
                    continue
            key = (iou, bed.conf)
            if key > best_key:
                best_key = key
                best_bed = bed
        if best_bed is not None:
            truck.bed_box_xyxy = best_bed.xyxy
            truck.bed_conf = best_bed.conf

    return truck_dets


def _compute_centeredness(box_xyxy: tuple[int, int, int, int], frame_w: int, frame_h: int) -> float:
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nx = cx / max(1.0, float(frame_w))
    ny = cy / max(1.0, float(frame_h))
    dx = nx - 0.5
    dy = ny - 0.5
    dist = math.sqrt(dx * dx + dy * dy)
    max_dist = math.sqrt(0.5 * 0.5 + 0.5 * 0.5)
    centeredness = 1.0 - min(1.0, dist / max_dist)
    return centeredness


def _calc_candidate_score_center_only(centeredness: float, conf: float) -> float:
    # Requirement: choose the one frame where bed center is closest to image center.
    # Tie-break very lightly by confidence.
    return float(centeredness) + (float(conf) * 1e-6)


def _draw_active_tracks_overlay(
    display_frame,
    tracker: IoUTracker,
    frame_w: int,
    frame_h: int,
    min_bed_persist_frames: int,
) -> None:
    truck_box_thickness = 4
    bed_box_thickness = 4
    for track_id in sorted(tracker.active_tracks.keys()):
        track = tracker.active_tracks[track_id]
        x1, y1, x2, y2 = clamp_xyxy(*track.truck_box_xyxy, frame_w, frame_h)
        if x2 <= x1 or y2 <= y1:
            continue

        # Truck anchor (blue-ish)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 100, 0), truck_box_thickness)

        # Bed child (green) if available
        if track.bed_box_xyxy is not None:
            bx1, by1, bx2, by2 = clamp_xyxy(*track.bed_box_xyxy, frame_w, frame_h)
            if bx2 > bx1 and by2 > by1:
                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 220, 0), bed_box_thickness)

        warming = track.bed_hits < min_bed_persist_frames
        tentative = not track.is_confirmed
        stale = track.missed_count > 0
        if tentative:
            color = (120, 120, 120)
        elif warming:
            color = (140, 140, 140)
        elif stale:
            color = (0, 80, 255)
        else:
            color = (0, 200, 0)

        if tentative:
            label = f"ID:{track_id} tentative {track.total_hits}"
        elif warming:
            label = f"ID:{track_id} warming {track.bed_hits}/{min_bed_persist_frames}"
        else:
            label = f"ID:{track_id} conf:{track.last_conf:.2f} bed_hits:{track.bed_hits}"
            if stale:
                label += f" miss:{track.missed_count}"
        cv2.putText(
            display_frame,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )


def _candidate_to_overlay_image(candidate: CropCandidate, text: str, out_path: Path) -> str:
    crop = candidate.crop_bgr.copy()
    cv2.rectangle(crop, (0, 0), (crop.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(crop, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), crop)
    return str(out_path)


def _infer_coverage(label: str, pred_idx: int, covered_ids: list[int], uncovered_ids: list[int]) -> bool | None:
    if covered_ids and pred_idx in covered_ids:
        return True
    if uncovered_ids and pred_idx in uncovered_ids:
        return False
    n = _normalize_label(label)
    if "uncovered" in n or "partial" in n:
        return False
    if "covered" in n:
        return True
    return None


def _infer_irregular(label: str, pred_idx: int, irregular_ids: list[int], regular_ids: list[int]) -> bool | None:
    if irregular_ids and pred_idx in irregular_ids:
        return True
    if regular_ids and pred_idx in regular_ids:
        return False
    n = _normalize_label(label)
    if "irregular" in n:
        return True
    if "regular" in n:
        return False
    return None


def _infer_irregular_type(
    label: str,
    pred_idx: int,
    sand_ids: list[int],
    iron_ids: list[int],
    unknown_ids: list[int],
) -> str | None:
    formatted = _format_irregular_label(label)
    if sand_ids and pred_idx in sand_ids:
        return formatted if formatted else "sand_like"
    if iron_ids and pred_idx in iron_ids:
        return formatted if formatted else "ironbars_like"
    if unknown_ids and pred_idx in unknown_ids:
        return formatted if formatted else "unknown"

    n = _normalize_label(label)
    if "sand" in n:
        return formatted if formatted else "sand_like"
    if "iron" in n or "bar" in n:
        return formatted if formatted else "ironbars_like"
    if "irregular" in n or "unknown" in n:
        return formatted if formatted else "unknown"
    if formatted:
        return formatted
    return None


def _compose_covered_irregular_status(irregular_type: dict[str, Any] | None) -> str:
    if irregular_type is None:
        return "covered_irregular_unknown"
    irr = str(irregular_type.get("irregular_type", "unknown")).strip().lower()
    if irr == "":
        return "covered_irregular_unknown"
    if irr.startswith("covered_"):
        return irr
    return f"covered_irregular_{irr}"


def _parse_segmentation(result, names: dict[int, str]) -> dict[str, Any]:
    boxes = result.boxes
    masks = result.masks
    if boxes is None or len(boxes) == 0:
        return {
            "detections_count": 0,
            "labels": ["no_load_detected"],
            "confs": [],
            "areas_px": [],
            "dominant_label": "no_load_detected",
            "dominant_conf": 0.0,
        }

    detections_count = len(boxes)
    labels: list[str] = []
    confs: list[float] = []
    areas_px: list[int] = []

    mask_data = masks.data.cpu().numpy() if masks is not None and masks.data is not None else None
    for i in range(detections_count):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        label = _normalize_label(names.get(cls_id, f"class_{cls_id}")).replace(" ", "_")
        area_px = 0
        if mask_data is not None and i < mask_data.shape[0]:
            area_px = int((mask_data[i] > 0.5).sum())
        else:
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            area_px = max(0, int(round((x2 - x1) * (y2 - y1))))
        labels.append(label)
        confs.append(conf)
        areas_px.append(area_px)

    best_idx = max(range(len(labels)), key=lambda idx: areas_px[idx]) if labels else 0
    dominant_label = labels[best_idx] if labels else "no_load_detected"
    dominant_conf = confs[best_idx] if labels else 0.0
    return {
        "detections_count": detections_count,
        "labels": labels if labels else ["no_load_detected"],
        "confs": confs,
        "areas_px": areas_px,
        "dominant_label": dominant_label,
        "dominant_conf": float(dominant_conf),
    }


def _pipeline_conf_score(
    coverage: dict[str, Any] | None,
    shape: dict[str, Any] | None,
    irregular_type: dict[str, Any] | None,
    segmentation: dict[str, Any] | None,
) -> float:
    if irregular_type is not None:
        return float(irregular_type.get("pred_conf", 0.0))
    if shape is not None:
        return float(shape.get("pred_conf", 0.0))
    if segmentation is not None:
        return float(segmentation.get("dominant_conf", 0.0))
    if coverage is not None:
        return float(coverage.get("pred_conf", 0.0))
    return 0.0


def _build_result_summary(
    coverage: dict[str, Any] | None,
    shape: dict[str, Any] | None,
    irregular_type: dict[str, Any] | None,
    segmentation: dict[str, Any] | None,
) -> str:
    if coverage is None:
        return "unknown"

    is_cov = bool(coverage.get("is_fully_covered", False))
    if is_cov:
        if shape is None:
            return "covered_unknown_shape"
        if not bool(shape.get("is_irregular", False)):
            return "covered_regular"
        return _compose_covered_irregular_status(irregular_type)

    seg_label = "unknown"
    if segmentation is not None:
        seg_label = str(segmentation.get("dominant_label", "unknown"))
    return f"uncovered_or_partial_{seg_label}"


def _event_cover_status(
    coverage: dict[str, Any] | None,
    shape: dict[str, Any] | None,
    irregular_type: dict[str, Any] | None,
) -> str:
    if coverage is None:
        return "unknown"
    if not bool(coverage.get("is_fully_covered", False)):
        return "uncovered_or_partial"
    if shape is None:
        return "covered_unknown_shape"
    if bool(shape.get("is_irregular", False)):
        return _compose_covered_irregular_status(irregular_type)
    return "covered_regular"


def _event_material(segmentation: dict[str, Any] | None) -> str:
    if segmentation is None:
        return "n/a"
    labels = segmentation.get("labels", [])
    if isinstance(labels, list) and labels:
        unique = []
        for x in labels:
            s = str(x)
            if s not in unique:
                unique.append(s)
        return "|".join(unique) if unique else "n/a"
    dom = segmentation.get("dominant_label")
    return str(dom) if dom else "n/a"


def _format_event_console_line(
    ts: str,
    event_id: int,
    cover_status: str,
    material: str,
    fill_display: str,
    violation_display: str,
    fill_best_frame: int | None = None,
) -> str:
    line = (
        f"{ts} | TRUCK_ID={event_id} | COVER_STATUS={cover_status} | "
        f"MATERIAL={material} | FILL={fill_display} | VIOLATION={violation_display}"
    )
    if fill_best_frame is not None:
        line += f" | FILL_FRAME={fill_best_frame}"
    return line


def _event_violation_display(event: dict[str, Any]) -> str:
    violation = bool(event.get("violation", False))
    if not violation:
        return "under_investigation"

    coverage = event.get("coverage")
    shape = event.get("shape")
    irregular_type = event.get("irregular_type")
    from_phase4_cls = bool(
        isinstance(coverage, dict)
        and bool(coverage.get("is_fully_covered", False))
        and isinstance(shape, dict)
        and bool(shape.get("is_irregular", False))
        and (irregular_type is not None)
    )
    if from_phase4_cls:
        return "under_investigation"

    return "True"


def _event_fill_display(event: dict[str, Any]) -> str:
    fill_estimation = event.get("fill_estimation")
    if not isinstance(fill_estimation, dict):
        return "n/a"

    fill_percentage = fill_estimation.get("fill_percentage")
    status = str(fill_estimation.get("status", "")).strip().lower()
    if fill_percentage is None:
        return status if status else "n/a"

    try:
        return f"{float(fill_percentage):.2f}%"
    except Exception:
        return str(fill_percentage)


def _match_size_event(candidate_event: dict[str, Any], size_events: list[SizeEvent], used_indices: set[int]) -> SizeEvent | None:
    start_frame = int(candidate_event.get("start_frame", -1))
    end_frame = int(candidate_event.get("end_frame", -1))
    best_frame = int(candidate_event.get("best_frame", -1))
    best_match = None
    best_key = None
    for index, size_event in enumerate(size_events):
        if index in used_indices:
            continue
        overlap = max(0, min(end_frame, size_event.end_frame) - max(start_frame, size_event.start_frame) + 1)
        gap = min(abs(best_frame - size_event.best_frame), abs(start_frame - size_event.start_frame), abs(end_frame - size_event.end_frame))
        key = (1 if overlap > 0 else 0, overlap, -gap)
        if best_key is None or key > best_key:
            best_key = key
            best_match = (index, size_event)
    if best_match is None:
        return None
    used_indices.add(best_match[0])
    return best_match[1]


def _merge_size_result_into_event(event: dict[str, Any], size_event: SizeEvent | None) -> dict[str, Any]:
    if size_event is None:
        return event
    merged = dict(event)
    merged["fill_estimation"] = {
        "status": size_event.fill_status,
        "fill_percentage": size_event.fill_percentage,
        "raw_output": size_event.raw_output,
    }
    merged["fill_best_frame"] = int(size_event.best_frame)
    merged["fill_candidate_frames"] = [int(x) for x in size_event.history_frames]
    merged["fill_shortlist_frames"] = [int(x) for x in size_event.shortlist_frames]
    artifacts = dict(merged.get("artifacts", {}))
    artifacts["fill_frame"] = size_event.best_frame_path
    merged["artifacts"] = artifacts
    return merged


def _run_heavy_phases_on_crop(candidate: CropCandidate, models: ModelBundle, seg_conf_threshold: float) -> dict[str, Any]:
    crop = candidate.crop_bgr

    cls1_result = models.cls1.predict(source=crop, device="cpu", verbose=False)[0]
    cls1_idx = int(cls1_result.probs.top1)
    cls1_conf = float(cls1_result.probs.top1conf.item())
    cls1_label = models.cls1_names.get(cls1_idx, f"class_{cls1_idx}")
    is_fully_covered = _infer_coverage(cls1_label, cls1_idx, models.covered_ids, models.uncovered_ids)
    if is_fully_covered is None:
        is_fully_covered = True

    coverage = {
        "pred_idx": cls1_idx,
        "pred_label": cls1_label,
        "pred_conf": cls1_conf,
        "is_fully_covered": bool(is_fully_covered),
    }

    shape = None
    irregular_type = None
    segmentation = None
    fill_estimation = None
    violation = False

    if is_fully_covered:
        cls2_result = models.cls2.predict(source=crop, device="cpu", verbose=False)[0]
        cls2_idx = int(cls2_result.probs.top1)
        cls2_conf = float(cls2_result.probs.top1conf.item())
        cls2_label = models.cls2_names.get(cls2_idx, f"class_{cls2_idx}")
        is_irregular = _infer_irregular(cls2_label, cls2_idx, models.irregular_ids, models.regular_ids)
        if is_irregular is None:
            is_irregular = False

        shape = {
            "pred_idx": cls2_idx,
            "pred_label": cls2_label,
            "pred_conf": cls2_conf,
            "is_irregular": bool(is_irregular),
        }

        if is_irregular:
            cls3_result = models.cls3.predict(source=crop, device="cpu", verbose=False)[0]
            cls3_idx = int(cls3_result.probs.top1)
            cls3_conf = float(cls3_result.probs.top1conf.item())
            cls3_label = models.cls3_names.get(cls3_idx, f"class_{cls3_idx}")
            irr_type = _infer_irregular_type(
                cls3_label,
                cls3_idx,
                models.sand_ids,
                models.iron_ids,
                models.unknown_ids,
            )
            if irr_type is None:
                irr_type = _format_irregular_label(cls3_label)
            if irr_type == "":
                irr_type = "unknown"
            irregular_type = {
                "pred_idx": cls3_idx,
                "pred_label": cls3_label,
                "pred_conf": cls3_conf,
                "irregular_type": irr_type,
            }
            violation = True
        else:
            violation = False
    else:
        seg_result = models.seg.predict(source=crop, device="cpu", conf=seg_conf_threshold, verbose=False)[0]
        segmentation = _parse_segmentation(seg_result, models.seg_names)
        violation = True

    pipeline_conf = _pipeline_conf_score(coverage, shape, irregular_type, segmentation)
    return {
        "coverage": coverage,
        "shape": shape,
        "irregular_type": irregular_type,
        "segmentation": segmentation,
        "violation": violation,
        "pipeline_conf": pipeline_conf,
    }


def _load_models(
    detect_model_path: Path,
    cls1_model_path: Path,
    cls2_model_path: Path,
    cls3_model_path: Path,
    seg_model_path: Path,
    logger,
    bed_class_ids: list[int] | None,
    truck_class_ids: list[int] | None,
) -> ModelBundle:
    logger.info("Loading stream models on CPU")
    detect = YOLO(str(detect_model_path))
    cls1 = YOLO(str(cls1_model_path))
    cls2 = YOLO(str(cls2_model_path))
    cls3 = YOLO(str(cls3_model_path))
    seg = YOLO(str(seg_model_path))

    detect_names = _names_to_dict(detect.names)
    cls1_names = _names_to_dict(cls1.names)
    cls2_names = _names_to_dict(cls2.names)
    cls3_names = _names_to_dict(cls3.names)
    seg_names = _names_to_dict(seg.names)

    bed_ids, truck_ids = _resolve_bed_and_truck_ids(
        detect_names,
        bed_class_ids=bed_class_ids,
        truck_class_ids=truck_class_ids,
    )
    covered_ids = _resolve_ids_by_names(cls1_names, config.PHASE2_COVERED_CLASS_NAMES)
    uncovered_ids = _resolve_ids_by_names(cls1_names, config.PHASE2_UNCOVERED_CLASS_NAMES)
    irregular_ids = _resolve_ids_by_names(cls2_names, config.PHASE3_IRREGULAR_CLASS_NAMES)
    regular_ids = _resolve_ids_by_names(cls2_names, config.PHASE3_REGULAR_CLASS_NAMES)
    sand_ids = _resolve_ids_by_names(cls3_names, config.PHASE4_SANDLIKE_CLASS_NAMES)
    iron_ids = _resolve_ids_by_names(cls3_names, config.PHASE4_IRONBARS_CLASS_NAMES)
    unknown_ids = _resolve_ids_by_names(cls3_names, config.PHASE4_UNKNOWN_CLASS_NAMES)

    logger.info("Detection model classes: %s", detect_names)
    logger.info("Phase2 model classes: %s", cls1_names)
    logger.info("Phase3 model classes: %s", cls2_names)
    logger.info("Phase4 model classes: %s", cls3_names)
    logger.info("Phase5 model classes: %s", seg_names)
    logger.info(
        "Bed class ids: %s | Truck class ids: %s | explicit truck_class_ids=%s | explicit bed_class_ids=%s",
        bed_ids,
        truck_ids,
        truck_class_ids,
        bed_class_ids,
    )

    return ModelBundle(
        detect=detect,
        cls1=cls1,
        cls2=cls2,
        cls3=cls3,
        seg=seg,
        detect_names=detect_names,
        cls1_names=cls1_names,
        cls2_names=cls2_names,
        cls3_names=cls3_names,
        seg_names=seg_names,
        bed_ids=bed_ids,
        truck_ids=truck_ids,
        covered_ids=covered_ids,
        uncovered_ids=uncovered_ids,
        irregular_ids=irregular_ids,
        regular_ids=regular_ids,
        sand_ids=sand_ids,
        iron_ids=iron_ids,
        unknown_ids=unknown_ids,
    )


def _should_early_infer(track: TrackState, min_best_area: int, stable_frames: int) -> bool:
    best = track.best_candidate
    if best is None:
        return False
    if best.area < min_best_area:
        return False
    near_max = track.last_area >= int(0.95 * max(1, track.max_area_seen))
    return bool(track.stable_count >= stable_frames or near_max)


def _next_pending_candidate(track: TrackState) -> CropCandidate | None:
    for candidate in track.top_candidates:
        already = False
        for run in track.inference_runs:
            if int(run.get("candidate_frame", -1)) == int(candidate.frame_idx):
                already = True
                break
        if not already:
            return candidate
    return None


def _majority_vote_run(runs: list[dict[str, Any]]) -> tuple[dict[str, Any], str, dict[str, int]]:
    if not runs:
        raise ValueError("runs must not be empty")

    labels: list[str] = []
    for run in runs:
        label = _build_result_summary(
            coverage=run.get("coverage"),
            shape=run.get("shape"),
            irregular_type=run.get("irregular_type"),
            segmentation=run.get("segmentation"),
        )
        run["result_summary"] = label
        labels.append(label)

    counts = Counter(labels)
    winners = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_count = winners[0][1]
    top_labels = [lbl for lbl, c in winners if c == top_count]

    # Tie-break: choose label with highest average pipeline_conf.
    if len(top_labels) > 1:
        avg_conf_by_label: dict[str, float] = {}
        for lbl in top_labels:
            vals = [float(r.get("pipeline_conf", 0.0)) for r in runs if r.get("result_summary") == lbl]
            avg_conf_by_label[lbl] = (sum(vals) / len(vals)) if vals else 0.0
        chosen_label = max(top_labels, key=lambda lbl: avg_conf_by_label.get(lbl, 0.0))
    else:
        chosen_label = top_labels[0]

    label_runs = [r for r in runs if r.get("result_summary") == chosen_label]
    chosen_run = max(label_runs, key=lambda r: float(r.get("pipeline_conf", 0.0)))
    return chosen_run, chosen_label, dict(counts)




def _event_box_xyxy(event: dict[str, Any]) -> tuple[int, int, int, int] | None:
    box = event.get("best_box_xyxy")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    except Exception:
        return None


def _event_box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, (x2 - x1) * (y2 - y1))


def _event_center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return float(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def _find_dedup_target_event(
    candidate_event: dict[str, Any],
    recent_events: deque[dict[str, Any]],
    frame_shape: tuple[int, int] | None,
    window_frames: int,
    iou_threshold: float,
    center_dist_ratio: float,
) -> tuple[dict[str, Any], float, float] | None:
    candidate_box = _event_box_xyxy(candidate_event)
    if candidate_box is None:
        return None
    c_start = int(candidate_event.get("start_frame", -1))
    c_end = int(candidate_event.get("end_frame", -1))
    if c_end < 0:
        return None

    frame_h, frame_w = frame_shape if frame_shape is not None else (0, 0)
    frame_diag = max(1.0, float(math.sqrt((frame_w ** 2) + (frame_h ** 2))))
    max_center_dist = float(max(0.0, center_dist_ratio)) * frame_diag
    candidate_area = _event_box_area(candidate_box)

    for prev in reversed(recent_events):
        p_start = int(prev.get("start_frame", -1))
        p_end = int(prev.get("end_frame", -1))
        if p_end < 0:
            continue
        if (c_start - p_end) > window_frames:
            break
        if (p_start - c_end) > window_frames:
            continue

        prev_box = _event_box_xyxy(prev)
        if prev_box is None:
            continue

        iou = iou_xyxy(candidate_box, prev_box)
        dist = _event_center_distance(candidate_box, prev_box)
        prev_area = _event_box_area(prev_box)
        area_ratio = (
            float(min(candidate_area, prev_area)) / float(max(1, max(candidate_area, prev_area)))
        )
        if iou >= iou_threshold or (dist <= max_center_dist and area_ratio >= 0.60):
            return prev, float(iou), float(dist)
    return None


def _emit_event(event: dict[str, Any], events_jsonl_path: Path, logger) -> None:
    with events_jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cover_status = _event_cover_status(
        coverage=event.get("coverage"),
        shape=event.get("shape"),
        irregular_type=event.get("irregular_type"),
    )
    material = _event_material(event.get("segmentation"))
    fill_display = _event_fill_display(event)
    violation_display = _event_violation_display(event)
    event_line = _format_event_console_line(
        ts=ts,
        event_id=int(event["event_id"]),
        cover_status=cover_status,
        material=material,
        fill_display=fill_display,
        violation_display=violation_display,
        fill_best_frame=event.get("fill_best_frame") if event.get("fill_best_frame") is not None else None,
    )
    banner = "=" * max(100, len(event_line))
    print(f"\n\033[1m{banner}\n{event_line}\n{banner}\033[0m")
    logger.info(
        "Finalized event %s with %s heavy call(s), violation=%s vote=%s",
        event["event_id"],
        event.get("heavy_inference_calls", 0),
        event.get("violation", False),
        "on" if bool(event.get("vote", {}).get("enabled", False)) else "off",
    )


def _finalize_track_event(
    track: TrackState,
    models: ModelBundle,
    event_infer_mode: str,
    top2: bool,
    seg_conf_threshold: float,
    events_images_dir: Path,
    final_reason: str,
    vote_enable: bool,
    vote_every_n_frames: int,
    logger,
) -> dict[str, Any]:
    if vote_enable:
        vote_candidates = list(track.vote_candidates)
        if not vote_candidates and track.best_candidate is not None:
            vote_candidates = [track.best_candidate]
        track.inference_runs = []
        for candidate in vote_candidates:
            run = _run_heavy_phases_on_crop(candidate, models, seg_conf_threshold)
            track.inference_runs.append(
                {
                    "candidate_frame": candidate.frame_idx,
                    "candidate_score": candidate.score,
                    "candidate": candidate,
                    **run,
                }
            )
    elif event_infer_mode == "finalize":
        if track.best_candidate is not None:
            run = _run_heavy_phases_on_crop(track.best_candidate, models, seg_conf_threshold)
            track.inference_runs.append(
                {
                    "candidate_frame": track.best_candidate.frame_idx,
                    "candidate_score": track.best_candidate.score,
                    "candidate": track.best_candidate,
                    **run,
                }
            )
        if top2:
            pending = _next_pending_candidate(track)
            if pending is not None:
                run = _run_heavy_phases_on_crop(pending, models, seg_conf_threshold)
                track.inference_runs.append(
                    {
                        "candidate_frame": pending.frame_idx,
                        "candidate_score": pending.score,
                        "candidate": pending,
                        **run,
                    }
                )
    else:
        if not track.inference_runs and track.best_candidate is not None:
            run = _run_heavy_phases_on_crop(track.best_candidate, models, seg_conf_threshold)
            track.inference_runs.append(
                {
                    "candidate_frame": track.best_candidate.frame_idx,
                    "candidate_score": track.best_candidate.score,
                    "candidate": track.best_candidate,
                    **run,
                }
            )
        if top2 and len(track.inference_runs) < 2:
            pending = _next_pending_candidate(track)
            if pending is not None:
                run = _run_heavy_phases_on_crop(pending, models, seg_conf_threshold)
                track.inference_runs.append(
                    {
                        "candidate_frame": pending.frame_idx,
                        "candidate_score": pending.score,
                        "candidate": pending,
                        **run,
                    }
                )

    if not track.inference_runs:
        empty_event = {
            "event_id": track.track_id,
            "start_frame": track.start_frame,
            "end_frame": track.last_seen_frame,
            "duration_frames": max(1, track.last_seen_frame - track.start_frame + 1),
            "best_frame": None,
            "best_box_xyxy": None,
            "best_score": None,
            "coverage": None,
            "shape": None,
            "irregular_type": None,
            "segmentation": None,
            "fill_estimation": None,
            "violation": False,
            "final_reason": final_reason,
            "artifacts": {},
            "heavy_inference_calls": 0,
        }
        return empty_event

    if vote_enable:
        best_run, result_summary, vote_counts = _majority_vote_run(track.inference_runs)
    else:
        best_run = max(track.inference_runs, key=lambda r: float(r.get("pipeline_conf", 0.0)))
        result_summary = _build_result_summary(
            coverage=best_run.get("coverage"),
            shape=best_run.get("shape"),
            irregular_type=best_run.get("irregular_type"),
            segmentation=best_run.get("segmentation"),
        )
        vote_counts = {}
    selected_candidate: CropCandidate = best_run["candidate"]

    raw_crop_path = events_images_dir / f"event_{track.track_id:05d}_bed_crop.jpg"
    cv2.imwrite(str(raw_crop_path), selected_candidate.crop_bgr)
    raw_truck_crop_path = None
    if selected_candidate.truck_crop_bgr is not None and getattr(selected_candidate.truck_crop_bgr, "size", 0) > 0:
        raw_truck_crop_path = events_images_dir / f"event_{track.track_id:05d}_truck_crop.jpg"
        cv2.imwrite(str(raw_truck_crop_path), selected_candidate.truck_crop_bgr)
    overlay_text = f"event={track.track_id} {result_summary} violation={best_run.get('violation', False)}"
    overlay_path = events_images_dir / f"event_{track.track_id:05d}_overlay.jpg"
    _candidate_to_overlay_image(selected_candidate, overlay_text, overlay_path)

    event = {
        "event_id": track.track_id,
        "start_frame": track.start_frame,
        "end_frame": track.last_seen_frame,
        "duration_frames": max(1, track.last_seen_frame - track.start_frame + 1),
        "best_frame": selected_candidate.frame_idx,
        "best_box_xyxy": list(selected_candidate.xyxy),
        "best_score": float(selected_candidate.score),
        "coverage": best_run.get("coverage"),
        "shape": best_run.get("shape"),
        "irregular_type": best_run.get("irregular_type"),
        "segmentation": best_run.get("segmentation"),
        "fill_estimation": None,
        "violation": bool(best_run.get("violation", False)),
        "result_summary": result_summary,
        "final_reason": final_reason,
        "artifacts": {
            "best_image": track.best_image_path,
            "bed_crop": str(raw_crop_path),
            "truck_crop": str(raw_truck_crop_path) if raw_truck_crop_path is not None else None,
            "overlay_image": str(overlay_path),
        },
        "heavy_inference_calls": len(track.inference_runs),
        "vote": {
            "enabled": bool(vote_enable),
            "sample_every_frames": int(vote_every_n_frames),
            "samples_used": len(track.inference_runs),
            "counts": vote_counts,
            "winner": result_summary,
        },
    }
    return event


def run_stream_event(
    video_path: Path | str,
    detect_model_path: Path | str | None = None,
    cls1_model_path: Path | str | None = None,
    cls2_model_path: Path | str | None = None,
    cls3_model_path: Path | str | None = None,
    seg_model_path: Path | str | None = None,
    size_seg_model_path: Path | str | None = None,
    missed_M: int = config.STREAM_MISSED_M,
    iou_threshold: float = config.STREAM_IOU_THRESHOLD,
    merge_window_frames: int = config.STREAM_MERGE_WINDOW,
    merge_iou_threshold: float = config.STREAM_MERGE_IOU,
    merge_center_dist_ratio: float = config.STREAM_MERGE_CENTER_RATIO,
    edge_guard: bool = config.STREAM_EDGE_GUARD,
    edge_margin: int = config.STREAM_EDGE_MARGIN,
    event_infer_mode: str = config.STREAM_EVENT_INFER_MODE,
    top2: bool = config.STREAM_TOP2,
    every_n: int = config.STREAM_EVERY_N,
    max_detect_fps: float = config.STREAM_MAX_DETECT_FPS,
    size_sampling_fps: float = config.SIZE_SAMPLING_FPS,
    size_preview_every_samples: int = 1,
    detect_conf_threshold: float | None = None,
    seg_conf_threshold: float | None = None,
    min_best_area: int = config.STREAM_MIN_BEST_AREA,
    stable_frames: int = config.STREAM_STABLE_FRAMES,
    bed_class_ids: list[int] | None = None,
    truck_class_ids: list[int] | None = None,
    show_preview: bool = False,
    size_show_preview: bool | None = None,
    preview_scale: float = 1.0,
    preview_fullscreen: bool = False,
    summary_only: bool = False,
    min_event_hits: int = config.STREAM_MIN_EVENT_HITS,
    min_bed_persist_frames: int = config.STREAM_MIN_BED_PERSIST_FRAMES,
    vote_enable: bool = config.STREAM_VOTE_ENABLE,
    vote_every_n_frames: int = config.STREAM_VOTE_EVERY_N_FRAMES,
    vote_max_samples: int = config.STREAM_VOTE_MAX_SAMPLES,
    track_confirm_hits: int = config.STREAM_TRACK_CONFIRM_HITS,
    track_smooth_alpha: float = config.STREAM_TRACK_SMOOTH_ALPHA,
    track_deadband_px: float = config.STREAM_TRACK_DEADBAND_PX,
    track_max_step_px: float = config.STREAM_TRACK_MAX_STEP_PX,
    active_match_center_dist_ratio: float = config.STREAM_ACTIVE_MATCH_CENTER_RATIO,
    duplicate_iou_threshold: float = config.STREAM_DUPLICATE_IOU_THRESHOLD,
    active_duplicate_iou_threshold: float = config.STREAM_ACTIVE_DUPLICATE_IOU_THRESHOLD,
    active_duplicate_bed_iou_threshold: float = config.STREAM_ACTIVE_DUPLICATE_BED_IOU_THRESHOLD,
    startup_duplicate_window_frames: int = config.STREAM_STARTUP_DUPLICATE_WINDOW_FRAMES,
    startup_duplicate_iou_threshold: float = config.STREAM_STARTUP_DUPLICATE_IOU_THRESHOLD,
    startup_duplicate_bed_iou_threshold: float = config.STREAM_STARTUP_DUPLICATE_BED_IOU_THRESHOLD,
    startup_duplicate_center_dist_ratio: float = config.STREAM_STARTUP_DUPLICATE_CENTER_RATIO,
    startup_duplicate_overlap_ratio: float = config.STREAM_STARTUP_DUPLICATE_OVERLAP_RATIO,
    exit_duplicate_iou_threshold: float = config.STREAM_EXIT_DUPLICATE_IOU_THRESHOLD,
    exit_duplicate_bed_iou_threshold: float = config.STREAM_EXIT_DUPLICATE_BED_IOU_THRESHOLD,
    new_track_ignore_lower_ratio: float = config.STREAM_NEW_TRACK_IGNORE_LOWER_RATIO,
    event_dedup_enable: bool = config.STREAM_EVENT_DEDUP_ENABLE,
    event_dedup_window_frames: int = config.STREAM_EVENT_DEDUP_WINDOW_FRAMES,
    event_dedup_iou_threshold: float = config.STREAM_EVENT_DEDUP_IOU_THRESHOLD,
    event_dedup_center_dist_ratio: float = config.STREAM_EVENT_DEDUP_CENTER_DIST_RATIO,
    debug_tracking: bool = False,
    max_frames: int | None = None,
) -> bool:
    if size_show_preview is None:
        size_show_preview = show_preview
    size_preview_every_samples = max(1, int(size_preview_every_samples))

    ensure_dirs([config.OUTPUT_DIR, config.STREAM_EVENTS_IMAGES_DIR, config.PROJECT_ROOT / "logs"])
    logger = setup_logger("stream_event", config.PROJECT_ROOT / "logs" / "stream_event.log")
    collected_events: list[dict[str, Any]] = []
    if summary_only:
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                h.setLevel(logging.INFO)
            else:
                h.setLevel(logging.WARNING)

    if event_infer_mode not in {"finalize", "early"}:
        raise ValueError("event_infer_mode must be one of: finalize, early")
    if vote_enable and event_infer_mode != "finalize":
        logger.warning("vote_enable=1 works with finalize mode; forcing event_infer_mode=finalize.")
        event_infer_mode = "finalize"
    if top2:
        # Requirement: keep only one selected frame per truck (best centered frame).
        top2 = False
        logger.warning("top2 was requested but is disabled by centered-frame policy; forcing top2=0.")
    every_n = max(1, int(every_n))
    max_detect_fps = max(0.0, float(max_detect_fps))
    size_sampling_fps = max(0.1, float(size_sampling_fps))
    missed_M = max(1, int(missed_M))
    merge_window_frames = max(1, int(merge_window_frames))
    merge_iou_threshold = float(max(0.0, min(1.0, merge_iou_threshold)))
    merge_center_dist_ratio = float(max(0.0, merge_center_dist_ratio))
    edge_margin = max(0, int(edge_margin))
    edge_guard = bool(edge_guard)
    min_best_area = max(1, int(min_best_area))
    stable_frames = max(1, int(stable_frames))
    min_event_hits = max(1, int(min_event_hits))
    min_bed_persist_frames = max(1, int(min_bed_persist_frames))
    vote_enable = bool(vote_enable)
    vote_every_n_frames = max(1, int(vote_every_n_frames))
    vote_max_samples = max(1, int(vote_max_samples))
    track_confirm_hits = max(1, int(track_confirm_hits))
    track_smooth_alpha = float(max(0.01, min(1.0, track_smooth_alpha)))
    track_deadband_px = float(max(0.0, track_deadband_px))
    track_max_step_px = float(max(0.0, track_max_step_px))
    active_match_center_dist_ratio = float(max(0.0, active_match_center_dist_ratio))
    duplicate_iou_threshold = float(max(0.0, min(1.0, duplicate_iou_threshold)))
    active_duplicate_iou_threshold = float(max(0.0, min(1.0, active_duplicate_iou_threshold)))
    active_duplicate_bed_iou_threshold = float(max(0.0, min(1.0, active_duplicate_bed_iou_threshold)))
    startup_duplicate_window_frames = max(0, int(startup_duplicate_window_frames))
    startup_duplicate_iou_threshold = float(max(0.0, min(1.0, startup_duplicate_iou_threshold)))
    startup_duplicate_bed_iou_threshold = float(max(0.0, min(1.0, startup_duplicate_bed_iou_threshold)))
    startup_duplicate_center_dist_ratio = float(max(0.0, startup_duplicate_center_dist_ratio))
    startup_duplicate_overlap_ratio = float(max(0.0, min(1.0, startup_duplicate_overlap_ratio)))
    exit_duplicate_iou_threshold = float(max(0.0, min(1.0, exit_duplicate_iou_threshold)))
    exit_duplicate_bed_iou_threshold = float(max(0.0, min(1.0, exit_duplicate_bed_iou_threshold)))
    new_track_ignore_lower_ratio = float(max(0.0, min(0.95, new_track_ignore_lower_ratio)))
    event_dedup_enable = bool(event_dedup_enable)
    event_dedup_window_frames = max(1, int(event_dedup_window_frames))
    event_dedup_iou_threshold = float(max(0.0, min(1.0, event_dedup_iou_threshold)))
    event_dedup_center_dist_ratio = float(max(0.0, event_dedup_center_dist_ratio))

    detect_conf = config.DETECT_CONF_THRESHOLD if detect_conf_threshold is None else float(detect_conf_threshold)
    seg_conf = config.PHASE5_SEG_CONF_THRESHOLD if seg_conf_threshold is None else float(seg_conf_threshold)

    video = Path(video_path)
    if not video.is_absolute():
        video = (config.PROJECT_ROOT / video).resolve()
    if not video.exists():
        logger.error("Video path does not exist: %s", video)
        return False

    detect_path = _ensure_model_path(detect_model_path, config.DETECT_MODEL_PATH, "DETECT_MODEL_PATH")
    cls1_path = _ensure_model_path(cls1_model_path, config.CLS1_MODEL_PATH, "CLS1_MODEL_PATH")
    cls2_path = _ensure_model_path(cls2_model_path, config.CLS2_MODEL_PATH, "CLS2_MODEL_PATH")
    cls3_path = _ensure_model_path(cls3_model_path, config.CLS3_MODEL_PATH, "CLS3_MODEL_PATH")
    seg_path = _ensure_model_path(seg_model_path, config.SEG_MODEL_PATH, "SEG_MODEL_PATH")

    models = _load_models(
        detect_model_path=detect_path,
        cls1_model_path=cls1_path,
        cls2_model_path=cls2_path,
        cls3_model_path=cls3_path,
        seg_model_path=seg_path,
        logger=logger,
        bed_class_ids=bed_class_ids if bed_class_ids is not None else config.BED_CLASS_IDS,
        truck_class_ids=truck_class_ids if truck_class_ids is not None else config.TRUCK_CLASS_IDS,
    )

    events_jsonl = config.STREAM_EVENTS_JSONL
    events_images_dir = config.STREAM_EVENTS_IMAGES_DIR
    ensure_dirs([events_jsonl.parent, events_images_dir])

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        logger.error("Unable to open video: %s", video)
        return False

    preview_window_name = "stream_event"
    if show_preview:
        cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
        if preview_fullscreen:
            cv2.setWindowProperty(
                preview_window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    size_pipeline: OnlineSizeEventPipeline | None = None
    size_events: list[SizeEvent] = []
    used_size_indices: set[int] = set()
    try:
        size_truck_path = _ensure_model_path(None, config.SIZE_TRUCK_MODEL_PATH, "SIZE_TRUCK_MODEL_PATH")
        size_seg_path = _ensure_model_path(size_seg_model_path, config.SIZE_SEG_MODEL_PATH, "SIZE_SEG_MODEL_PATH")
        size_output_dir = config.SIZE_PIPELINE_OUTPUT_DIR / video.stem
        size_pipeline = OnlineSizeEventPipeline(
            video_path=video,
            truck_model_path=size_truck_path,
            size_model_path=size_seg_path,
            sampling_fps=size_sampling_fps,
            output_dir=size_output_dir,
            source_fps=video_fps or 25.0,
            show_preview=bool(size_show_preview),
            preview_scale=preview_scale,
            preview_every_samples=size_preview_every_samples,
        )
    except FileNotFoundError as exc:
        logger.warning("Independent size pipeline disabled: %s", exc)

    if max_detect_fps > 0.0 and video_fps > 0.0:
        auto_every_n = max(1, int(math.ceil(video_fps / max_detect_fps)))
        if auto_every_n > every_n:
            logger.info(
                "Applied max_detect_fps=%.2f with video_fps=%.2f -> every_n=%s (was %s)",
                max_detect_fps,
                video_fps,
                auto_every_n,
                every_n,
            )
            every_n = auto_every_n
    tracker = IoUTracker(
        iou_threshold=iou_threshold,
        missed_M=missed_M,
        track_confirm_hits=track_confirm_hits,
        top2=top2,
        smooth_alpha=track_smooth_alpha,
        smooth_deadband_px=track_deadband_px,
        smooth_max_step_px=track_max_step_px,
        merge_window_frames=merge_window_frames,
        merge_iou_threshold=merge_iou_threshold,
        merge_center_dist_ratio=merge_center_dist_ratio,
        active_match_center_dist_ratio=active_match_center_dist_ratio,
        duplicate_iou_threshold=duplicate_iou_threshold,
        active_duplicate_iou_threshold=active_duplicate_iou_threshold,
        active_duplicate_bed_iou_threshold=active_duplicate_bed_iou_threshold,
        startup_duplicate_window_frames=startup_duplicate_window_frames,
        startup_duplicate_iou_threshold=startup_duplicate_iou_threshold,
        startup_duplicate_bed_iou_threshold=startup_duplicate_bed_iou_threshold,
        startup_duplicate_center_dist_ratio=startup_duplicate_center_dist_ratio,
        startup_duplicate_overlap_ratio=startup_duplicate_overlap_ratio,
        exit_duplicate_iou_threshold=exit_duplicate_iou_threshold,
        exit_duplicate_bed_iou_threshold=exit_duplicate_bed_iou_threshold,
        new_track_ignore_lower_ratio=new_track_ignore_lower_ratio,
        edge_guard=edge_guard,
        edge_margin=edge_margin,
        debug_tracking=debug_tracking,
    )

    logger.info(
        "Starting stream_event on %s | every_n=%s | missed_M=%s | iou_threshold=%.3f | "
        "merge_window=%s merge_iou=%.3f merge_center_ratio=%.3f edge_guard=%s edge_margin=%s | "
        "infer_mode=%s | top2=%s | vote=%s vote_every=%s vote_max_samples=%s | "
        "confirm_hits=%s | dedup=%s dedup_window=%s dedup_iou=%.2f dedup_center_ratio=%.3f | "
        "smooth_alpha=%.2f deadband=%.1f max_step=%.1f active_center_ratio=%.3f "
        "dup_iou=%.2f max_detect_fps=%.2f ignore_new_lower=%.2f",
        video,
        every_n,
        missed_M,
        iou_threshold,
        merge_window_frames,
        merge_iou_threshold,
        merge_center_dist_ratio,
        int(edge_guard),
        edge_margin,
        event_infer_mode,
        int(top2),
        int(vote_enable),
        vote_every_n_frames,
        vote_max_samples,
        track_confirm_hits,
        int(event_dedup_enable),
        event_dedup_window_frames,
        event_dedup_iou_threshold,
        event_dedup_center_dist_ratio,
        track_smooth_alpha,
        track_deadband_px,
        track_max_step_px,
        active_match_center_dist_ratio,
        duplicate_iou_threshold,
        max_detect_fps,
        new_track_ignore_lower_ratio,
    )

    frame_idx = -1
    events_finalized = 0
    events_deduped = 0
    heavy_inference_calls_total = 0
    phase_calls_by_event: list[int] = []
    recent_emitted_events: deque[dict[str, Any]] = deque(maxlen=512)
    detection_frames = 0
    max_frame_seen = -1
    last_frame_shape: tuple[int, int] | None = None

    def emit_event_with_current_size_result(event: dict[str, Any]) -> dict[str, Any]:
        if size_pipeline is not None:
            size_events.extend(size_pipeline.pop_completed_events())
        size_event = _match_size_event(event, size_events, used_size_indices)
        if size_event is None and size_pipeline is not None:
            size_event = size_pipeline.force_finalize_best_match(event)
        merged_event = _merge_size_result_into_event(event, size_event)
        _emit_event(merged_event, events_jsonl_path=events_jsonl, logger=logger)
        return merged_event

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            max_frame_seen = frame_idx
            if size_pipeline is not None:
                size_pipeline.process_frame(frame, frame_idx)
            if max_frames is not None and frame_idx >= int(max_frames):
                logger.info("Stopping early at frame=%s due to max_frames=%s", frame_idx, max_frames)
                break
            display_frame = frame.copy() if show_preview else None

            if frame_idx % every_n != 0:
                if show_preview and display_frame is not None:
                    frame_h, frame_w = display_frame.shape[:2]
                    _draw_active_tracks_overlay(
                        display_frame=display_frame,
                        tracker=tracker,
                        frame_w=frame_w,
                        frame_h=frame_h,
                        min_bed_persist_frames=min_bed_persist_frames,
                    )
                    cv2.putText(
                        display_frame,
                        f"frame={frame_idx} (skip detect) active={len(tracker.active_tracks)} events={events_finalized}",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    if preview_scale != 1.0:
                        display_frame = cv2.resize(
                            display_frame,
                            None,
                            fx=float(preview_scale),
                            fy=float(preview_scale),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    cv2.imshow(preview_window_name, display_frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        logger.info("Preview interrupted by user (q).")
                        break
                continue

            detection_frames += 1
            detect_result = models.detect.predict(source=frame, device="cpu", conf=detect_conf, verbose=False)[0]
            frame_h, frame_w = frame.shape[:2]
            last_frame_shape = (frame_h, frame_w)
            detections = _extract_anchor_detections(
                result=detect_result,
                names=models.detect_names,
                truck_ids=models.truck_ids,
                bed_ids=models.bed_ids,
            )
            active_before = set(tracker.active_tracks.keys())
            active_tracks, finalized_tracks, merge_logs = tracker.update(
                detections=detections,
                frame_idx=frame_idx,
                frame_shape=(frame_h, frame_w),
            )
            for msg in merge_logs:
                logger.info(msg)

            for new_id in sorted(set(active_tracks.keys()) - active_before):
                logger.info("Assigned TRACK_ID=%s at frame=%s", new_id, frame_idx)

            for track in list(active_tracks.values()):
                if track.last_seen_frame != frame_idx:
                    continue
                if not track.is_confirmed:
                    continue
                if track.bed_box_xyxy is None:
                    continue

                bx1, by1, bx2, by2 = clamp_xyxy(*track.bed_box_xyxy, frame_w, frame_h)
                if bx2 <= bx1 or by2 <= by1:
                    continue
                tx1, ty1, tx2, ty2 = clamp_xyxy(*track.truck_box_xyxy, frame_w, frame_h)
                if tx2 <= tx1 or ty2 <= ty1:
                    continue
                raw_tx1, raw_ty1, raw_tx2, raw_ty2 = clamp_xyxy(*track.raw_truck_box_xyxy, frame_w, frame_h)
                truck_crop = frame[raw_ty1:raw_ty2, raw_tx1:raw_tx2]
                crop = frame[by1:by2, bx1:bx2]
                if crop is None or crop.size == 0:
                    continue

                # Gate: do not consider bed for best-crop/event logic until it persists enough frames.
                if track.bed_hits < min_bed_persist_frames:
                    continue

                area = max(0, (bx2 - bx1) * (by2 - by1))
                if area < min_best_area:
                    track.stable_count = 0
                    continue
                centeredness = _compute_centeredness((bx1, by1, bx2, by2), frame_w, frame_h)
                score = _calc_candidate_score_center_only(centeredness=centeredness, conf=track.bed_conf)
                candidate = CropCandidate(
                    frame_idx=frame_idx,
                    xyxy=(bx1, by1, bx2, by2),
                    area=area,
                    conf=track.bed_conf,
                    centeredness=centeredness,
                    score=score,
                    crop_bgr=crop.copy(),
                    truck_xyxy=(raw_tx1, raw_ty1, raw_tx2, raw_ty2),
                    truck_crop_bgr=truck_crop.copy(),
                )
                best_updated = tracker.add_candidate(track.track_id, candidate)
                if vote_enable:
                    tracker.add_vote_candidate(
                        track_id=track.track_id,
                        candidate=candidate,
                        sample_every_frames=vote_every_n_frames,
                        max_samples=vote_max_samples,
                    )

                track.stable_count += 1

                if best_updated:
                    debug_frame = frame.copy()
                    cv2.rectangle(debug_frame, (tx1, ty1), (tx2, ty2), (255, 100, 0), 4)
                    cv2.rectangle(debug_frame, (bx1, by1), (bx2, by2), (0, 220, 0), 4)
                    cv2.putText(
                        debug_frame,
                        f"track={track.track_id} score={score:.3f}",
                        (bx1, max(15, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    best_image_path = events_images_dir / f"event_{track.track_id:05d}_best.jpg"
                    cv2.imwrite(str(best_image_path), debug_frame)
                    track.best_image_path = str(best_image_path)

            if event_infer_mode == "early":
                for track in list(active_tracks.values()):
                    if not track.is_confirmed:
                        continue
                    if track.bed_hits < min_bed_persist_frames:
                        continue
                    if not _should_early_infer(track, min_best_area=min_best_area, stable_frames=stable_frames):
                        continue
                    if not top2 and len(track.inference_runs) >= 1:
                        continue
                    if top2 and len(track.inference_runs) >= 2:
                        continue
                    pending = _next_pending_candidate(track)
                    if pending is None:
                        continue
                    run = _run_heavy_phases_on_crop(pending, models, seg_conf)
                    track.inference_runs.append(
                        {
                            "candidate_frame": pending.frame_idx,
                            "candidate_score": pending.score,
                            "candidate": pending,
                            **run,
                        }
                    )
                    logger.info(
                        "Early heavy inference for track=%s frame=%s (run_count=%s)",
                        track.track_id,
                        pending.frame_idx,
                        len(track.inference_runs),
                    )

            for track in finalized_tracks:
                if not track.is_confirmed:
                    logger.info(
                        "Dropping unconfirmed track %s (state=%s, hits=%s/%s)",
                        track.track_id,
                        track.track_state,
                        track.total_hits,
                        track.bed_hits,
                    )
                    continue
                if track.total_hits < min_event_hits or track.bed_hits < min_bed_persist_frames:
                    logger.info(
                        "Dropping short/noisy track %s (truck_hits=%s, bed_hits=%s, min_event_hits=%s, min_bed_persist_frames=%s)",
                        track.track_id,
                        track.total_hits,
                        track.bed_hits,
                        min_event_hits,
                        min_bed_persist_frames,
                    )
                    continue
                best_area = int(track.best_candidate.area) if track.best_candidate is not None else 0
                if track.best_candidate is None or best_area < min_best_area:
                    logger.info(
                        "Dropping low-quality track %s (best_area=%s, min_best_area=%s)",
                        track.track_id,
                        best_area,
                        min_best_area,
                    )
                    continue
                event = _finalize_track_event(
                    track=track,
                    models=models,
                    event_infer_mode=event_infer_mode,
                    top2=bool(top2),
                    seg_conf_threshold=seg_conf,
                    events_images_dir=events_images_dir,
                    final_reason="missed_M_frames",
                    vote_enable=vote_enable,
                    vote_every_n_frames=vote_every_n_frames,
                    logger=logger,
                )
                dedup_target = None
                if event_dedup_enable:
                    dedup_target = _find_dedup_target_event(
                        candidate_event=event,
                        recent_events=recent_emitted_events,
                        frame_shape=last_frame_shape,
                        window_frames=event_dedup_window_frames,
                        iou_threshold=event_dedup_iou_threshold,
                        center_dist_ratio=event_dedup_center_dist_ratio,
                    )
                if dedup_target is not None:
                    prev_event, d_iou, d_dist = dedup_target
                    events_deduped += 1
                    logger.info(
                        "EVENT_DEDUP dropped event_id=%s as duplicate of event_id=%s (iou=%.3f, center_dist=%.1f)",
                        event.get("event_id"),
                        prev_event.get("event_id"),
                        d_iou,
                        d_dist,
                    )
                    continue
                emitted_event = emit_event_with_current_size_result(event)
                recent_emitted_events.append(emitted_event)
                if recent_emitted_events:
                    newest_end = int(emitted_event.get("end_frame", -1))
                    while recent_emitted_events:
                        oldest_end = int(recent_emitted_events[0].get("end_frame", -1))
                        if newest_end - oldest_end <= event_dedup_window_frames:
                            break
                        recent_emitted_events.popleft()
                events_finalized += 1
                heavy_calls = int(event.get("heavy_inference_calls", 0))
                heavy_inference_calls_total += heavy_calls
                phase_calls_by_event.append(heavy_calls)

            if show_preview and display_frame is not None:
                _draw_active_tracks_overlay(
                    display_frame=display_frame,
                    tracker=tracker,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    min_bed_persist_frames=min_bed_persist_frames,
                )
                status = (
                    f"frame={frame_idx} active={len(tracker.active_tracks)} "
                    f"events={events_finalized} mode={event_infer_mode} every_n={every_n}"
                )
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 36), (0, 0, 0), -1)
                cv2.putText(
                    display_frame,
                    status,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if preview_scale != 1.0:
                    display_frame = cv2.resize(
                        display_frame,
                        None,
                        fx=float(preview_scale),
                        fy=float(preview_scale),
                        interpolation=cv2.INTER_LINEAR,
                    )
                cv2.imshow(preview_window_name, display_frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    logger.info("Preview interrupted by user (q).")
                    break

        if max_frame_seen >= 0:
            for track_id in sorted(list(tracker.active_tracks.keys())):
                track = tracker.active_tracks.get(track_id)
                if track is None:
                    continue
                if not track.is_confirmed:
                    logger.info(
                        "Dropping unconfirmed track %s at EOF (state=%s, hits=%s/%s)",
                        track.track_id,
                        track.track_state,
                        track.total_hits,
                        track.bed_hits,
                    )
                    tracker.active_tracks.pop(track_id, None)
                    continue
                if track.total_hits < min_event_hits or track.bed_hits < min_bed_persist_frames:
                    logger.info(
                        "Dropping short/noisy track %s at EOF (truck_hits=%s, bed_hits=%s, min_event_hits=%s, min_bed_persist_frames=%s)",
                        track.track_id,
                        track.total_hits,
                        track.bed_hits,
                        min_event_hits,
                        min_bed_persist_frames,
                    )
                    tracker.active_tracks.pop(track_id, None)
                    continue
                best_area = int(track.best_candidate.area) if track.best_candidate is not None else 0
                if track.best_candidate is None or best_area < min_best_area:
                    logger.info(
                        "Dropping low-quality track %s at EOF (best_area=%s, min_best_area=%s)",
                        track.track_id,
                        best_area,
                        min_best_area,
                    )
                    tracker.active_tracks.pop(track_id, None)
                    continue
                event = _finalize_track_event(
                    track=track,
                    models=models,
                    event_infer_mode=event_infer_mode,
                    top2=bool(top2),
                    seg_conf_threshold=seg_conf,
                    events_images_dir=events_images_dir,
                    final_reason="end_of_video",
                    vote_enable=vote_enable,
                    vote_every_n_frames=vote_every_n_frames,
                    logger=logger,
                )
                dedup_target = None
                if event_dedup_enable:
                    dedup_target = _find_dedup_target_event(
                        candidate_event=event,
                        recent_events=recent_emitted_events,
                        frame_shape=last_frame_shape,
                        window_frames=event_dedup_window_frames,
                        iou_threshold=event_dedup_iou_threshold,
                        center_dist_ratio=event_dedup_center_dist_ratio,
                    )
                if dedup_target is not None:
                    prev_event, d_iou, d_dist = dedup_target
                    events_deduped += 1
                    logger.info(
                        "EVENT_DEDUP dropped EOF event_id=%s as duplicate of event_id=%s (iou=%.3f, center_dist=%.1f)",
                        event.get("event_id"),
                        prev_event.get("event_id"),
                        d_iou,
                        d_dist,
                    )
                    tracker.active_tracks.pop(track_id, None)
                    continue
                emitted_event = emit_event_with_current_size_result(event)
                recent_emitted_events.append(emitted_event)
                if recent_emitted_events:
                    newest_end = int(emitted_event.get("end_frame", -1))
                    while recent_emitted_events:
                        oldest_end = int(recent_emitted_events[0].get("end_frame", -1))
                        if newest_end - oldest_end <= event_dedup_window_frames:
                            break
                        recent_emitted_events.popleft()
                events_finalized += 1
                heavy_calls = int(event.get("heavy_inference_calls", 0))
                heavy_inference_calls_total += heavy_calls
                phase_calls_by_event.append(heavy_calls)
                tracker.active_tracks.pop(track_id, None)

    finally:
        cap.release()
        if size_pipeline is not None:
            size_pipeline.close()
        if show_preview:
            cv2.destroyAllWindows()

    avg_heavy = (sum(phase_calls_by_event) / len(phase_calls_by_event)) if phase_calls_by_event else 0.0
    logger.info(
        "stream_event complete | detection_frames=%s total_tracks_created=%s merges=%s events_finalized=%s deduped=%s "
        "heavy_calls_total=%s avg_heavy=%.3f",
        detection_frames,
        tracker.total_tracks_created,
        tracker.total_merges,
        events_finalized,
        events_deduped,
        heavy_inference_calls_total,
        avg_heavy,
    )
    return True
