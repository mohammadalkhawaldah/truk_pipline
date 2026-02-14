from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from src import config
from src.tracker import CropCandidate, Detection, IoUTracker, TrackState, iou_xyxy
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


def _resolve_bed_and_truck_ids(names: dict[int, str], bed_class_ids: list[int] | None) -> tuple[list[int], list[int]]:
    available_ids = set(names.keys())
    resolved_bed_ids: list[int] = []
    if bed_class_ids:
        resolved_bed_ids = sorted([int(i) for i in bed_class_ids if int(i) in available_ids])
    if not resolved_bed_ids:
        resolved_bed_ids = _resolve_ids_by_names(names, config.BED_CLASS_NAMES)
    resolved_truck_ids = _resolve_ids_by_names(names, config.TRUCK_FALLBACK_CLASS_NAMES)
    return resolved_bed_ids, resolved_truck_ids


def _heuristic_bed_from_truck_bbox(xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    width = x2 - x1
    height = y2 - y1
    hx1 = x1 + config.HEURISTIC_X_START_RATIO * width
    hx2 = x1 + config.HEURISTIC_X_END_RATIO * width
    hy1 = y1 + config.HEURISTIC_Y_START_RATIO * height
    hy2 = y1 + config.HEURISTIC_Y_END_RATIO * height
    return hx1, hy1, hx2, hy2


def _extract_bed_detections(
    result,
    names: dict[int, str],
    bed_ids: list[int],
    truck_ids: list[int],
) -> list[Detection]:
    orig = result.orig_img
    h, w = orig.shape[:2]
    boxes = result.boxes
    detections: list[Detection] = []

    if boxes is None or len(boxes) == 0:
        return detections

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        if bed_ids and cls_id not in bed_ids:
            continue
        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        cx1, cy1, cx2, cy2 = clamp_xyxy(x1, y1, x2, y2, w, h)
        detections.append(
            Detection(
                xyxy=(cx1, cy1, cx2, cy2),
                conf=conf,
                cls_id=cls_id,
                cls_name=names.get(cls_id, f"class_{cls_id}"),
                source="direct",
            )
        )

    if detections:
        return detections

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        if truck_ids and cls_id not in truck_ids:
            continue
        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        tx1, ty1, tx2, ty2 = clamp_xyxy(x1, y1, x2, y2, w, h)
        hx1, hy1, hx2, hy2 = _heuristic_bed_from_truck_bbox((x1, y1, x2, y2))
        cx1, cy1, cx2, cy2 = clamp_xyxy(hx1, hy1, hx2, hy2, w, h)
        detections.append(
            Detection(
                xyxy=(cx1, cy1, cx2, cy2),
                conf=max(0.0, conf * 0.80),
                cls_id=cls_id,
                cls_name=f"{names.get(cls_id, f'class_{cls_id}')}_heuristic",
                source="heuristic",
                truck_box_xyxy=(tx1, ty1, tx2, ty2),
            )
        )

    return detections


def _extract_truck_boxes(result, truck_ids: list[int]) -> list[tuple[int, int, int, int]]:
    orig = result.orig_img
    h, w = orig.shape[:2]
    boxes = result.boxes
    truck_boxes: list[tuple[int, int, int, int]] = []
    if boxes is None or len(boxes) == 0:
        return truck_boxes

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        if truck_ids and cls_id not in truck_ids:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        truck_boxes.append(clamp_xyxy(x1, y1, x2, y2, w, h))
    return truck_boxes


def _bbox_center_xy(box_xyxy: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _point_in_box(px: float, py: float, box_xyxy: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box_xyxy
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def _filter_beds_inside_trucks(
    bed_detections: list[Detection],
    truck_boxes: list[tuple[int, int, int, int]],
) -> list[Detection]:
    if not truck_boxes:
        return []

    kept: list[Detection] = []
    for det in bed_detections:
        cx, cy = _bbox_center_xy(det.xyxy)
        containing = [tbox for tbox in truck_boxes if _point_in_box(cx, cy, tbox)]
        if containing:
            best_truck = min(
                containing,
                key=lambda b: (cx - _bbox_center_xy(b)[0]) ** 2 + (cy - _bbox_center_xy(b)[1]) ** 2,
            )
            det.truck_box_xyxy = best_truck
            kept.append(det)
    return kept


def _dedupe_bed_detections(detections: list[Detection], iou_threshold: float = 0.6) -> list[Detection]:
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d.conf, reverse=True)
    kept: list[Detection] = []
    for det in sorted_dets:
        if any(iou_xyxy(det.xyxy, k.xyxy) >= iou_threshold for k in kept):
            continue
        kept.append(det)
    return kept


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
    for track_id in sorted(tracker.active_tracks.keys()):
        track = tracker.active_tracks[track_id]
        x1, y1, x2, y2 = clamp_xyxy(*track.last_box_xyxy, frame_w, frame_h)
        if x2 <= x1 or y2 <= y1:
            continue

        if track.smooth_truck_box_xyxy_f is not None:
            tx1, ty1, tx2, ty2 = (
                int(round(track.smooth_truck_box_xyxy_f[0])),
                int(round(track.smooth_truck_box_xyxy_f[1])),
                int(round(track.smooth_truck_box_xyxy_f[2])),
                int(round(track.smooth_truck_box_xyxy_f[3])),
            )
            tx1, ty1, tx2, ty2 = clamp_xyxy(tx1, ty1, tx2, ty2, frame_w, frame_h)
            if tx2 > tx1 and ty2 > ty1:
                cv2.rectangle(display_frame, (tx1, ty1), (tx2, ty2), (255, 100, 0), 2)

        warming = track.total_hits < min_bed_persist_frames
        stale = track.missed_count > 0
        if warming:
            color = (140, 140, 140)
        elif stale:
            color = (0, 200, 255)
        else:
            color = (0, 200, 0)

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        if warming:
            label = f"ID:{track_id} warming {track.total_hits}/{min_bed_persist_frames}"
        else:
            label = f"ID:{track_id} conf:{track.last_conf:.2f}"
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
    cv2.putText(crop, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
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
    if sand_ids and pred_idx in sand_ids:
        return "sand_like"
    if iron_ids and pred_idx in iron_ids:
        return "ironbars_like"
    if unknown_ids and pred_idx in unknown_ids:
        return "unknown"

    n = _normalize_label(label)
    if "sand" in n:
        return "sand_like"
    if "iron" in n or "bar" in n:
        return "ironbars_like"
    if "irregular" in n or "unknown" in n:
        return "unknown"
    return None


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
        if irregular_type is None:
            return "covered_irregular_unknown"
        return f"covered_irregular_{irregular_type.get('irregular_type', 'unknown')}"

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
        if irregular_type is None:
            return "covered_irregular"
        return f"covered_irregular_{irregular_type.get('irregular_type', 'unknown')}"
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
    violation: bool,
) -> str:
    return (
        f"{ts} | TRUCK_ID={event_id} | COVER_STATUS={cover_status} | "
        f"MATERIAL={material} | VIOLATION={violation}"
    )


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

    bed_ids, truck_ids = _resolve_bed_and_truck_ids(detect_names, bed_class_ids)
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
    logger.info("Bed class ids: %s | Truck fallback ids: %s", bed_ids, truck_ids)

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


def _finalize_track_event(
    track: TrackState,
    models: ModelBundle,
    event_infer_mode: str,
    top2: bool,
    seg_conf_threshold: float,
    events_jsonl_path: Path,
    events_images_dir: Path,
    final_reason: str,
    logger,
) -> dict[str, Any]:
    if event_infer_mode == "finalize":
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
            "violation": False,
            "final_reason": final_reason,
            "artifacts": {},
            "heavy_inference_calls": 0,
        }
        with events_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(empty_event) + "\n")
        return empty_event

    best_run = max(track.inference_runs, key=lambda r: float(r.get("pipeline_conf", 0.0)))
    selected_candidate: CropCandidate = best_run["candidate"]

    result_summary = _build_result_summary(
        coverage=best_run.get("coverage"),
        shape=best_run.get("shape"),
        irregular_type=best_run.get("irregular_type"),
        segmentation=best_run.get("segmentation"),
    )
    raw_crop_path = events_images_dir / f"event_{track.track_id:05d}_bed_crop.jpg"
    cv2.imwrite(str(raw_crop_path), selected_candidate.crop_bgr)
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
        "violation": bool(best_run.get("violation", False)),
        "result_summary": result_summary,
        "final_reason": final_reason,
        "artifacts": {
            "best_image": track.best_image_path,
            "bed_crop": str(raw_crop_path),
            "overlay_image": str(overlay_path),
        },
        "heavy_inference_calls": len(track.inference_runs),
    }
    with events_jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cover_status = _event_cover_status(
        coverage=event.get("coverage"),
        shape=event.get("shape"),
        irregular_type=event.get("irregular_type"),
    )
    material = _event_material(event.get("segmentation"))
    event_line = _format_event_console_line(
        ts=ts,
        event_id=int(event["event_id"]),
        cover_status=cover_status,
        material=material,
        violation=bool(event["violation"]),
    )
    banner = "=" * max(100, len(event_line))
    print(f"\n\033[1m{banner}\n{event_line}\n{banner}\033[0m")
    logger.info(
        "Finalized event %s with %s heavy call(s), violation=%s",
        event["event_id"],
        event["heavy_inference_calls"],
        event["violation"],
    )
    return event


def run_stream_event(
    video_path: Path | str,
    detect_model_path: Path | str | None = None,
    cls1_model_path: Path | str | None = None,
    cls2_model_path: Path | str | None = None,
    cls3_model_path: Path | str | None = None,
    seg_model_path: Path | str | None = None,
    missed_M: int = config.STREAM_MISSED_M,
    iou_threshold: float = config.STREAM_IOU_THRESHOLD,
    event_infer_mode: str = config.STREAM_EVENT_INFER_MODE,
    top2: bool = config.STREAM_TOP2,
    every_n: int = config.STREAM_EVERY_N,
    max_detect_fps: float = config.STREAM_MAX_DETECT_FPS,
    detect_conf_threshold: float | None = None,
    seg_conf_threshold: float | None = None,
    min_best_area: int = config.STREAM_MIN_BEST_AREA,
    stable_frames: int = config.STREAM_STABLE_FRAMES,
    bed_class_ids: list[int] | None = None,
    show_preview: bool = False,
    preview_scale: float = 1.0,
    preview_fullscreen: bool = False,
    summary_only: bool = False,
    min_event_hits: int = config.STREAM_MIN_EVENT_HITS,
    min_bed_persist_frames: int = config.STREAM_MIN_BED_PERSIST_FRAMES,
    track_smooth_alpha: float = config.STREAM_TRACK_SMOOTH_ALPHA,
    track_deadband_px: float = config.STREAM_TRACK_DEADBAND_PX,
    track_max_step_px: float = config.STREAM_TRACK_MAX_STEP_PX,
) -> bool:
    ensure_dirs([config.OUTPUT_DIR, config.STREAM_EVENTS_IMAGES_DIR, config.PROJECT_ROOT / "logs"])
    logger = setup_logger("stream_event", config.PROJECT_ROOT / "logs" / "stream_event.log")
    if summary_only:
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                h.setLevel(logging.INFO)
            else:
                h.setLevel(logging.WARNING)

    if event_infer_mode not in {"finalize", "early"}:
        raise ValueError("event_infer_mode must be one of: finalize, early")
    if top2:
        # Requirement: keep only one selected frame per truck (best centered frame).
        top2 = False
        logger.warning("top2 was requested but is disabled by centered-frame policy; forcing top2=0.")
    every_n = max(1, int(every_n))
    max_detect_fps = max(0.0, float(max_detect_fps))
    missed_M = max(1, int(missed_M))
    min_best_area = max(1, int(min_best_area))
    stable_frames = max(1, int(stable_frames))
    min_event_hits = max(1, int(min_event_hits))
    min_bed_persist_frames = max(1, int(min_bed_persist_frames))
    track_smooth_alpha = float(max(0.01, min(1.0, track_smooth_alpha)))
    track_deadband_px = float(max(0.0, track_deadband_px))
    track_max_step_px = float(max(0.0, track_max_step_px))
    effective_min_hits = max(min_event_hits, min_bed_persist_frames)

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
        top2=top2,
        smooth_alpha=track_smooth_alpha,
        smooth_deadband_px=track_deadband_px,
        smooth_max_step_px=track_max_step_px,
    )

    logger.info(
        "Starting stream_event on %s | every_n=%s | missed_M=%s | iou_threshold=%.3f | infer_mode=%s | top2=%s "
        "| smooth_alpha=%.2f deadband=%.1f max_step=%.1f max_detect_fps=%.2f",
        video,
        every_n,
        missed_M,
        iou_threshold,
        event_infer_mode,
        int(top2),
        track_smooth_alpha,
        track_deadband_px,
        track_max_step_px,
        max_detect_fps,
    )

    frame_idx = -1
    events_finalized = 0
    heavy_inference_calls_total = 0
    phase_calls_by_event: list[int] = []
    detection_frames = 0
    max_frame_seen = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            max_frame_seen = frame_idx
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
                        (0, 255, 255),
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
            truck_boxes = _extract_truck_boxes(detect_result, models.truck_ids)
            raw_bed_detections = _extract_bed_detections(
                result=detect_result,
                names=models.detect_names,
                bed_ids=models.bed_ids,
                truck_ids=models.truck_ids,
            )
            detections = _filter_beds_inside_trucks(raw_bed_detections, truck_boxes)
            detections = _dedupe_bed_detections(detections, iou_threshold=0.6)

            matched = tracker.update(detections=detections, frame_idx=frame_idx)

            frame_h, frame_w = frame.shape[:2]
            for track_id, det, is_new in matched:
                track = tracker.active_tracks.get(track_id)
                if track is None:
                    continue
                if is_new:
                    logger.info("Assigned TRACK_ID=%s at frame=%s", track_id, frame_idx)

                x1, y1, x2, y2 = track.last_box_xyxy
                x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, frame_w, frame_h)
                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue

                # Gate: do not consider bed for best-crop/event logic until it persists enough frames.
                if track.total_hits < min_bed_persist_frames:
                    continue

                area = max(0, (x2 - x1) * (y2 - y1))
                centeredness = _compute_centeredness((x1, y1, x2, y2), frame_w, frame_h)
                score = _calc_candidate_score_center_only(centeredness=centeredness, conf=det.conf)
                candidate = CropCandidate(
                    frame_idx=frame_idx,
                    xyxy=(x1, y1, x2, y2),
                    area=area,
                    conf=det.conf,
                    centeredness=centeredness,
                    score=score,
                    crop_bgr=crop.copy(),
                )
                best_updated = tracker.add_candidate(track_id, candidate)

                near_max = area >= int(0.95 * max(1, track.max_area_seen))
                if area >= min_best_area and near_max:
                    track.stable_count += 1
                elif area >= min_best_area and track.stable_count > 0:
                    track.stable_count += 1
                else:
                    track.stable_count = 0

                if best_updated:
                    debug_frame = frame.copy()
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    if track.smooth_truck_box_xyxy_f is not None:
                        tx1, ty1, tx2, ty2 = (
                            int(round(track.smooth_truck_box_xyxy_f[0])),
                            int(round(track.smooth_truck_box_xyxy_f[1])),
                            int(round(track.smooth_truck_box_xyxy_f[2])),
                            int(round(track.smooth_truck_box_xyxy_f[3])),
                        )
                        tx1, ty1, tx2, ty2 = clamp_xyxy(tx1, ty1, tx2, ty2, frame_w, frame_h)
                        cv2.rectangle(debug_frame, (tx1, ty1), (tx2, ty2), (255, 100, 0), 2)
                    cv2.putText(
                        debug_frame,
                        f"track={track_id} score={score:.1f}",
                        (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    best_image_path = events_images_dir / f"event_{track_id:05d}_best.jpg"
                    cv2.imwrite(str(best_image_path), debug_frame)
                    track.best_image_path = str(best_image_path)

            if event_infer_mode == "early":
                for track in list(tracker.active_tracks.values()):
                    if track.total_hits < min_bed_persist_frames:
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

            ready_ids = tracker.finalize_ready_ids(frame_idx=frame_idx)
            for track_id in ready_ids:
                track = tracker.pop_track(track_id)
                if track is None:
                    continue
                if track.total_hits < effective_min_hits:
                    logger.info(
                        "Dropping short/noisy track %s (hits=%s < effective_min_hits=%s)",
                        track.track_id,
                        track.total_hits,
                        effective_min_hits,
                    )
                    continue
                event = _finalize_track_event(
                    track=track,
                    models=models,
                    event_infer_mode=event_infer_mode,
                    top2=bool(top2),
                    seg_conf_threshold=seg_conf,
                    events_jsonl_path=events_jsonl,
                    events_images_dir=events_images_dir,
                    final_reason="missed_M_frames",
                    logger=logger,
                )
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
                    (0, 255, 255),
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
                track = tracker.pop_track(track_id)
                if track is None:
                    continue
                if track.total_hits < effective_min_hits:
                    logger.info(
                        "Dropping short/noisy track %s at EOF (hits=%s < effective_min_hits=%s)",
                        track.track_id,
                        track.total_hits,
                        effective_min_hits,
                    )
                    continue
                event = _finalize_track_event(
                    track=track,
                    models=models,
                    event_infer_mode=event_infer_mode,
                    top2=bool(top2),
                    seg_conf_threshold=seg_conf,
                    events_jsonl_path=events_jsonl,
                    events_images_dir=events_images_dir,
                    final_reason="end_of_video",
                    logger=logger,
                )
                events_finalized += 1
                heavy_calls = int(event.get("heavy_inference_calls", 0))
                heavy_inference_calls_total += heavy_calls
                phase_calls_by_event.append(heavy_calls)

    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

    avg_heavy = (sum(phase_calls_by_event) / len(phase_calls_by_event)) if phase_calls_by_event else 0.0
    logger.info(
        "stream_event complete | detection_frames=%s total_tracks_created=%s events_finalized=%s "
        "heavy_calls_total=%s avg_heavy=%.3f",
        detection_frames,
        tracker.total_tracks_created,
        events_finalized,
        heavy_inference_calls_total,
        avg_heavy,
    )
    return True
