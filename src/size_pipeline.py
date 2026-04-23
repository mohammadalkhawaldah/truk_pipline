from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2

from src import config
from src.fill_estimation import estimate_fill_for_truck_crop
from src.tracker import CropCandidate


MIN_RELIABLE_FILL = 5.0
MAX_SELECTION_CANDIDATES = 7


@dataclass
class SizeCandidateResult:
    candidate: CropCandidate
    fill_estimation: dict[str, Any] | None

    @property
    def frame_idx(self) -> int:
        return int(self.candidate.frame_idx)

    @property
    def score(self) -> float:
        return float(self.candidate.score)


@dataclass
class SizeSelection:
    candidate: CropCandidate
    fill_estimation: dict[str, Any] | None
    reason: str


def blur_score(frame_bgr, box_xyxy: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box_xyxy
    crop = frame_bgr[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(min(float(variance) / 400.0, 1.0))


def edge_penalty(
    box_xyxy: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    margin_ratio: float = 0.03,
) -> float:
    x1, y1, x2, y2 = box_xyxy
    x_margin = float(frame_width) * float(margin_ratio)
    y_margin = float(frame_height) * float(margin_ratio)
    touches_edge = (
        x1 <= x_margin
        or y1 <= y_margin
        or x2 >= frame_width - x_margin
        or y2 >= frame_height - y_margin
    )
    return 1.0 if touches_edge else 0.0


def centeredness(box_xyxy: tuple[int, int, int, int], frame_width: int, frame_height: int) -> float:
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = (cx / max(1.0, float(frame_width))) - 0.5
    dy = (cy / max(1.0, float(frame_height))) - 0.5
    dist = (dx * dx + dy * dy) ** 0.5
    max_dist = (0.5 * 0.5 + 0.5 * 0.5) ** 0.5
    return float(1.0 - min(1.0, dist / max_dist))


def score_truck_candidate(frame_bgr, bbox_xyxy: tuple[int, int, int, int], confidence: float) -> float:
    frame_height, frame_width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = max(1, frame_width * frame_height)
    area_score = float(box_area) / float(frame_area)
    center_score = centeredness(bbox_xyxy, frame_width, frame_height)
    sharpness_score = blur_score(frame_bgr, bbox_xyxy)
    border_penalty = edge_penalty(bbox_xyxy, frame_width, frame_height)
    return float(confidence) + (2.0 * area_score) + center_score + sharpness_score - (1.5 * border_penalty)


def select_fill_candidates(candidates: list[CropCandidate]) -> list[CropCandidate]:
    ordered_by_score = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    score_candidates = ordered_by_score[:MAX_SELECTION_CANDIDATES]
    if not candidates:
        return score_candidates

    midpoint_candidate = candidates[len(candidates) // 2]
    midpoint_candidates = sorted(
        candidates,
        key=lambda candidate: abs(int(candidate.frame_idx) - int(midpoint_candidate.frame_idx)),
    )[:3]

    merged: dict[int, CropCandidate] = {}
    for candidate in score_candidates + midpoint_candidates:
        existing = merged.get(int(candidate.frame_idx))
        if existing is None or candidate.score > existing.score:
            merged[int(candidate.frame_idx)] = candidate
    return sorted(merged.values(), key=lambda candidate: candidate.frame_idx)


def evaluate_fill_candidates(candidates: list[CropCandidate], size_model, size_names: dict[int, str]) -> list[SizeCandidateResult]:
    results: list[SizeCandidateResult] = []
    for candidate in candidates:
        fill_estimation = None
        if candidate.truck_crop_bgr is not None:
            fill_estimation = estimate_fill_for_truck_crop(
                truck_crop_bgr=candidate.truck_crop_bgr,
                size_model=size_model,
                size_names=size_names,
                seg_conf_threshold=config.SIZE_SEG_CONF_THRESHOLD,
            )
        results.append(SizeCandidateResult(candidate=candidate, fill_estimation=fill_estimation))
    return results


def select_best_fill_result(
    all_candidates: list[CropCandidate],
    results: list[SizeCandidateResult],
) -> SizeSelection | None:
    if not all_candidates:
        return None

    valid_results: list[SizeCandidateResult] = []
    positive_results: list[SizeCandidateResult] = []
    for result in results:
        fill_estimation = result.fill_estimation
        if not isinstance(fill_estimation, dict):
            continue
        fill_percentage = fill_estimation.get("fill_percentage")
        status = str(fill_estimation.get("status", "")).strip().lower()
        if fill_percentage is not None and status == "ok":
            valid_results.append(result)
            if float(fill_percentage) >= MIN_RELIABLE_FILL:
                positive_results.append(result)

    midpoint_frame = int(all_candidates[len(all_candidates) // 2].frame_idx)
    if len(positive_results) >= 2:
        positive_values = sorted(float(result.fill_estimation["fill_percentage"]) for result in positive_results)
        median_fill = positive_values[len(positive_values) // 2]
        best = min(
            positive_results,
            key=lambda result: (
                abs(float(result.fill_estimation["fill_percentage"]) - median_fill),
                abs(int(result.frame_idx) - int(midpoint_frame)),
                -result.score,
            ),
        )
        return SizeSelection(candidate=best.candidate, fill_estimation=best.fill_estimation, reason="median_positive_fill")
    if positive_results:
        best = max(positive_results, key=lambda result: result.score)
        return SizeSelection(candidate=best.candidate, fill_estimation=best.fill_estimation, reason="highest_score_positive_fill")
    if valid_results:
        best = max(valid_results, key=lambda result: result.score)
        return SizeSelection(candidate=best.candidate, fill_estimation=best.fill_estimation, reason="highest_score_valid_fill")

    best_candidate = max(all_candidates, key=lambda candidate: candidate.score)
    matching_result = next((result for result in results if int(result.frame_idx) == int(best_candidate.frame_idx)), None)
    fill_estimation = matching_result.fill_estimation if matching_result is not None else None
    return SizeSelection(candidate=best_candidate, fill_estimation=fill_estimation, reason="highest_score_fallback")
