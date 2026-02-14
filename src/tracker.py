from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    xyxy: tuple[int, int, int, int]
    conf: float
    cls_id: int
    cls_name: str
    source: str = "direct"
    truck_box_xyxy: tuple[int, int, int, int] | None = None


@dataclass
class CropCandidate:
    frame_idx: int
    xyxy: tuple[int, int, int, int]
    area: int
    conf: float
    centeredness: float
    score: float
    crop_bgr: Any


@dataclass
class TrackState:
    track_id: int
    start_frame: int
    last_seen_frame: int
    last_box_xyxy: tuple[int, int, int, int]
    raw_box_xyxy: tuple[int, int, int, int]
    last_conf: float
    missed_count: int = 0
    matched_in_update: bool = False
    total_hits: int = 1
    stable_count: int = 0
    max_area_seen: int = 0
    last_area: int = 0
    smooth_box_xyxy_f: tuple[float, float, float, float] | None = None
    smooth_truck_box_xyxy_f: tuple[float, float, float, float] | None = None
    best_candidate: CropCandidate | None = None
    top_candidates: list[CropCandidate] = field(default_factory=list)
    inference_runs: list[dict[str, Any]] = field(default_factory=list)
    phase_results: dict[str, Any] | None = None
    best_image_path: str | None = None


def iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _center_xyxy(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _diag_xyxy(box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    return (w * w + h * h) ** 0.5


def _center_distance_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay = _center_xyxy(a)
    bx, by = _center_xyxy(b)
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return float(dist / max(1.0, _diag_xyxy(b)))


def _smooth_box(
    prev: tuple[float, float, float, float],
    new_box: tuple[int, int, int, int],
    alpha: float,
    deadband_px: float,
    max_step_px: float,
) -> tuple[float, float, float, float]:
    target = (float(new_box[0]), float(new_box[1]), float(new_box[2]), float(new_box[3]))
    smoothed = []
    for p, t in zip(prev, target):
        val = (1.0 - alpha) * p + alpha * t
        if max_step_px > 0.0:
            delta = val - p
            if delta > max_step_px:
                val = p + max_step_px
            elif delta < -max_step_px:
                val = p - max_step_px
        if abs(val - p) < deadband_px:
            val = p
        smoothed.append(val)
    return (smoothed[0], smoothed[1], smoothed[2], smoothed[3])


def _round_box(box_f: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return (int(round(box_f[0])), int(round(box_f[1])), int(round(box_f[2])), int(round(box_f[3])))


class IoUTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        missed_M: int = 10,
        top2: bool = False,
        smooth_alpha: float = 0.35,
        smooth_deadband_px: float = 2.0,
        smooth_max_step_px: float = 0.0,
        center_dist_ratio_threshold: float = 0.85,
        reacquire_iou_threshold: float = 0.05,
        reacquire_center_dist_ratio_threshold: float = 2.20,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.missed_M = int(missed_M)
        self.keep_top_k = 2 if bool(top2) else 1
        self.smooth_alpha = float(max(0.01, min(1.0, smooth_alpha)))
        self.smooth_deadband_px = float(max(0.0, smooth_deadband_px))
        self.smooth_max_step_px = float(max(0.0, smooth_max_step_px))
        self.center_dist_ratio_threshold = float(max(0.01, center_dist_ratio_threshold))
        self.reacquire_iou_threshold = float(max(0.0, min(1.0, reacquire_iou_threshold)))
        self.reacquire_center_dist_ratio_threshold = float(max(0.05, reacquire_center_dist_ratio_threshold))
        self.active_tracks: dict[int, TrackState] = {}
        self.next_track_id = 1
        self.total_tracks_created = 0

    def _create_track(self, det: Detection, frame_idx: int) -> TrackState:
        track_id = self.next_track_id
        self.next_track_id += 1
        self.total_tracks_created += 1
        x1, y1, x2, y2 = det.xyxy
        area = max(0, (x2 - x1) * (y2 - y1))
        track = TrackState(
            track_id=track_id,
            start_frame=frame_idx,
            last_seen_frame=frame_idx,
            last_box_xyxy=det.xyxy,
            raw_box_xyxy=det.xyxy,
            last_conf=det.conf,
            max_area_seen=area,
            last_area=area,
            smooth_box_xyxy_f=(float(det.xyxy[0]), float(det.xyxy[1]), float(det.xyxy[2]), float(det.xyxy[3])),
            smooth_truck_box_xyxy_f=(
                (float(det.truck_box_xyxy[0]), float(det.truck_box_xyxy[1]), float(det.truck_box_xyxy[2]), float(det.truck_box_xyxy[3]))
                if det.truck_box_xyxy is not None
                else None
            ),
        )
        self.active_tracks[track_id] = track
        return track

    def update(self, detections: list[Detection], frame_idx: int) -> list[tuple[int, Detection, bool]]:
        for track in self.active_tracks.values():
            track.matched_in_update = False

        matched_results: list[tuple[int, Detection, bool]] = []
        assigned_tracks: set[int] = set()

        for det in sorted(detections, key=lambda d: d.conf, reverse=True):
            best_track_id = None
            best_iou = 0.0
            best_center_ratio = float("inf")
            best_is_iou_match = False
            for track_id, track in self.active_tracks.items():
                if track_id in assigned_tracks:
                    continue
                reference_box = track.raw_box_xyxy
                current_iou = iou_xyxy(det.xyxy, reference_box)
                center_ratio = _center_distance_ratio(det.xyxy, reference_box)
                is_iou_match = current_iou >= self.iou_threshold
                is_center_match = center_ratio <= self.center_dist_ratio_threshold

                # Reacquire fallback: if the track is still active (missed < M), allow much looser matching
                # using truck box relation, preventing fragmentation into new IDs on brief jitter/occlusion.
                is_reacquire_match = False
                if (
                    not (is_iou_match or is_center_match)
                    and (0 < track.missed_count < self.missed_M)
                ):
                    if det.truck_box_xyxy is not None:
                        if track.smooth_truck_box_xyxy_f is not None:
                            tx1, ty1, tx2, ty2 = _round_box(track.smooth_truck_box_xyxy_f)
                            track_truck_box = (tx1, ty1, tx2, ty2)
                        elif track.last_box_xyxy is not None:
                            track_truck_box = track.last_box_xyxy
                        else:
                            track_truck_box = None

                        if track_truck_box is not None:
                            t_iou = iou_xyxy(det.truck_box_xyxy, track_truck_box)
                            t_center_ratio = _center_distance_ratio(det.truck_box_xyxy, track_truck_box)
                            if (
                                t_iou >= self.reacquire_iou_threshold
                                or t_center_ratio <= self.reacquire_center_dist_ratio_threshold
                            ):
                                is_reacquire_match = True

                if not (is_iou_match or is_center_match or is_reacquire_match):
                    continue

                if best_track_id is None:
                    best_track_id = track_id
                    best_iou = current_iou
                    best_center_ratio = center_ratio
                    best_is_iou_match = is_iou_match
                    continue

                candidate_tuple = (1 if is_iou_match else 0, current_iou, -center_ratio)
                best_tuple = (1 if best_is_iou_match else 0, best_iou, -best_center_ratio)
                if candidate_tuple > best_tuple:
                    best_track_id = track_id
                    best_iou = current_iou
                    best_center_ratio = center_ratio
                    best_is_iou_match = is_iou_match

            if best_track_id is not None:
                track = self.active_tracks[best_track_id]
                track.last_seen_frame = frame_idx
                track.raw_box_xyxy = det.xyxy
                if track.smooth_box_xyxy_f is None:
                    track.smooth_box_xyxy_f = (
                        float(det.xyxy[0]),
                        float(det.xyxy[1]),
                        float(det.xyxy[2]),
                        float(det.xyxy[3]),
                    )
                else:
                    track.smooth_box_xyxy_f = _smooth_box(
                        prev=track.smooth_box_xyxy_f,
                        new_box=det.xyxy,
                        alpha=self.smooth_alpha,
                        deadband_px=self.smooth_deadband_px,
                        max_step_px=self.smooth_max_step_px,
                    )
                track.last_box_xyxy = _round_box(track.smooth_box_xyxy_f)
                if det.truck_box_xyxy is not None:
                    if track.smooth_truck_box_xyxy_f is None:
                        track.smooth_truck_box_xyxy_f = (
                            float(det.truck_box_xyxy[0]),
                            float(det.truck_box_xyxy[1]),
                            float(det.truck_box_xyxy[2]),
                            float(det.truck_box_xyxy[3]),
                        )
                    else:
                        track.smooth_truck_box_xyxy_f = _smooth_box(
                            prev=track.smooth_truck_box_xyxy_f,
                            new_box=det.truck_box_xyxy,
                            alpha=self.smooth_alpha,
                            deadband_px=self.smooth_deadband_px,
                            max_step_px=self.smooth_max_step_px,
                        )
                track.last_conf = det.conf
                track.matched_in_update = True
                track.total_hits += 1
                x1, y1, x2, y2 = track.last_box_xyxy
                track.last_area = max(0, (x2 - x1) * (y2 - y1))
                track.max_area_seen = max(track.max_area_seen, track.last_area)
                track.missed_count = 0
                assigned_tracks.add(best_track_id)
                matched_results.append((best_track_id, det, False))
            else:
                new_track = self._create_track(det, frame_idx)
                new_track.matched_in_update = True
                assigned_tracks.add(new_track.track_id)
                matched_results.append((new_track.track_id, det, True))

        for track in self.active_tracks.values():
            if not track.matched_in_update:
                track.missed_count = max(0, frame_idx - track.last_seen_frame)

        return matched_results

    def add_candidate(self, track_id: int, candidate: CropCandidate) -> bool:
        track = self.active_tracks.get(track_id)
        if track is None:
            return False

        old_best_score = track.best_candidate.score if track.best_candidate is not None else float("-inf")
        if track.best_candidate is None or candidate.score > track.best_candidate.score:
            track.best_candidate = candidate

        existing_frames = {c.frame_idx for c in track.top_candidates}
        if candidate.frame_idx in existing_frames:
            for i, c in enumerate(track.top_candidates):
                if c.frame_idx == candidate.frame_idx and candidate.score > c.score:
                    track.top_candidates[i] = candidate
            track.top_candidates = sorted(track.top_candidates, key=lambda c: c.score, reverse=True)
        else:
            track.top_candidates.append(candidate)
            track.top_candidates = sorted(track.top_candidates, key=lambda c: c.score, reverse=True)[: self.keep_top_k]

        new_best_score = track.best_candidate.score if track.best_candidate is not None else float("-inf")
        return new_best_score > old_best_score

    def candidate_already_inferred(self, track: TrackState, frame_idx: int) -> bool:
        for run in track.inference_runs:
            if int(run.get("candidate_frame", -1)) == int(frame_idx):
                return True
        return False

    def finalize_ready_ids(self, frame_idx: int) -> list[int]:
        ready: list[int] = []
        for track_id, track in self.active_tracks.items():
            if (frame_idx - track.last_seen_frame) >= self.missed_M:
                ready.append(track_id)
        return sorted(ready)

    def pop_track(self, track_id: int) -> TrackState | None:
        return self.active_tracks.pop(track_id, None)
