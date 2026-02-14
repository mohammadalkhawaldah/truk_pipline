from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    # Anchor box for identity matching (truck box).
    xyxy: tuple[int, int, int, int]
    conf: float
    cls_id: int
    cls_name: str
    source: str = "truck_direct"
    # Optional child bed for crop/inference on same frame.
    bed_box_xyxy: tuple[int, int, int, int] | None = None
    bed_conf: float = 0.0


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
    truck_box_xyxy: tuple[int, int, int, int]
    raw_truck_box_xyxy: tuple[int, int, int, int]
    bed_box_xyxy: tuple[int, int, int, int] | None
    bed_conf: float
    last_conf: float
    missed_count: int = 0
    matched_in_update: bool = False
    total_hits: int = 1
    bed_hits: int = 0
    stable_count: int = 0
    max_area_seen: int = 0
    last_area: int = 0
    smooth_truck_box_xyxy_f: tuple[float, float, float, float] | None = None
    best_candidate: CropCandidate | None = None
    top_candidates: list[CropCandidate] = field(default_factory=list)
    vote_candidates: list[CropCandidate] = field(default_factory=list)
    last_vote_sample_frame: int | None = None
    inference_runs: list[dict[str, Any]] = field(default_factory=list)
    phase_results: dict[str, Any] | None = None
    best_image_path: str | None = None

    # Backward-compatible aliases for older stream code.
    @property
    def last_box_xyxy(self) -> tuple[int, int, int, int]:
        return self.truck_box_xyxy

    @property
    def smooth_box_xyxy_f(self) -> tuple[float, float, float, float] | None:
        return self.smooth_truck_box_xyxy_f


@dataclass
class LostTrackSnapshot:
    track_id: int
    last_truck_box: tuple[int, int, int, int]
    last_seen_frame: int
    best_candidate: CropCandidate | None
    vote_candidates: list[CropCandidate]
    last_vote_sample_frame: int | None
    phase_results: dict[str, Any] | None


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

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def center_xyxy(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay = center_xyxy(a)
    bx, by = center_xyxy(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def clamp_box_xyxy(
    box: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), max(0, frame_w - 1)))
    y1 = max(0, min(int(round(y1)), max(0, frame_h - 1)))
    x2 = max(0, min(int(round(x2)), frame_w))
    y2 = max(0, min(int(round(y2)), frame_h))
    if x2 <= x1:
        x2 = min(frame_w, x1 + 1)
    if y2 <= y1:
        y2 = min(frame_h, y1 + 1)
    return (x1, y1, x2, y2)


def expand_box_xyxy(
    box: tuple[int, int, int, int],
    x_scale: float,
    y_scale: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1.0, float(x2 - x1)) * max(0.01, float(x_scale))
    h = max(1.0, float(y2 - y1)) * max(0.01, float(y_scale))
    nx1 = cx - (w / 2.0)
    ny1 = cy - (h / 2.0)
    nx2 = cx + (w / 2.0)
    ny2 = cy + (h / 2.0)
    return clamp_box_xyxy((nx1, ny1, nx2, ny2), frame_w, frame_h)


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
    return (
        int(round(box_f[0])),
        int(round(box_f[1])),
        int(round(box_f[2])),
        int(round(box_f[3])),
    )


class IoUTracker:
    def __init__(
        self,
        iou_threshold: float = 0.25,
        missed_M: int = 15,
        top2: bool = False,
        smooth_alpha: float = 0.20,
        smooth_deadband_px: float = 4.0,
        smooth_max_step_px: float = 12.0,
        merge_window_frames: int = 10,
        merge_iou_threshold: float = 0.20,
        merge_center_dist_ratio: float = 0.15,
        edge_guard: bool = True,
        edge_margin: int = 80,
        recently_lost_maxlen: int = 256,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.missed_M = int(missed_M)
        self.keep_top_k = 2 if bool(top2) else 1
        self.smooth_alpha = float(max(0.01, min(1.0, smooth_alpha)))
        self.smooth_deadband_px = float(max(0.0, smooth_deadband_px))
        self.smooth_max_step_px = float(max(0.0, smooth_max_step_px))
        self.merge_window_frames = int(max(1, merge_window_frames))
        self.merge_iou_threshold = float(max(0.0, min(1.0, merge_iou_threshold)))
        self.merge_center_dist_ratio = float(max(0.0, merge_center_dist_ratio))
        self.edge_guard = bool(edge_guard)
        self.edge_margin = int(max(0, edge_margin))

        self.active_tracks: dict[int, TrackState] = {}
        self.recently_lost: deque[LostTrackSnapshot] = deque(maxlen=max(8, int(recently_lost_maxlen)))
        self.next_track_id = 1
        self.total_tracks_created = 0
        self.total_merges = 0

    def _create_track(self, det: Detection, frame_idx: int, track_id: int | None = None) -> TrackState:
        if track_id is None:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.total_tracks_created += 1
        else:
            self.next_track_id = max(self.next_track_id, track_id + 1)

        tx1, ty1, tx2, ty2 = det.xyxy
        t_area = max(0, (tx2 - tx1) * (ty2 - ty1))
        track = TrackState(
            track_id=track_id,
            start_frame=frame_idx,
            last_seen_frame=frame_idx,
            truck_box_xyxy=det.xyxy,
            raw_truck_box_xyxy=det.xyxy,
            bed_box_xyxy=det.bed_box_xyxy,
            bed_conf=float(det.bed_conf),
            last_conf=float(det.conf),
            max_area_seen=t_area,
            last_area=t_area,
            smooth_truck_box_xyxy_f=(float(tx1), float(ty1), float(tx2), float(ty2)),
            bed_hits=1 if det.bed_box_xyxy is not None else 0,
        )
        self.active_tracks[track_id] = track
        return track

    def _remove_lost_snapshot(self, track_id: int) -> None:
        if not self.recently_lost:
            return
        self.recently_lost = deque(
            [x for x in self.recently_lost if x.track_id != track_id],
            maxlen=self.recently_lost.maxlen,
        )

    def _upsert_lost_snapshot(self, track: TrackState) -> None:
        self._remove_lost_snapshot(track.track_id)
        self.recently_lost.append(
            LostTrackSnapshot(
                track_id=track.track_id,
                last_truck_box=track.raw_truck_box_xyxy,
                last_seen_frame=track.last_seen_frame,
                best_candidate=track.best_candidate,
                vote_candidates=list(track.vote_candidates),
                last_vote_sample_frame=track.last_vote_sample_frame,
                phase_results=track.phase_results,
            )
        )

    def _prune_lost(self, frame_idx: int) -> None:
        if not self.recently_lost:
            return
        kept = [
            x for x in self.recently_lost
            if (frame_idx - x.last_seen_frame) <= self.merge_window_frames
        ]
        self.recently_lost = deque(kept, maxlen=self.recently_lost.maxlen)

    def _near_edge(self, box: tuple[int, int, int, int], frame_w: int, frame_h: int) -> bool:
        cx, cy = center_xyxy(box)
        return bool(
            cx <= self.edge_margin
            or cy <= self.edge_margin
            or cx >= (frame_w - self.edge_margin)
            or cy >= (frame_h - self.edge_margin)
        )

    def _try_merge_with_recently_lost(
        self,
        det: Detection,
        frame_idx: int,
        frame_shape: tuple[int, int],
    ) -> tuple[LostTrackSnapshot | None, float, float]:
        self._prune_lost(frame_idx)
        frame_h, frame_w = frame_shape
        frame_diag = max(1.0, float((frame_w ** 2 + frame_h ** 2) ** 0.5))
        max_center_dist = self.merge_center_dist_ratio * frame_diag

        best_item = None
        best_iou = -1.0
        best_dist = float("inf")
        for item in self.recently_lost:
            gap = frame_idx - item.last_seen_frame
            if gap > self.merge_window_frames:
                continue
            iou = iou_xyxy(det.xyxy, item.last_truck_box)
            dist = center_distance(det.xyxy, item.last_truck_box)
            if iou >= self.merge_iou_threshold or dist <= max_center_dist:
                if (iou, -dist) > (best_iou, -best_dist):
                    best_item = item
                    best_iou = iou
                    best_dist = dist
        return best_item, best_iou, best_dist

    def _match_detections_to_tracks(
        self,
        detections: list[Detection],
        frame_idx: int,
    ) -> list[tuple[int, Detection, bool]]:
        for track in self.active_tracks.values():
            track.matched_in_update = False

        matches: list[tuple[int, Detection, bool]] = []
        assigned_tracks: set[int] = set()

        for det in sorted(detections, key=lambda d: d.conf, reverse=True):
            best_track_id = None
            best_iou = -1.0
            for track_id, track in self.active_tracks.items():
                if track_id in assigned_tracks:
                    continue
                iou = iou_xyxy(det.xyxy, track.raw_truck_box_xyxy)
                if iou < self.iou_threshold:
                    continue
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is None:
                continue

            track = self.active_tracks[best_track_id]
            track.last_seen_frame = frame_idx
            track.raw_truck_box_xyxy = det.xyxy
            if track.smooth_truck_box_xyxy_f is None:
                track.smooth_truck_box_xyxy_f = (
                    float(det.xyxy[0]),
                    float(det.xyxy[1]),
                    float(det.xyxy[2]),
                    float(det.xyxy[3]),
                )
            else:
                track.smooth_truck_box_xyxy_f = _smooth_box(
                    prev=track.smooth_truck_box_xyxy_f,
                    new_box=det.xyxy,
                    alpha=self.smooth_alpha,
                    deadband_px=self.smooth_deadband_px,
                    max_step_px=self.smooth_max_step_px,
                )
            track.truck_box_xyxy = _round_box(track.smooth_truck_box_xyxy_f)
            track.bed_box_xyxy = det.bed_box_xyxy
            track.bed_conf = float(det.bed_conf)
            track.last_conf = float(det.conf)
            track.matched_in_update = True
            track.total_hits += 1
            if det.bed_box_xyxy is not None:
                track.bed_hits += 1
            tx1, ty1, tx2, ty2 = track.truck_box_xyxy
            track.last_area = max(0, (tx2 - tx1) * (ty2 - ty1))
            track.max_area_seen = max(track.max_area_seen, track.last_area)
            track.missed_count = 0
            assigned_tracks.add(best_track_id)
            self._remove_lost_snapshot(best_track_id)
            matches.append((best_track_id, det, False))
        return matches

    def update(
        self,
        detections: list[Detection],
        frame_idx: int,
        frame_shape: tuple[int, int],
    ) -> tuple[dict[int, TrackState], list[TrackState], list[str]]:
        frame_h, frame_w = frame_shape
        merge_logs: list[str] = []
        matches = self._match_detections_to_tracks(detections=detections, frame_idx=frame_idx)
        matched_boxes = {det.xyxy for _, det, _ in matches}

        for det in sorted(detections, key=lambda d: d.conf, reverse=True):
            if det.xyxy in matched_boxes:
                continue
            lost_item, m_iou, m_dist = self._try_merge_with_recently_lost(
                det=det,
                frame_idx=frame_idx,
                frame_shape=frame_shape,
            )
            if lost_item is not None and lost_item.track_id not in self.active_tracks:
                track = self._create_track(det=det, frame_idx=frame_idx, track_id=lost_item.track_id)
                track.best_candidate = lost_item.best_candidate
                track.vote_candidates = list(lost_item.vote_candidates)
                track.last_vote_sample_frame = lost_item.last_vote_sample_frame
                track.phase_results = lost_item.phase_results
                track.matched_in_update = True
                self._remove_lost_snapshot(lost_item.track_id)
                self.total_merges += 1
                gap = frame_idx - lost_item.last_seen_frame
                merge_logs.append(
                    f"MERGE: new temp track merged into ID={lost_item.track_id} "
                    f"(gap={gap}, iou={m_iou:.3f}, center_dist={m_dist:.1f})"
                )
                matches.append((track.track_id, det, True))
                continue

            new_track = self._create_track(det=det, frame_idx=frame_idx, track_id=None)
            new_track.matched_in_update = True
            matches.append((new_track.track_id, det, True))

        for track in self.active_tracks.values():
            if track.matched_in_update:
                continue
            track.missed_count = max(0, frame_idx - track.last_seen_frame)
            if track.missed_count > 0:
                self._upsert_lost_snapshot(track)

        finalized: list[TrackState] = []
        remove_ids: list[int] = []
        for track_id, track in self.active_tracks.items():
            if track.missed_count < self.missed_M:
                continue
            if self.edge_guard and (not self._near_edge(track.truck_box_xyxy, frame_w, frame_h)):
                continue
            finalized.append(track)
            remove_ids.append(track_id)

        for track_id in remove_ids:
            self.active_tracks.pop(track_id, None)
            self._remove_lost_snapshot(track_id)

        self._prune_lost(frame_idx)
        return self.active_tracks, finalized, merge_logs

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

    def add_vote_candidate(
        self,
        track_id: int,
        candidate: CropCandidate,
        sample_every_frames: int = 5,
        max_samples: int = 80,
    ) -> bool:
        track = self.active_tracks.get(track_id)
        if track is None:
            return False

        stride = max(1, int(sample_every_frames))
        if track.last_vote_sample_frame is not None:
            if (candidate.frame_idx - track.last_vote_sample_frame) < stride:
                return False

        track.vote_candidates.append(candidate)
        track.last_vote_sample_frame = int(candidate.frame_idx)
        keep = max(1, int(max_samples))
        if len(track.vote_candidates) > keep:
            track.vote_candidates = track.vote_candidates[-keep:]
        return True

    def candidate_already_inferred(self, track: TrackState, frame_idx: int) -> bool:
        for run in track.inference_runs:
            if int(run.get("candidate_frame", -1)) == int(frame_idx):
                return True
        return False
