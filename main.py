from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import config
from src.phase1_detect import run_phase1
from src.phase2_classify import run_phase2
from src.phase3_classify import run_phase3
from src.phase4_classify import run_phase4
from src.phase5_segment import run_phase5
from src.stream_pipeline import run_stream_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Truck pipeline (Phase 1/2/3/4/5)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["phase", "stream_event"],
        default="phase",
        help="Execution mode: phase-by-phase image mode or event-based stream mode.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["phase1", "phase2", "phase3", "phase4", "phase5"],
        default="phase1",
        help="Pipeline phase to run.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(config.INPUT_IMAGES_DIR),
        help="Folder of input images for Phase 1.",
    )
    parser.add_argument(
        "--detect-model",
        type=str,
        default=str(config.DETECT_MODEL_PATH) if config.DETECT_MODEL_PATH else "",
        help="Path to detection .pt model. If omitted, auto-discovery/interactive selection is used.",
    )
    parser.add_argument(
        "--top1-only",
        action="store_true",
        default=config.TOP1_ONLY,
        help="Keep only highest-confidence bed crop per image.",
    )
    parser.add_argument(
        "--all-crops",
        action="store_true",
        help="Override top1-only and save all bed crops per image.",
    )
    parser.add_argument(
        "--bed-class-id",
        action="append",
        type=int,
        default=None,
        help="Explicit bed class id(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--truck-class-id",
        action="append",
        type=int,
        default=None,
        help="Explicit truck class id(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--bed-class-name",
        action="append",
        type=str,
        default=None,
        help="Bed class name(s) to match from model.names. Can be passed multiple times.",
    )
    parser.add_argument(
        "--detect-conf",
        type=float,
        default=None,
        help="Phase 1 detection confidence threshold override (e.g., 0.005).",
    )
    parser.add_argument(
        "--phase1-crops-dir",
        type=str,
        default=str(config.PHASE1_CROPS_DIR),
        help="Folder of Phase 1 bed crops for Phase 2.",
    )
    parser.add_argument(
        "--cls1-model",
        type=str,
        default=str(config.CLS1_MODEL_PATH) if config.CLS1_MODEL_PATH else "",
        help="Path to Phase 2 classification#1 .pt model.",
    )
    parser.add_argument(
        "--phase2-summary-csv",
        type=str,
        default=str(config.PHASE2_SUMMARY_CSV),
        help="Phase 2 summary CSV used as input gate for Phase 3.",
    )
    parser.add_argument(
        "--cls2-model",
        type=str,
        default=str(config.CLS2_MODEL_PATH) if config.CLS2_MODEL_PATH else "",
        help="Path to Phase 3 classification#2 .pt model.",
    )
    parser.add_argument(
        "--phase3-summary-csv",
        type=str,
        default=str(config.PHASE3_SUMMARY_CSV),
        help="Phase 3 summary CSV used as input gate for Phase 4.",
    )
    parser.add_argument(
        "--cls3-model",
        type=str,
        default=str(config.CLS3_MODEL_PATH) if config.CLS3_MODEL_PATH else "",
        help="Path to Phase 4 classification#3 .pt model.",
    )
    parser.add_argument(
        "--seg-model",
        type=str,
        default=str(config.SEG_MODEL_PATH) if config.SEG_MODEL_PATH else "",
        help="Path to Phase 5 segmentation .pt model.",
    )
    parser.add_argument(
        "--phase5-input-csv",
        type=str,
        default=str(config.PHASE2_SUMMARY_CSV),
        help="Phase 2 summary CSV used as input gate for Phase 5.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="",
        help="Input video path for --mode stream_event.",
    )
    parser.add_argument(
        "--covered-class-name",
        action="append",
        type=str,
        default=None,
        help="Override covered class name mapping for Phase 2. Can be passed multiple times.",
    )
    parser.add_argument(
        "--uncovered-class-name",
        action="append",
        type=str,
        default=None,
        help="Override uncovered/partial class name mapping for Phase 2. Can be passed multiple times.",
    )
    parser.add_argument(
        "--irregular-class-name",
        action="append",
        type=str,
        default=None,
        help="Override irregular class name mapping for Phase 3. Can be passed multiple times.",
    )
    parser.add_argument(
        "--regular-class-name",
        action="append",
        type=str,
        default=None,
        help="Override regular class name mapping for Phase 3. Can be passed multiple times.",
    )
    parser.add_argument(
        "--sandlike-class-name",
        action="append",
        type=str,
        default=None,
        help="Override sand-like class name mapping for Phase 4. Can be passed multiple times.",
    )
    parser.add_argument(
        "--ironbars-class-name",
        action="append",
        type=str,
        default=None,
        help="Override ironbars-like class name mapping for Phase 4. Can be passed multiple times.",
    )
    parser.add_argument(
        "--unknown-class-name",
        action="append",
        type=str,
        default=None,
        help="Override unknown class name mapping for Phase 4. Can be passed multiple times.",
    )
    parser.add_argument(
        "--seg-conf",
        type=float,
        default=None,
        help="Phase 5 segmentation confidence threshold override.",
    )
    parser.add_argument(
        "--missed_M",
        type=int,
        default=config.STREAM_MISSED_M,
        help="Finalize event if a track is not matched for M frames.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=config.STREAM_IOU_THRESHOLD,
        help="IoU threshold for matching detections to existing tracks.",
    )
    parser.add_argument(
        "--merge_window",
        type=int,
        default=config.STREAM_MERGE_WINDOW,
        help="Merge window in frames for reusing IDs after short dropouts.",
    )
    parser.add_argument(
        "--merge_iou",
        type=float,
        default=config.STREAM_MERGE_IOU,
        help="Merge IoU threshold for old->new ID reuse.",
    )
    parser.add_argument(
        "--merge_center_ratio",
        type=float,
        default=config.STREAM_MERGE_CENTER_RATIO,
        help="Merge center distance threshold ratio (of frame diagonal).",
    )
    parser.add_argument(
        "--edge_guard",
        type=int,
        choices=[0, 1],
        default=1 if config.STREAM_EDGE_GUARD else 0,
        help="Finalize only near image edge (1) or always (0).",
    )
    parser.add_argument(
        "--edge_margin",
        type=int,
        default=config.STREAM_EDGE_MARGIN,
        help="Edge margin in pixels for edge_guard finalization.",
    )
    parser.add_argument(
        "--event_infer_mode",
        type=str,
        choices=["finalize", "early"],
        default=config.STREAM_EVENT_INFER_MODE,
        help="When to run heavy phases in stream_event mode.",
    )
    parser.add_argument(
        "--top2",
        type=int,
        choices=[0, 1],
        default=1 if config.STREAM_TOP2 else 0,
        help="In stream_event mode: infer on top-2 best moments per track (1) or top-1 only (0).",
    )
    parser.add_argument(
        "--every_n",
        type=int,
        default=config.STREAM_EVERY_N,
        help="Run detection every Nth frame in stream_event mode.",
    )
    parser.add_argument(
        "--max-detect-fps",
        type=float,
        default=config.STREAM_MAX_DETECT_FPS,
        help="Optional cap for detector run rate. 0 disables cap.",
    )
    parser.add_argument(
        "--track-smooth-alpha",
        type=float,
        default=config.STREAM_TRACK_SMOOTH_ALPHA,
        help="EMA alpha for track box smoothing (lower = more stable).",
    )
    parser.add_argument(
        "--track-deadband-px",
        type=float,
        default=config.STREAM_TRACK_DEADBAND_PX,
        help="Pixel deadband for smoothed track boxes.",
    )
    parser.add_argument(
        "--track-max-step-px",
        type=float,
        default=config.STREAM_TRACK_MAX_STEP_PX,
        help="Max per-update movement of a smoothed track box edge in pixels (0 disables).",
    )
    parser.add_argument(
        "--show",
        type=int,
        choices=[0, 1],
        default=0,
        help="Show live preview window during stream_event detection/tracking.",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        help="Preview window resize scale for --show 1 (e.g., 0.7).",
    )
    parser.add_argument(
        "--preview-fullscreen",
        type=int,
        choices=[0, 1],
        default=0,
        help="Show preview window in fullscreen mode during stream_event.",
    )
    parser.add_argument(
        "--summary-only",
        type=int,
        choices=[0, 1],
        default=0,
        help="In stream_event mode: print only event summary lines and final stats to console.",
    )
    parser.add_argument(
        "--min-event-hits",
        type=int,
        default=config.STREAM_MIN_EVENT_HITS,
        help="Minimum matched detections required before a track is finalized into an event.",
    )
    parser.add_argument(
        "--min-bed-persist-frames",
        type=int,
        default=config.STREAM_MIN_BED_PERSIST_FRAMES,
        help="Bed must persist for at least this many matched frames before being considered.",
    )
    parser.add_argument(
        "--vote-enable",
        type=int,
        choices=[0, 1],
        default=1 if config.STREAM_VOTE_ENABLE else 0,
        help="Enable event-level majority voting using sampled bed crops across the track.",
    )
    parser.add_argument(
        "--vote-every",
        type=int,
        default=config.STREAM_VOTE_EVERY_N_FRAMES,
        help="Sample one bed crop every N processed frames for voting.",
    )
    parser.add_argument(
        "--vote-max-samples",
        type=int,
        default=config.STREAM_VOTE_MAX_SAMPLES,
        help="Maximum number of sampled crops kept per track for voting.",
    )
    parser.add_argument(
        "--non-interactive-model-select",
        action="store_true",
        help="Disable interactive prompt if multiple models are found; highest-scored candidate is used.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of images/crops for acceptance tests (uses all if fewer are available).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ok = False

    if args.mode == "stream_event":
        if not args.video_path:
            print("Error: --video-path is required when --mode stream_event")
            return 1
        detect_model = Path(args.detect_model) if args.detect_model else None
        cls1_model = Path(args.cls1_model) if args.cls1_model else None
        cls2_model = Path(args.cls2_model) if args.cls2_model else None
        cls3_model = Path(args.cls3_model) if args.cls3_model else None
        seg_model = Path(args.seg_model) if args.seg_model else None
        ok = run_stream_event(
            video_path=Path(args.video_path),
            detect_model_path=detect_model,
            cls1_model_path=cls1_model,
            cls2_model_path=cls2_model,
            cls3_model_path=cls3_model,
            seg_model_path=seg_model,
            missed_M=args.missed_M,
            iou_threshold=args.iou_threshold,
            merge_window_frames=args.merge_window,
            merge_iou_threshold=args.merge_iou,
            merge_center_dist_ratio=args.merge_center_ratio,
            edge_guard=bool(args.edge_guard),
            edge_margin=args.edge_margin,
            event_infer_mode=args.event_infer_mode,
            top2=bool(args.top2),
            every_n=args.every_n,
            max_detect_fps=args.max_detect_fps,
            detect_conf_threshold=args.detect_conf,
            seg_conf_threshold=args.seg_conf,
            bed_class_ids=args.bed_class_id,
            truck_class_ids=args.truck_class_id,
            show_preview=bool(args.show),
            preview_scale=args.preview_scale,
            preview_fullscreen=bool(args.preview_fullscreen),
            summary_only=bool(args.summary_only),
            min_event_hits=args.min_event_hits,
            min_bed_persist_frames=args.min_bed_persist_frames,
            vote_enable=bool(args.vote_enable),
            vote_every_n_frames=args.vote_every,
            vote_max_samples=args.vote_max_samples,
            track_smooth_alpha=args.track_smooth_alpha,
            track_deadband_px=args.track_deadband_px,
            track_max_step_px=args.track_max_step_px,
        )
    else:
        if args.phase == "phase1":
            detect_model = Path(args.detect_model) if args.detect_model else None
            top1_only = False if args.all_crops else bool(args.top1_only)
            ok = run_phase1(
                input_images_dir=Path(args.input_dir),
                detect_model_path=detect_model,
                top1_only=top1_only,
                bed_class_ids=args.bed_class_id,
                bed_class_names=args.bed_class_name,
                detect_conf_threshold=args.detect_conf,
                interactive_model_select=not args.non_interactive_model_select,
                sample_size=args.sample_size,
            )
        elif args.phase == "phase2":
            cls1_model = Path(args.cls1_model) if args.cls1_model else None
            ok = run_phase2(
                phase1_crops_dir=Path(args.phase1_crops_dir),
                cls1_model_path=cls1_model,
                interactive_model_select=not args.non_interactive_model_select,
                sample_size=args.sample_size,
                covered_class_names=args.covered_class_name,
                uncovered_class_names=args.uncovered_class_name,
            )
        elif args.phase == "phase3":
            cls2_model = Path(args.cls2_model) if args.cls2_model else None
            ok = run_phase3(
                phase2_summary_csv=Path(args.phase2_summary_csv),
                cls2_model_path=cls2_model,
                interactive_model_select=not args.non_interactive_model_select,
                sample_size=args.sample_size,
                irregular_class_names=args.irregular_class_name,
                regular_class_names=args.regular_class_name,
            )
        elif args.phase == "phase4":
            cls3_model = Path(args.cls3_model) if args.cls3_model else None
            ok = run_phase4(
                phase3_summary_csv=Path(args.phase3_summary_csv),
                cls3_model_path=cls3_model,
                interactive_model_select=not args.non_interactive_model_select,
                sample_size=args.sample_size,
                sandlike_class_names=args.sandlike_class_name,
                ironbars_class_names=args.ironbars_class_name,
                unknown_class_names=args.unknown_class_name,
            )
        elif args.phase == "phase5":
            seg_model = Path(args.seg_model) if args.seg_model else None
            ok = run_phase5(
                phase2_summary_csv=Path(args.phase5_input_csv),
                seg_model_path=seg_model,
                interactive_model_select=not args.non_interactive_model_select,
                sample_size=args.sample_size,
                seg_conf_threshold=args.seg_conf,
            )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
