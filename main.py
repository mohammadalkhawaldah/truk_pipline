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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Truck pipeline (Phase 1/2/3/4/5)")
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
