# Truck Bed Compliance Pipeline

Event-based multi-phase computer vision pipeline for truck-bed monitoring.

The system supports:
- `phase` mode: run individual phases on images/crops.
- `stream_event` mode (default): run on video with tracking + event finalization.

Built with:
- Python
- Ultralytics YOLO (detection / classification / segmentation)
- OpenCV

## 1) What This Repo Does

Given image/video input, the pipeline performs:
1. Phase 1: detect truck + bed region.
2. Phase 2: classify bed as covered vs uncovered/partial.
3. Phase 3 (covered path): classify regular vs irregular.
4. Phase 4 (covered + irregular path): classify irregular type.
5. Phase 5 (uncovered/partial path): segment load material (e.g. sand/barrels/empty/...)

In `stream_event` mode, heavy phases are run per event (not per frame), optionally with voting across sampled crops.

## 2) Repository Layout

```text
truck_pipline/
  main.py
  requirements.txt
  README.md
  COLLEAGUE_RUN.md
  src/
    config.py
    utils.py
    phase1_detect.py
    phase2_classify.py
    phase3_classify.py
    phase4_classify.py
    phase5_segment.py
    tracker.py
    stream_pipeline.py
  weights/
    ... model files (.pt)
  outputs/
    ... generated summaries/images/jsonl
  logs/
    ... runtime logs
```

## 3) Default Models (Current)

Current auto-selected defaults from `src/config.py` scoring logic:

- Phase 1 detect:
  - `weights/weights_March_25/best_Truck_Box_Extraction_March_25.pt`
- Phase 2 classify:
  - `weights/weights_March_25/best_1st_cls_March_25.pt`
- Phase 3 classify:
  - `weights/classification#2/best.pt`
- Phase 4 classify:
  - `weights/classification#3_new1/best_5classes.pt`
- Phase 5 segment:
  - `weights/weights_March_25/best_yolo11_seg_march_26v2.pt`

You can override any model at runtime:
- `--detect-model`
- `--cls1-model`
- `--cls2-model`
- `--cls3-model`
- `--seg-model`

You can also override by environment variables:
- `DETECT_MODEL_PATH`, `CLS1_MODEL_PATH`, `CLS2_MODEL_PATH`, `CLS3_MODEL_PATH`, `SEG_MODEL_PATH`

## 4) Environment Setup

### Integrated Clone (Recommended)

This repository can now run the `truck_size_2` fill estimator and the `truk_pipline` compliance pipeline together.

Clone with submodules so both codebases are pulled together:

```powershell
git clone --recurse-submodules https://github.com/mohammadalkhawaldah/truk_pipline.git
cd truk_pipline
```

If you already cloned without submodules:

```powershell
git submodule update --init --recursive
```

### Windows (PowerShell)

```powershell
cd C:\Users\moham\OneDrive\Documents\truck_pipline
.\setup_integrated_env.ps1
```

### macOS/Linux (bash)

```bash
cd /path/to/truck_pipline
python3 -m venv .venv_truk_pipline
python3 -m venv .venv_truck_size_2
source .venv_truk_pipline/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
deactivate
source .venv_truck_size_2/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r truck_size_2/requirements.txt
python -m pip install ultralytics==8.4.30 torch==2.11.0 torchvision==0.26.0 numpy==2.2.6 opencv-python==4.13.0.92
```

The integrated setup uses two repo-local virtual environments:
- `.venv_truk_pipline` for this repo
- `.venv_truck_size_2` for the `truck_size_2` submodule

This is intentional because the two codebases require different model/runtime versions.

## 5) Main Run Modes

## 5.1 Stream Event Mode (Default)

`main.py` defaults to `--mode stream_event`.

Example (strict settings used in current tuning):

```powershell
python main.py --video-path "E:\path\to\video.mp4" --event_infer_mode finalize --vote-enable 1 --vote-every 5 --vote-max-samples 10 --missed_M 8 --iou_threshold 0.30 --merge_window 12 --merge_iou 0.30 --merge_center_ratio 0.08 --edge_guard 0 --edge_margin 80 --top2 0 --every_n 1 --max-detect-fps 0 --detect-conf 0.09 --min-event-hits 8 --min-bed-persist-frames 12 --track-confirm-hits 6 --event-dedup 1 --event-dedup-window 180 --event-dedup-iou 0.10 --event-dedup-center-ratio 0.15 --new-track-ignore-lower-ratio 0.30 --show 1 --preview-scale 0.5 --preview-fullscreen 0 --summary-only 1 --non-interactive-model-select
```

### Batch run all videos in a folder

```powershell
Get-ChildItem "E:\path\to\videos\*" -File -Include *.mp4,*.avi,*.mov,*.mkv |
ForEach-Object {
  python main.py --video-path "$($_.FullName)" --event_infer_mode finalize --vote-enable 1 --vote-every 5 --vote-max-samples 10 --missed_M 8 --iou_threshold 0.30 --merge_window 12 --merge_iou 0.30 --merge_center_ratio 0.08 --edge_guard 0 --edge_margin 80 --top2 0 --every_n 1 --max-detect-fps 0 --detect-conf 0.09 --min-event-hits 8 --min-bed-persist-frames 12 --track-confirm-hits 6 --event-dedup 1 --event-dedup-window 180 --event-dedup-iou 0.10 --event-dedup-center-ratio 0.15 --new-track-ignore-lower-ratio 0.30 --show 1 --preview-scale 0.5 --preview-fullscreen 0 --summary-only 1 --non-interactive-model-select
}
```

## 5.2 Phase Mode (Image Pipeline)

### Phase 1
```powershell
python main.py --mode phase --phase phase1 --input-dir "data/images" --non-interactive-model-select
```

### Phase 2
```powershell
python main.py --mode phase --phase phase2 --phase1-crops-dir "outputs/phase1_crops" --non-interactive-model-select
```

### Phase 3
```powershell
python main.py --mode phase --phase phase3 --phase2-summary-csv "outputs/phase2_summary.csv" --non-interactive-model-select
```

### Phase 4
```powershell
python main.py --mode phase --phase phase4 --phase3-summary-csv "outputs/phase3_summary.csv" --non-interactive-model-select
```

### Phase 5
```powershell
python main.py --mode phase --phase phase5 --phase5-input-csv "outputs/phase2_summary.csv" --non-interactive-model-select
```

## 6) Key Stream Parameters

- `--every_n`: detector cadence (1 = every frame).
- `--max-detect-fps`: cap detector rate (0 disables cap).
- `--missed_M`: frames before finalizing a disappeared track.
- `--track-confirm-hits`: hits needed before track is confirmed.
- `--min-event-hits`: minimum truck detections for event eligibility.
- `--min-bed-persist-frames`: minimum bed detections for event eligibility.
- `--vote-enable`: enable per-event voting.
- `--vote-every`: sample interval for voting crops.
- `--vote-max-samples`: cap voting sample count.
- `--event-dedup*`: post-finalization duplicate suppression.
- `--new-track-ignore-lower-ratio`: ignore creating new tracks in lower image band (e.g. `0.30` = bottom 30%).

## 7) How Crop Selection Works in Stream Mode

For each active truck track, bed crop candidates are created when bed is detected.

Best-frame selection policy is center-priority:
- score prefers bed center nearest to image center.
- confidence is used only as a tiny tie-breaker.

With voting enabled:
- multiple sampled crops can be inferred per track.
- winner is chosen by majority vote; ties broken by average pipeline confidence.

## 8) Outputs

Main outputs:
- `outputs/events.jsonl`
  - one JSON event per finalized truck event.
- `outputs/events_images/`
  - selected event crop and overlay images.
- Phase outputs:
  - `outputs/phase1_summary.csv`
  - `outputs/phase2_summary.csv`
  - `outputs/phase3_summary.csv`
  - `outputs/phase4_summary.csv`
  - `outputs/phase5_summary.csv`

Logs:
- `logs/stream_event.log`
- `combined_logs/*.log` from integrated runs

## 8.1 Integrated Run Output

The integrated runner combines:
- `truck_size_2` fill level output
- `truk_pipline` event output

It prints only merged final lines like:

```text
2026-03-28 11:49:23 | TRUCK_ID=1 | COVER_STATUS=uncovered_or_partial | MATERIAL=sand | VIOLATION=True | FILL_LEVEL=96.21%
```

Run one video:

```powershell
.\run_integrated_pipeline.ps1 -VideoPath "E:\path\to\video.mp4"
```

Run all `.mp4` videos in a folder:

```powershell
Get-ChildItem "E:\path\to\videos" -File -Filter *.mp4 |
ForEach-Object {
  .\run_integrated_pipeline.ps1 -VideoPath $_.FullName
}
```

Integrated runner behavior:
- shows preview only from `truck_size_2`
- runs `truk_pipline` in parallel without preview
- writes only merged final event lines to `combined_logs/`

## 9) Event Line Meaning (Console)

Example:

```text
2026-03-27 10:09:07 | TRUCK_ID=1 | COVER_STATUS=uncovered_or_partial | MATERIAL=sand | VIOLATION=True
```

- `TRUCK_ID`: track/event id.
- `COVER_STATUS`: final covered/uncovered/irregular status.
- `MATERIAL`: segmented material labels for uncovered/partial path.
- `VIOLATION`:
  - `True` for violation path.
  - `under_investigation` for non-violation or phase-4-classification-only cases per current business rule.

## 10) Common Troubleshooting

1. `can't open file ... main.py`
   - You are in wrong folder. `cd` to repo root first.

2. `argument --detect-conf: expected one argument`
   - You passed `--detect-conf` without value.

3. No files processed in folder loop
   - Use wildcard path with `Get-ChildItem "...\*"` when using `-Include`.

4. Segmentation model load error with custom layer
   - Ensure model is compatible with installed ultralytics/torch versions.

5. Duplicate events near frame edge
   - Increase strictness using:
     - `--track-confirm-hits`
     - `--min-event-hits`
     - `--min-bed-persist-frames`
     - `--new-track-ignore-lower-ratio`
     - `--event-dedup-*`

## 11) Useful Helper Commands

Print detection model classes:

```powershell
python -c "from ultralytics import YOLO; m=YOLO(r'weights\weights_March_25\best_Truck_Box_Extraction_March_25.pt'); print(m.names)"
```

Show current git status:

```powershell
git status --short
```

## 12) Notes for Collaborators

See `COLLEAGUE_RUN.md` for a compact quick-start.

