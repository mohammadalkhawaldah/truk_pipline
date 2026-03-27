# Truck Pipeline Quick Run

## 1) Pull latest code
```powershell
git pull origin main
```

## 2) Create and activate venv
```powershell
py -3.11 -m venv pipeline_venv
.\pipeline_venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3) Run on one video
```powershell
python main.py --video-path "C:\path\to\video.mp4" --detect-model "C:\Users\moham\OneDrive\Documents\truck_pipline\weights\truck_and_bed_2trucks\best.pt" --event_infer_mode finalize --vote-enable 1 --vote-every 5 --vote-max-samples 10 --missed_M 8 --iou_threshold 0.40 --merge_window 12 --merge_iou 0.30 --merge_center_ratio 0.08 --edge_guard 0 --edge_margin 80 --top2 0 --every_n 1 --max-detect-fps 0 --detect-conf 0.06 --min-event-hits 6 --min-bed-persist-frames 12 --track-confirm-hits 3 --event-dedup 1 --event-dedup-window 80 --event-dedup-iou 0.35 --event-dedup-center-ratio 0.08 --show 1 --preview-scale 0.5 --preview-fullscreen 0 --summary-only 1 --non-interactive-model-select
```

## 4) Outputs
- Event log: `outputs/events.jsonl`
- Event images: `outputs/events_images/`
- Phase summaries: `outputs/phase*_summary.csv`

## Notes
- Weights are included in the repo under `weights/`.
- If preview opens, press `q` to stop.
