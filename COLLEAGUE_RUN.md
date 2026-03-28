# Truck Pipeline Quick Run

## 1) Clone with submodule
```powershell
git clone --recurse-submodules https://github.com/mohammadalkhawaldah/truk_pipline.git
cd truk_pipline
```

If you already cloned without submodules:

```powershell
git submodule update --init --recursive
```

## 2) Create the two local environments
```powershell
.\setup_integrated_env.ps1
```

## 3) Run the integrated pipeline on one video
```powershell
.\run_integrated_pipeline.ps1 -VideoPath "C:\path\to\video.mp4"
```

## 4) Run all mp4 videos in a folder
```powershell
Get-ChildItem "C:\path\to\videos" -File -Filter *.mp4 |
ForEach-Object {
    .\run_integrated_pipeline.ps1 -VideoPath $_.FullName
}
```

## 5) Output
- Console and `combined_logs/*.log` contain only merged final lines such as:
  `2026-03-28 11:49:23 | TRUCK_ID=1 | COVER_STATUS=uncovered_or_partial | MATERIAL=sand | VIOLATION=True | FILL_LEVEL=96.21%`
- `truck_size_2` preview is shown during the run.
- `truk_pipline` runs without preview in parallel.

## Notes
- The repository now contains two repo-local venvs after setup:
  `.venv_truk_pipline` and `.venv_truck_size_2`.
- The integrated runner does not require manual activation.
