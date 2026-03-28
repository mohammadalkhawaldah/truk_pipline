# Integrated Pipeline Quick Start

This guide is for a fresh GitHub clone of `truk_pipline` when you want one final merged event line per truck, for example:

```text
2026-03-28 12:22:25 | TRUCK_ID=1 | COVER_STATUS=uncovered_or_partial | MATERIAL=sand | VIOLATION=True | FILL_LEVEL=97.60%
```

The merged line is produced by running:
- `truk_pipline` for compliance classification
- `truck_size_2` for fill-level estimation

## 1) Clone the repo with submodules

```powershell
git clone --recurse-submodules https://github.com/mohammadalkhawaldah/truk_pipline.git
cd truk_pipline
```

If you already cloned without submodules:

```powershell
git submodule update --init --recursive
```

## 2) Set up the local environments

Run:

```powershell
.\setup_integrated_env.ps1
```

This creates two repo-local virtual environments:
- `.venv_truk_pipline`
- `.venv_truck_size_2`

Two environments are required because the two pipelines depend on different `ultralytics` and `torch` versions.

## 3) Run one video

```powershell
.\run_integrated_pipeline.ps1 -VideoPath "C:\path\to\video.mp4"
```

Behavior:
- `truck_size_2` preview window is shown
- `truk_pipline` runs in parallel without preview
- the console prints only merged final lines
- the same lines are saved under `combined_logs\`

## 4) Run all `.mp4` videos in a folder

```powershell
Get-ChildItem "C:\path\to\videos" -File -Filter *.mp4 |
ForEach-Object {
    .\run_integrated_pipeline.ps1 -VideoPath $_.FullName
}
```

If you are not currently inside the repo folder, use the absolute script path:

```powershell
Get-ChildItem "C:\path\to\videos" -File -Filter *.mp4 |
ForEach-Object {
    & "C:\path\to\truk_pipline\run_integrated_pipeline.ps1" -VideoPath $_.FullName
}
```

## 5) Output format

Expected console/log format:

```text
YYYY-MM-DD HH:MM:SS | TRUCK_ID=1 | COVER_STATUS=uncovered_or_partial | MATERIAL=sand | VIOLATION=True | FILL_LEVEL=97.60%
```

Meaning:
- `TRUCK_ID`: finalized event id from `truk_pipline`
- `COVER_STATUS`: final cover decision
- `MATERIAL`: dominant material label
- `VIOLATION`: final violation flag
- `FILL_LEVEL`: fill percentage from `truck_size_2`

## 6) Requirements

Successful local run assumes:
- Python is installed and available as `python`
- PowerShell script execution is allowed
- model weights are present in the repo/submodule as committed
- the machine can run the required inference workload

## 7) Troubleshooting

### `run_integrated_pipeline.ps1` not found

You are not in the repo root. Either:

```powershell
cd C:\path\to\truk_pipline
```

or call the script with an absolute path.

### Missing submodule

Run:

```powershell
git submodule update --init --recursive
```

### Missing local environments

Run:

```powershell
.\setup_integrated_env.ps1
```

### You want to update to the latest code

```powershell
git pull origin main
git submodule update --init --recursive --remote
```

## 8) Recommended commands for collaborators

Fresh setup:

```powershell
git clone --recurse-submodules https://github.com/mohammadalkhawaldah/truk_pipline.git
cd truk_pipline
.\setup_integrated_env.ps1
```

Single video:

```powershell
.\run_integrated_pipeline.ps1 -VideoPath "C:\path\to\video.mp4"
```

Folder run:

```powershell
Get-ChildItem "C:\path\to\videos" -File -Filter *.mp4 |
ForEach-Object {
    .\run_integrated_pipeline.ps1 -VideoPath $_.FullName
}
```
