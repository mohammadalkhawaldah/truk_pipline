param(
    [Parameter(Mandatory = $true)]
    [string]$VideoPath,

    [int]$TruckSizeFps = 2,

    [string]$LogDir = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$truckSizeRepo = Join-Path $repoRoot "truck_size_2"
$pipelinePython = Join-Path $repoRoot ".venv_truk_pipline\Scripts\python.exe"
$truckSizePython = Join-Path $repoRoot ".venv_truck_size_2\Scripts\python.exe"

if (-not (Test-Path $VideoPath)) {
    throw "Video not found: $VideoPath"
}
if (-not (Test-Path $truckSizeRepo)) {
    throw "Missing submodule at $truckSizeRepo. Clone with --recurse-submodules or run: git submodule update --init --recursive"
}
if (-not (Test-Path $pipelinePython)) {
    throw "Missing truk_pipline venv at $pipelinePython. Run .\setup_integrated_env.ps1 first."
}
if (-not (Test-Path $truckSizePython)) {
    throw "Missing truck_size_2 venv at $truckSizePython. Run .\setup_integrated_env.ps1 first."
}

if ([string]::IsNullOrWhiteSpace($LogDir)) {
    $LogDir = Join-Path $repoRoot "combined_logs"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$videoName = [System.IO.Path]::GetFileNameWithoutExtension($VideoPath)
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $LogDir ("combined_{0}_{1}.log" -f $videoName, $timestamp)
$repo1OutputDir = Join-Path $truckSizeRepo ("auto_outputs\{0}" -f $videoName)
$repo1SummaryPath = Join-Path $truckSizeRepo ("auto_outputs\{0}\truck_fill_summary.csv" -f $videoName)
$repo2EventsPath = Join-Path $repoRoot "outputs\events.jsonl"

function Get-CoverStatus {
    param($Event)

    if ($null -eq $Event.coverage) {
        return "unknown"
    }

    $predLabel = [string]$Event.coverage.pred_label
    if ($predLabel -match 'uncovered|partial') {
        return $predLabel
    }
    if ($predLabel -eq 'covered') {
        if ($null -ne $Event.shape -and [string]$Event.shape.pred_label -eq 'regular') {
            return "covered_regular"
        }
        if ($null -ne $Event.shape -and [string]$Event.shape.pred_label -eq 'irregular') {
            return "covered_irregular"
        }
        return "covered"
    }
    return $predLabel
}

function Get-Material {
    param($Event)

    if ($null -eq $Event.segmentation) {
        return "unknown"
    }
    $label = [string]$Event.segmentation.dominant_label
    if ([string]::IsNullOrWhiteSpace($label)) {
        return "unknown"
    }
    return $label
}

function Clear-DirectoryFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DirectoryPath
    )

    if (-not (Test-Path $DirectoryPath)) {
        return
    }

    Get-ChildItem -LiteralPath $DirectoryPath -File | Remove-Item -Force
}

function Remove-DirectoryFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DirectoryPath
    )

    if (-not (Test-Path $DirectoryPath)) {
        return
    }

    Get-ChildItem -LiteralPath $DirectoryPath -File | Remove-Item -Force
}

if (Test-Path $repo1OutputDir) {
    Clear-DirectoryFiles -DirectoryPath $repo1OutputDir
}
else {
    New-Item -ItemType Directory -Force -Path $repo1OutputDir | Out-Null
}

$repo2StartCount = 0
if (Test-Path $repo2EventsPath) {
    $repo2StartCount = (Get-Content $repo2EventsPath | Measure-Object -Line).Lines
}

$repo2StdOut = Join-Path $env:TEMP ("truk_pipline_stdout_{0}.log" -f $timestamp)
$repo2StdErr = Join-Path $env:TEMP ("truk_pipline_stderr_{0}.log" -f $timestamp)
$escapedVideoPath = $VideoPath.Replace('"', '\"')
$repo2Args = @(
    '.\main.py',
    '--video-path', "`"$escapedVideoPath`"",
    '--event_infer_mode', 'finalize',
    '--vote-enable', '1',
    '--vote-every', '5',
    '--vote-max-samples', '5',
    '--missed_M', '16',
    '--iou_threshold', '0.25',
    '--merge_window', '24',
    '--merge_iou', '0.20',
    '--merge_center_ratio', '0.12',
    '--edge_guard', '1',
    '--edge_margin', '80',
    '--top2', '0',
    '--every_n', '1',
    '--max-detect-fps', '0',
    '--detect-conf', '0.09',
    '--min-event-hits', '8',
    '--min-bed-persist-frames', '12',
    '--track-confirm-hits', '6',
    '--event-dedup', '1',
    '--event-dedup-window', '180',
    '--event-dedup-iou', '0.10',
    '--event-dedup-center-ratio', '0.15',
    '--new-track-ignore-lower-ratio', '0.30',
    '--show', '0',
    '--preview-scale', '0.5',
    '--preview-fullscreen', '0',
    '--summary-only', '1',
    '--non-interactive-model-select'
)

$repo2Process = Start-Process `
    -FilePath $pipelinePython `
    -ArgumentList ($repo2Args -join ' ') `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $repo2StdOut `
    -RedirectStandardError $repo2StdErr `
    -PassThru

Push-Location $truckSizeRepo
try {
    & $truckSizePython ".\auto_select_truck_frames.py" $VideoPath "--fps" "$TruckSizeFps" "--write-summary-csv" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "truck_size_2 failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$repo2Process.WaitForExit()
$repo2Process.Refresh()
$repo2ExitCode = $repo2Process.ExitCode
if ($null -eq $repo2ExitCode -or "$repo2ExitCode" -eq "") {
    $repo2ExitCode = 0
}
if ($repo2ExitCode -ne 0) {
    $stderrText = ""
    if (Test-Path $repo2StdErr) {
        $stderrText = Get-Content $repo2StdErr -Raw
    }
    throw "truk_pipline failed with exit code $repo2ExitCode. $stderrText"
}

$fillRows = @()
if (Test-Path $repo1SummaryPath) {
    $fillRows = @(Import-Csv $repo1SummaryPath)
}

$newEventLines = @()
if (Test-Path $repo2EventsPath) {
    $allEventLines = Get-Content $repo2EventsPath
    if ($repo2StartCount -lt $allEventLines.Count) {
        $newEventLines = $allEventLines[$repo2StartCount..($allEventLines.Count - 1)]
    }
}

$events = @()
foreach ($line in $newEventLines) {
    if ([string]::IsNullOrWhiteSpace($line)) {
        continue
    }
    $events += ($line | ConvertFrom-Json)
}
$events = @($events)

$combinedLines = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $events.Count; $i++) {
    $fill = "N/A"
    if ($i -lt $fillRows.Count -and $null -ne $fillRows[$i].fill_percentage -and $fillRows[$i].fill_percentage -ne "") {
        $fill = ("{0}%" -f $fillRows[$i].fill_percentage)
    }
    $event = $events[$i]
    $lineTs = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $combinedLines.Add(
        [string]::Format(
            "{0} | TRUCK_ID={1} | COVER_STATUS={2} | MATERIAL={3} | VIOLATION={4} | FILL_LEVEL={5}",
            $lineTs,
            [string]$event.event_id,
            (Get-CoverStatus -Event $event),
            (Get-Material -Event $event),
            [string][bool]$event.violation,
            $fill
        )
    )
}

if ($combinedLines.Count -eq 0) {
    throw "No combined event lines were produced."
}

$combinedLines | Set-Content -Path $logPath -Encoding utf8
$combinedLines | ForEach-Object { $_ }

Remove-DirectoryFiles -DirectoryPath $repo1OutputDir
