param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pipelineVenvDir = Join-Path $repoRoot ".venv_truk_pipline"
$pipelinePython = Join-Path $pipelineVenvDir "Scripts\python.exe"
$truckSizeRepo = Join-Path $repoRoot "truck_size_2"
$truckSizeVenvDir = Join-Path $repoRoot ".venv_truck_size_2"
$truckSizePython = Join-Path $truckSizeVenvDir "Scripts\python.exe"

$truckSizePinnedPackages = @(
    "ultralytics==8.4.30",
    "torch==2.11.0",
    "torchvision==0.26.0",
    "numpy==2.2.6",
    "opencv-python==4.13.0.92"
)

function New-OrUpdateVenv {
    param(
        [Parameter(Mandatory = $true)]
        [string]$VenvDir,
        [Parameter(Mandatory = $true)]
        [string]$PythonPath,
        [Parameter(Mandatory = $true)]
        [string[]]$InstallArgs
    )

    if (-not (Test-Path $PythonPath)) {
        & $PythonExe -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create venv at $VenvDir with interpreter: $PythonExe"
        }
    }

    & $PythonPath -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip tooling in $VenvDir"
    }

    foreach ($installArgSet in $InstallArgs) {
        & $PythonPath -m pip install @($installArgSet -split ' ')
        if ($LASTEXITCODE -ne 0) {
            throw "Failed installing dependencies in $VenvDir using: $installArgSet"
        }
    }

    & $PythonPath -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "pip check reported dependency issues in $VenvDir"
    }
}

if (-not (Test-Path $truckSizeRepo)) {
    throw "Missing submodule at $truckSizeRepo. Clone with --recurse-submodules or run: git submodule update --init --recursive"
}

Push-Location $repoRoot
try {
    New-OrUpdateVenv `
        -VenvDir $pipelineVenvDir `
        -PythonPath $pipelinePython `
        -InstallArgs @(
            "-r .\requirements.txt"
        )

    New-OrUpdateVenv `
        -VenvDir $truckSizeVenvDir `
        -PythonPath $truckSizePython `
        -InstallArgs @(
            "-r .\truck_size_2\requirements.txt",
            ($truckSizePinnedPackages -join ' ')
        )
}
finally {
    Pop-Location
}

Write-Output "Integrated environments are ready:"
Write-Output "  truk_pipline : $pipelinePython"
Write-Output "  truck_size_2 : $truckSizePython"
