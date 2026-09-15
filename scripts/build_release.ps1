param(
    [string]$Destination = ".\release\SatQuery-AI-rc1.zip"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Stage = Join-Path $env:TEMP ("satquery_release_" + [guid]::NewGuid().ToString("N"))
$Package = Join-Path $Stage "SatQuery-AI"
New-Item -ItemType Directory -Path $Package -Force | Out-Null

function Copy-CleanTree([string]$Relative) {
    $Source = Join-Path $Root $Relative
    if (-not (Test-Path $Source)) { return }
    $Target = Join-Path $Package $Relative
    New-Item -ItemType Directory -Path $Target -Force | Out-Null
    robocopy $Source $Target /E /NFL /NDL /NJH /NJS /NP `
        /XD __pycache__ uploads map_previews outputs .pytest_cache `
        /XF *.pyc *.pyo *.zip *.log *.before_* | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed for $Relative with code $LASTEXITCODE" }
}

foreach ($dir in @("ai", "backend", "frontend", "query_engine", "tests", "scripts")) {
    Copy-CleanTree $dir
}
foreach ($file in @("gee_temporal.py", ".gitignore", ".env.example", "requirements-runtime.txt")) {
    $Source = Join-Path $Root $file
    if (Test-Path $Source) { Copy-Item $Source (Join-Path $Package $file) -Force }
}

if (-not (Test-Path (Join-Path $Package "ai\__init__.py"))) {
    throw "Release build blocked: ai/ package is missing. A deployable SatQuery release must include its specialist execution package."
}

$DestinationPath = [System.IO.Path]::GetFullPath((Join-Path $Root $Destination))
New-Item -ItemType Directory -Path (Split-Path -Parent $DestinationPath) -Force | Out-Null
if (Test-Path $DestinationPath) { Remove-Item $DestinationPath -Force }
Compress-Archive -Path (Join-Path $Package "*") -DestinationPath $DestinationPath -CompressionLevel Optimal
Remove-Item $Stage -Recurse -Force
Write-Host "Release package created: $DestinationPath" -ForegroundColor Green
