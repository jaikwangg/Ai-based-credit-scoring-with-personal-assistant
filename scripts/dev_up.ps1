param(
    [switch]$Clean,
    [switch]$WithSmoke
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDir = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

function Stop-PortProcess {
    param([int]$Port)

    $lines = netstat -ano | Select-String ":$Port"
    if (-not $lines) {
        return
    }

    $pids = @()
    foreach ($line in $lines) {
        $parts = (($line.ToString().Trim()) -split "\s+")
        if ($parts.Length -ge 5) {
            $raw = $parts[-1]
            $parsed = 0
            if ([int]::TryParse($raw, [ref]$parsed) -and $parsed -gt 0) {
                $pids += $parsed
            }
        }
    }

    $pids = $pids | Select-Object -Unique
    foreach ($pidValue in $pids) {
        try {
            Stop-Process -Id $pidValue -Force -ErrorAction Stop
            Write-Host "Stopped PID $pidValue on port $Port"
        } catch {
            # Ignore processes that already exited.
        }
    }
}

function Wait-HttpHealthy {
    param(
        [string]$Name,
        [string]$Url,
        [int]$Retries = 45,
        [int]$SleepSec = 2
    )

    for ($i = 1; $i -le $Retries; $i++) {
        try {
            Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 5 | Out-Null
            Write-Host "$Name is up ($Url)"
            return $true
        } catch {
            Start-Sleep -Seconds $SleepSec
        }
    }

    throw "$Name failed to start in time: $Url"
}

if ($Clean) {
    Write-Host "Cleaning old listeners on ports 3000, 8000, 8001..."
    Stop-PortProcess -Port 3000
    Stop-PortProcess -Port 8000
    Stop-PortProcess -Port 8001
    Start-Sleep -Seconds 1
}

$plannerStdOut = Join-Path $logDir "planner.out.log"
$plannerStdErr = Join-Path $logDir "planner.err.log"
$backendStdOut = Join-Path $logDir "backend.out.log"
$backendStdErr = Join-Path $logDir "backend.err.log"
$frontendStdOut = Join-Path $logDir "frontend.out.log"
$frontendStdErr = Join-Path $logDir "frontend.err.log"

Write-Host "Starting planner (8001)..."
$planner = Start-Process `
    -FilePath "python" `
    -ArgumentList "-m uvicorn src.api.main:app --host 127.0.0.1 --port 8001" `
    -WorkingDirectory (Join-Path $repoRoot "Ai-Credit-Scoring") `
    -RedirectStandardOutput $plannerStdOut `
    -RedirectStandardError $plannerStdErr `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Starting backend bridge (8000)..."
$backend = Start-Process `
    -FilePath "python" `
    -ArgumentList "-m uvicorn main:app --app-dir app --host 0.0.0.0 --port 8000" `
    -WorkingDirectory (Join-Path $repoRoot "backend") `
    -RedirectStandardOutput $backendStdOut `
    -RedirectStandardError $backendStdErr `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Starting frontend (3000)..."
$frontend = Start-Process `
    -FilePath "cmd.exe" `
    -ArgumentList "/c npm run dev" `
    -WorkingDirectory (Join-Path $repoRoot "frontend") `
    -RedirectStandardOutput $frontendStdOut `
    -RedirectStandardError $frontendStdErr `
    -WindowStyle Hidden `
    -PassThru

Wait-HttpHealthy -Name "Planner" -Url "http://127.0.0.1:8001/health" | Out-Null
Wait-HttpHealthy -Name "Backend" -Url "http://127.0.0.1:8000/health" | Out-Null
Wait-HttpHealthy -Name "Frontend" -Url "http://127.0.0.1:3000/api/health" | Out-Null

Write-Host ""
Write-Host "All services are running."
Write-Host "Planner PID : $($planner.Id)"
Write-Host "Backend PID : $($backend.Id)"
Write-Host "Frontend PID: $($frontend.Id)"
Write-Host ""
Write-Host "Logs:"
Write-Host "  $plannerStdOut"
Write-Host "  $plannerStdErr"
Write-Host "  $backendStdOut"
Write-Host "  $backendStdErr"
Write-Host "  $frontendStdOut"
Write-Host "  $frontendStdErr"

if ($WithSmoke) {
    Write-Host ""
    Write-Host "Running smoke test..."
    powershell -ExecutionPolicy Bypass -File (Join-Path $repoRoot "scripts/prod_smoke.ps1")
}
