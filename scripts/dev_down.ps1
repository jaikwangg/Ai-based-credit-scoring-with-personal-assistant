$ErrorActionPreference = "Stop"

function Stop-PortProcess {
    param([int]$Port)

    $lines = netstat -ano | Select-String ":$Port"
    if (-not $lines) {
        Write-Host "Port $Port is already free."
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

Stop-PortProcess -Port 3000
Stop-PortProcess -Port 8000
Stop-PortProcess -Port 8001

Write-Host "Done."

