# Quick Start Script for Head Node (Your Computer)
# This script starts the Ray head node so your friend can connect
#
# Usage: Double-click start_head_node.bat or run this in PowerShell

# Set error handling
$ErrorActionPreference = "Continue"

# Global error handler
trap {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  Error occurred!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.InvocationInfo) {
        Write-Host "Location: Line $($_.InvocationInfo.ScriptLineNumber)" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Press Enter to exit..."
    try {
        Read-Host | Out-Null
    } catch {
        Start-Sleep -Seconds 5
    }
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Ray Cluster Head Node Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Script starting..." -ForegroundColor Gray
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonCheck = $false
try {
    $pythonVersion = & python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Found: $pythonVersion" -ForegroundColor Green
        $pythonCheck = $true
    } else {
        $pythonPath = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonPath) {
            Write-Host "Found Python at: $($pythonPath.Source)" -ForegroundColor Green
            $pythonCheck = $true
        }
    }
} catch {
    try {
        $pythonPath = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonPath) {
            Write-Host "Found Python at: $($pythonPath.Source)" -ForegroundColor Green
            $pythonCheck = $true
        }
    } catch {
        $pythonCheck = $false
    }
}

if (-not $pythonCheck) {
    Write-Host "Python not found! Please install Python first." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press Enter to exit..."
    try {
        Read-Host | Out-Null
    } catch {
        Start-Sleep -Seconds 5
    }
    exit 1
}

# Check if Ray is installed
Write-Host ""
Write-Host "Checking Ray installation..." -ForegroundColor Yellow
$rayInstalled = $false
try {
    $rayVersion = & ray --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Ray is installed: $rayVersion" -ForegroundColor Green
        $rayInstalled = $true
    }
} catch {
    $rayInstalled = $false
}

if (-not $rayInstalled) {
    Write-Host "Ray not found. Installing Ray..." -ForegroundColor Yellow
    Write-Host "This may take a few minutes..." -ForegroundColor Yellow
    try {
        & python -m pip install --upgrade pip 2>&1 | Out-Null
        & python -m pip install ray[default] 2>&1 | Out-Null
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Ray installed successfully!" -ForegroundColor Green
            $rayInstalled = $true
        } else {
            throw "Installation failed"
        }
    } catch {
        Write-Host "Failed to install Ray. Please install manually:" -ForegroundColor Red
        Write-Host "pip install ray[default]" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Press Enter to exit..."
        Read-Host | Out-Null
        exit 1
    }
}

# Check if GPU is available (non-blocking, optional)
Write-Host ""
Write-Host "Checking GPU availability..." -ForegroundColor Yellow
$gpuCheckJob = Start-Job -ScriptBlock {
    try {
        $gpuCheckCmd = 'import torch; print("CUDA available:", torch.cuda.is_available())'
        $gpuCheck = python -c $gpuCheckCmd 2>&1
        if ($LASTEXITCODE -eq 0 -and $gpuCheck -match "True") {
            $gpuCountCmd = 'import torch; print(torch.cuda.device_count())'
            $gpuCount = python -c $gpuCountCmd 2>&1
            return @{Success=$true; Count=$gpuCount}
        }
        return @{Success=$false}
    } catch {
        return @{Success=$false}
    }
}

# Wait max 3 seconds for GPU check
$gpuResult = $null
try {
    $gpuResult = Wait-Job $gpuCheckJob -Timeout 3 | Receive-Job
    Remove-Job $gpuCheckJob -Force -ErrorAction SilentlyContinue
} catch {
    Remove-Job $gpuCheckJob -Force -ErrorAction SilentlyContinue
}

if ($gpuResult -and $gpuResult.Success) {
    Write-Host "GPU detected and available!" -ForegroundColor Green
    if ($gpuResult.Count) {
        Write-Host "Found $($gpuResult.Count) GPU(s)" -ForegroundColor Green
    }
} else {
    Write-Host "GPU not detected or PyTorch not installed. Ray will still work but won't use GPU acceleration." -ForegroundColor Yellow
}

# Check for VPN/Tailscale
Write-Host ""
Write-Host "Checking for VPN connection..." -ForegroundColor Yellow
$tailscaleIP = $null
try {
    $tailscaleIP = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object {$_.IPAddress -like "100.*"} | Select-Object -First 1 -ExpandProperty IPAddress
} catch {
    # Ignore error
}

$useVPN = $false
if ($tailscaleIP) {
    Write-Host "Tailscale VPN detected! Your VPN IP: $tailscaleIP" -ForegroundColor Green
    Write-Host ""
    $useVPNChoice = Read-Host "Use Tailscale VPN IP? (Y/n)"
    if ([string]::IsNullOrWhiteSpace($useVPNChoice) -or $useVPNChoice -eq "Y" -or $useVPNChoice -eq "y") {
        $useVPN = $true
        $nodeIP = $tailscaleIP
    }
}

# Get local IP if not using VPN
if (-not $useVPN) {
    Write-Host ""
    Write-Host "Detecting local IP address..." -ForegroundColor Yellow
    $localIPs = @()
    try {
        $localIPs = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object {
            $_.IPAddress -notlike "127.*" -and 
            $_.IPAddress -notlike "169.254.*" -and
            $_.IPAddress -notlike "100.*"
        } | Select-Object -ExpandProperty IPAddress
    } catch {
        Write-Host "Could not automatically detect IP addresses." -ForegroundColor Yellow
    }
    
    if ($localIPs.Count -gt 1) {
        Write-Host "Multiple network interfaces found:" -ForegroundColor Yellow
        $i = 1
        foreach ($ip in $localIPs) {
            Write-Host "  $i. $ip" -ForegroundColor Gray
            $i++
        }
        Write-Host ""
        $choice = Read-Host "Select IP address (1-$($localIPs.Count))"
        if ($choice -match '^\d+$' -and [int]$choice -ge 1 -and [int]$choice -le $localIPs.Count) {
            $nodeIP = $localIPs[[int]$choice - 1]
        } else {
            $nodeIP = $localIPs[0]
            Write-Host "Using first IP: $nodeIP" -ForegroundColor Yellow
        }
    } elseif ($localIPs.Count -eq 1) {
        $nodeIP = $localIPs[0]
        Write-Host "Using local IP: $nodeIP" -ForegroundColor Green
    } else {
        Write-Host "Could not detect local IP. You'll need to specify it manually." -ForegroundColor Yellow
        $nodeIP = Read-Host "Enter your IP address"
        if ([string]::IsNullOrWhiteSpace($nodeIP)) {
            Write-Host "No IP address provided. Exiting." -ForegroundColor Red
            Write-Host ""
            Write-Host "Press Enter to exit..."
            Read-Host | Out-Null
            exit 1
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting Ray Head Node" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Ray is already running (with timeout to prevent hanging)
Write-Host "Checking if Ray is already running..." -ForegroundColor Yellow
$rayCheckJob = Start-Job -ScriptBlock {
    try {
        $status = ray status 2>&1 | Out-String
        return $status
    } catch {
        return "error"
    }
}

$rayStatus = $null
try {
    # Wait max 5 seconds for ray status check
    $rayStatus = Wait-Job $rayCheckJob -Timeout 5 | Receive-Job
    Remove-Job $rayCheckJob -Force -ErrorAction SilentlyContinue
} catch {
    Remove-Job $rayCheckJob -Force -ErrorAction SilentlyContinue
    Write-Host "Could not check Ray status (this is okay if Ray is not running)" -ForegroundColor Gray
}

if ($rayStatus -and $rayStatus -notmatch "No cluster status" -and $rayStatus -notmatch "error" -and $rayStatus.Length -gt 10) {
    Write-Host "Ray is already running. Stopping existing instance..." -ForegroundColor Yellow
    try {
        $stopJob = Start-Job -ScriptBlock { ray stop 2>&1 | Out-Null }
        Wait-Job $stopJob -Timeout 5 | Out-Null
        Remove-Job $stopJob -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    } catch {
        Write-Host "Note: Could not stop existing Ray instance. Continuing anyway..." -ForegroundColor Yellow
    }
} else {
    Write-Host "Ray is not running (or check timed out). Starting fresh..." -ForegroundColor Gray
}

# Start Ray head node
Write-Host "Starting Ray head node..." -ForegroundColor Yellow
if ($useVPN) {
    Write-Host "  Using VPN IP: $nodeIP" -ForegroundColor Gray
    Write-Host "  Share this IP with your friend: $nodeIP" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "  Using local IP: $nodeIP" -ForegroundColor Gray
    Write-Host "  Share this IP with your friend: $nodeIP" -ForegroundColor Cyan
    Write-Host ""
}

# Enable Windows/OSX cluster support (required for Windows)
Write-Host "Enabling Windows cluster support..." -ForegroundColor Gray
$env:RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER = "1"

# Execute Ray command
Write-Host "Executing Ray command..." -ForegroundColor Gray
$rayOutput = ""
$raySuccess = $false

try {
    if ($useVPN) {
        $rayOutput = & ray start --head --port=6379 --dashboard-host=0.0.0.0 --node-ip-address=$nodeIP 2>&1
    } else {
        $rayOutput = & ray start --head --port=6379 --dashboard-host=0.0.0.0 2>&1
    }
    
    if ($rayOutput -isnot [string]) {
        $rayOutput = $rayOutput | Out-String
    }
    
    if ($rayOutput -match "Ray runtime started" -or $rayOutput -match "Local node IP" -or $LASTEXITCODE -eq 0) {
        $raySuccess = $true
    } else {
        $raySuccess = $false
    }
} catch {
    $rayOutput = "Error executing Ray: $($_.Exception.Message)"
    $raySuccess = $false
}

# Show Ray output
if ($rayOutput) {
    Write-Host $rayOutput
}

if ($raySuccess) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Head Node Started Successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your Ray cluster is ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Share this information with your friend:" -ForegroundColor Cyan
    Write-Host "  IP Address: $nodeIP" -ForegroundColor Yellow
    Write-Host "  Port: 6379" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Your friend should run: start_worker_node.bat" -ForegroundColor Yellow
    Write-Host "  Or manually: ray start --address='$nodeIP`:6379'" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "Ray Dashboard:" -ForegroundColor Cyan
    Write-Host "  http://$nodeIP`:8265" -ForegroundColor Yellow
    Write-Host ""
    
    Write-Host "Current cluster status:" -ForegroundColor Cyan
    & ray status
    
    Write-Host ""
    Write-Host "Head node is running. Keep this window open!" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop when done." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Waiting for worker nodes to connect..." -ForegroundColor Gray
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
    Write-Host ""
    
    $iteration = 0
    while ($true) {
        Start-Sleep -Seconds 30
        $iteration++
        if ($iteration % 4 -eq 0) {
            Write-Host ""
            Write-Host "[Status Check] Current cluster:" -ForegroundColor Cyan
            & ray status
            Write-Host ""
        }
    }
} else {
    Write-Host ""
    Write-Host "Failed to start Ray head node." -ForegroundColor Red
    Write-Host ""
    Write-Host "Error details:" -ForegroundColor Yellow
    Write-Host $rayOutput -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check if port 6379 is already in use" -ForegroundColor Gray
    Write-Host "  2. Check firewall settings" -ForegroundColor Gray
    Write-Host "  3. Try running as administrator" -ForegroundColor Gray
    Write-Host "  4. Make sure Ray is properly installed: pip install ray[default]" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Press Enter to exit..."
    Read-Host | Out-Null
    exit 1
}
