# Quick Start Script for Friend's Computer (Worker Node)
# This script helps your friend connect their GPU to your Ray cluster
#
# Usage: Just run this script and follow the prompts!

$ErrorActionPreference = "Continue"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Ray Cluster Worker Node Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
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
    Read-Host | Out-Null
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

# Check if GPU is available
Write-Host ""
Write-Host "Checking GPU availability..." -ForegroundColor Yellow
try {
    $gpuCheckCmd = 'import torch; print("CUDA available:", torch.cuda.is_available())'
    $gpuCheck = & python -c $gpuCheckCmd 2>&1
    if ($gpuCheck -match "True") {
        Write-Host "GPU detected and available!" -ForegroundColor Green
        $gpuCountCmd = 'import torch; print(torch.cuda.device_count())'
        $gpuCount = & python -c $gpuCountCmd 2>&1
        Write-Host "Found $gpuCount GPU(s)" -ForegroundColor Green
    } else {
        Write-Host "GPU not detected. Ray will still work but won't use GPU acceleration." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Could not check GPU (PyTorch may not be installed). Continuing anyway..." -ForegroundColor Yellow
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

if ($tailscaleIP) {
    Write-Host "Tailscale VPN detected! Your VPN IP: $tailscaleIP" -ForegroundColor Green
    Write-Host "Make sure the head node is also on the same VPN network." -ForegroundColor Yellow
} else {
    Write-Host "No Tailscale VPN detected." -ForegroundColor Yellow
    Write-Host "If connecting remotely, make sure VPN is running!" -ForegroundColor Yellow
}

# Get head node address
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Connection Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Enter the head node address:" -ForegroundColor Yellow
Write-Host "  - For local network: Use the local IP (e.g., 192.168.1.100)" -ForegroundColor Gray
Write-Host "  - For VPN/remote: Use the VPN IP (e.g., 100.64.1.2)" -ForegroundColor Gray
Write-Host ""

$headNodeIP = Read-Host "Head node IP address"

if ($headNodeIP -notmatch '^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$') {
    Write-Host "Warning: IP address format looks unusual. Continuing anyway..." -ForegroundColor Yellow
}

Write-Host ""
$port = Read-Host "Port (press Enter for default: 6379)"
if ([string]::IsNullOrWhiteSpace($port)) {
    $port = "6379"
}

$rayAddress = "${headNodeIP}:${port}"

Write-Host ""
Write-Host "Connecting to Ray cluster at: $rayAddress" -ForegroundColor Cyan
Write-Host ""

# Check if Ray is already running
try {
    $rayStatus = & ray status 2>&1 | Out-String
    if ($rayStatus -notmatch "No cluster status") {
        Write-Host "Ray is already running. Stopping existing Ray instance..." -ForegroundColor Yellow
        & ray stop 2>&1 | Out-Null
        Start-Sleep -Seconds 2
    }
} catch {
    # Ray not running, which is fine
}

# Start Ray worker
Write-Host "Starting Ray worker node..." -ForegroundColor Yellow
Write-Host "This will connect to: $rayAddress" -ForegroundColor Gray
Write-Host ""

# Enable Windows/OSX cluster support (required for Windows)
Write-Host "Enabling Windows cluster support..." -ForegroundColor Gray
$env:RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER = "1"

try {
    $rayOutput = & ray start --address=$rayAddress 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "  Successfully Connected!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Your GPU is now available to the cluster!" -ForegroundColor Green
        Write-Host ""
        Write-Host "To verify connection, the head node can run: ray status" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "To stop this worker node, press Ctrl+C or run: ray stop" -ForegroundColor Yellow
        Write-Host ""
        
        Write-Host "Current cluster status:" -ForegroundColor Cyan
        & ray status
        
        Write-Host ""
        Write-Host "Worker node is running. Keep this window open!" -ForegroundColor Green
        Write-Host "Press Ctrl+C to stop when done." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Waiting... (Press Ctrl+C to stop)" -ForegroundColor Gray
        
        while ($true) {
            Start-Sleep -Seconds 10
        }
    } else {
        Write-Host ""
        Write-Host "Failed to connect to Ray cluster." -ForegroundColor Red
        Write-Host ""
        Write-Host "Troubleshooting:" -ForegroundColor Yellow
        Write-Host "  1. Make sure the head node is running:" -ForegroundColor Gray
        Write-Host "     Head node should run: ray start --head" -ForegroundColor Gray
        Write-Host "  2. Check the IP address is correct" -ForegroundColor Gray
        Write-Host "  3. Check firewall settings" -ForegroundColor Gray
        Write-Host "  4. If using VPN, make sure both computers are connected" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Press Enter to exit..."
        Read-Host | Out-Null
        exit 1
    }
} catch {
    Write-Host ""
    Write-Host "Error connecting to Ray cluster:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Verify the head node is running and accessible" -ForegroundColor Gray
    Write-Host "  2. Check network connectivity: ping $headNodeIP" -ForegroundColor Gray
    Write-Host "  3. Make sure firewall allows Ray connections" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Press Enter to exit..."
    Read-Host | Out-Null
    exit 1
}
