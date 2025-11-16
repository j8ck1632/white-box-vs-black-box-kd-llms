# Script to push code to GitHub
# First, create a repository on GitHub at https://github.com/new
# Then update the REPO_URL variable below with your repository URL

param(
    [Parameter(Mandatory=$false)]
    [string]$RepoUrl = ""
)

$ErrorActionPreference = "Stop"

Write-Host "=== GitHub Repository Setup ===" -ForegroundColor Cyan

if ([string]::IsNullOrEmpty($RepoUrl)) {
    Write-Host "`nPlease create a repository on GitHub first:" -ForegroundColor Yellow
    Write-Host "1. Go to: https://github.com/new" -ForegroundColor White
    Write-Host "2. Repository name: white-box-vs-black-box-kd-llms (or your preferred name)" -ForegroundColor White
    Write-Host "3. Choose Public or Private" -ForegroundColor White
    Write-Host "4. DO NOT initialize with README, .gitignore, or license" -ForegroundColor White
    Write-Host "5. Click 'Create repository'" -ForegroundColor White
    Write-Host "`nThen run this script again with your repository URL:" -ForegroundColor Yellow
    Write-Host "  .\push_to_github.ps1 -RepoUrl 'https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git'" -ForegroundColor Green
    exit 0
}

Write-Host "`nSetting up remote repository..." -ForegroundColor Green

# Check if remote already exists
$existingRemote = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Remote 'origin' already exists: $existingRemote" -ForegroundColor Yellow
    $overwrite = Read-Host "Do you want to update it? (y/n)"
    if ($overwrite -eq "y" -or $overwrite -eq "Y") {
        git remote set-url origin $RepoUrl
        Write-Host "Remote updated successfully!" -ForegroundColor Green
    } else {
        Write-Host "Keeping existing remote." -ForegroundColor Yellow
    }
} else {
    git remote add origin $RepoUrl
    Write-Host "Remote added successfully!" -ForegroundColor Green
}

# Ensure we're on main branch
git branch -M main 2>$null

Write-Host "`nPushing to GitHub..." -ForegroundColor Green
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "Repository URL: $RepoUrl" -ForegroundColor Cyan
} else {
    Write-Host "`n❌ Error pushing to GitHub. Please check:" -ForegroundColor Red
    Write-Host "1. Repository exists on GitHub" -ForegroundColor Yellow
    Write-Host "2. You have push access" -ForegroundColor Yellow
    Write-Host "3. Your GitHub credentials are configured" -ForegroundColor Yellow
}

