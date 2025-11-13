# Script to create GitHub repository and push code
# This script attempts to create a repository via GitHub API and push the code

param(
    [Parameter(Mandatory=$false)]
    [string]$RepoName = "white-box-vs-black-box-kd-llms",
    
    [Parameter(Mandatory=$false)]
    [string]$Description = "Comparative Analysis of White-Box vs. Black-Box Knowledge Distillation in Language Models",
    
    [Parameter(Mandatory=$false)]
    [switch]$Private = $false,
    
    [Parameter(Mandatory=$false)]
    [string]$GitHubToken = ""
)

$ErrorActionPreference = "Stop"

Write-Host "=== GitHub Repository Creation and Push ===" -ForegroundColor Cyan

# Try to get GitHub username
$githubUser = ""
try {
    $gitConfig = git config --get github.user 2>$null
    if ($gitConfig) {
        $githubUser = $gitConfig
    }
} catch {}

if ([string]::IsNullOrEmpty($githubUser)) {
    $githubUser = Read-Host "Enter your GitHub username"
}

# Try to get token from git credential or environment
if ([string]::IsNullOrEmpty($GitHubToken)) {
    $envToken = $env:GITHUB_TOKEN
    if ($envToken) {
        $GitHubToken = $envToken
    }
}

if ([string]::IsNullOrEmpty($GitHubToken)) {
    Write-Host "`nGitHub Personal Access Token is required to create repository via API." -ForegroundColor Yellow
    Write-Host "You can create one at: https://github.com/settings/tokens" -ForegroundColor White
    Write-Host "Required scope: 'repo' (full control of private repositories)" -ForegroundColor White
    Write-Host "`nOption 1: Enter token now (will not be saved)" -ForegroundColor Yellow
    Write-Host "Option 2: Create repository manually at https://github.com/new" -ForegroundColor Yellow
    $choice = Read-Host "`nEnter '1' to provide token, or '2' to create manually"
    
    if ($choice -eq "1") {
        $GitHubToken = Read-Host "Enter your GitHub Personal Access Token" -AsSecureString
        $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($GitHubToken)
        $GitHubToken = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
    } else {
        Write-Host "`nManual setup:" -ForegroundColor Cyan
        Write-Host "1. Go to: https://github.com/new" -ForegroundColor White
        Write-Host "2. Repository name: $RepoName" -ForegroundColor White
        Write-Host "3. Description: $Description" -ForegroundColor White
        Write-Host "4. Choose visibility: $(if($Private){'Private'}else{'Public'})" -ForegroundColor White
        Write-Host "5. DO NOT initialize with README, .gitignore, or license" -ForegroundColor White
        Write-Host "6. Click 'Create repository'" -ForegroundColor White
        Write-Host "`nThen run:" -ForegroundColor Yellow
        Write-Host "  git remote add origin https://github.com/$githubUser/$RepoName.git" -ForegroundColor Green
        Write-Host "  git push -u origin main" -ForegroundColor Green
        exit 0
    }
}

# Create repository via GitHub API
Write-Host "`nCreating repository '$RepoName' on GitHub..." -ForegroundColor Green

$headers = @{
    "Authorization" = "token $GitHubToken"
    "Accept" = "application/vnd.github.v3+json"
}

$body = @{
    name = $RepoName
    description = $Description
    private = $Private.IsPresent
    auto_init = $false
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Method Post -Headers $headers -Body $body -ContentType "application/json"
    Write-Host "✅ Repository created successfully!" -ForegroundColor Green
    $repoUrl = $response.html_url
    Write-Host "Repository URL: $repoUrl" -ForegroundColor Cyan
    
    # Add remote and push
    Write-Host "`nSetting up remote..." -ForegroundColor Green
    
    # Remove existing origin if it exists
    $existingRemote = git remote get-url origin 2>$null
    if ($LASTEXITCODE -eq 0) {
        git remote remove origin 2>$null
    }
    
    git remote add origin $response.clone_url
    Write-Host "Remote added successfully!" -ForegroundColor Green
    
    # Push to GitHub
    Write-Host "`nPushing to GitHub..." -ForegroundColor Green
    git branch -M main 2>$null
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Successfully pushed to GitHub!" -ForegroundColor Green
        Write-Host "Repository URL: $repoUrl" -ForegroundColor Cyan
    } else {
        Write-Host "`n❌ Error pushing to GitHub. Please check your credentials." -ForegroundColor Red
    }
    
} catch {
    Write-Host "`n❌ Error creating repository: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Response.StatusCode -eq 401) {
        Write-Host "Authentication failed. Please check your token." -ForegroundColor Yellow
    } elseif ($_.Exception.Response.StatusCode -eq 422) {
        Write-Host "Repository might already exist. Trying to use existing repository..." -ForegroundColor Yellow
        $repoUrl = "https://github.com/$githubUser/$RepoName"
        
        # Try to add remote and push anyway
        $existingRemote = git remote get-url origin 2>$null
        if ($LASTEXITCODE -eq 0) {
            git remote set-url origin "$repoUrl.git"
        } else {
            git remote add origin "$repoUrl.git"
        }
        
        Write-Host "Attempting to push to existing repository..." -ForegroundColor Yellow
        git branch -M main 2>$null
        git push -u origin main
    }
}

