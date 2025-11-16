# Quick script to create GitHub repo and push (requires GitHub token)

param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubToken,
    
    [Parameter(Mandatory=$false)]
    [string]$RepoName = "white-box-vs-black-box-kd-llms",
    
    [Parameter(Mandatory=$false)]
    [string]$Description = "Comparative Analysis of White-Box vs. Black-Box Knowledge Distillation in Language Models",
    
    [Parameter(Mandatory=$false)]
    [switch]$Private = $false
)

$ErrorActionPreference = "Stop"

Write-Host "Creating repository '$RepoName' on GitHub..." -ForegroundColor Green

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
    Write-Host "Repository URL: $($response.html_url)" -ForegroundColor Cyan
    
    Write-Host "`nPushing code to GitHub..." -ForegroundColor Green
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Successfully pushed to GitHub!" -ForegroundColor Green
        Write-Host "Repository URL: $($response.html_url)" -ForegroundColor Cyan
    }
} catch {
    Write-Host "`n❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Response.StatusCode -eq 422) {
        Write-Host "Repository might already exist. Trying to push anyway..." -ForegroundColor Yellow
        git push -u origin main
    }
}



