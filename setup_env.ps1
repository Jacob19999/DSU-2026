# Setup script for DSU-2026 project environment
# Run this to create a consistent virtual environment

Write-Host "Setting up virtual environment..." -ForegroundColor Green

# Create venv if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`nSetup complete! Virtual environment is ready." -ForegroundColor Green
Write-Host "To activate in the future, run: .venv\Scripts\Activate.ps1" -ForegroundColor Cyan
