$ErrorActionPreference = "Stop"

Write-Host "Checking Python version..."

python --version

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

Write-Host "Activating virtual environment..."
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Upgrading installer tools..."
python -m pip install --upgrade pip setuptools wheel

$DependencyFile = "requirements-lock.txt"

if (-not (Test-Path $DependencyFile)) {
    $DependencyFile = "requirements.txt"
}

Write-Host "Installing dependencies from $DependencyFile..."
python -m pip install -r $DependencyFile

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"

    Write-Host ""
    Write-Host "Created .env from .env.example."
    Write-Host "Add your GROQ_API_KEY before running the application."
}

Write-Host ""
Write-Host "Installation completed."
Write-Host ""
Write-Host "Next commands:"
Write-Host "  python -m src.vectordb"
Write-Host "  python -m src.rag"
Write-Host "  python -m src.run_eval"