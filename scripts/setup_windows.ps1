param(
  [string]$PythonExe = "py",
  [string]$PythonVersion = "3.12",
  [string]$VenvDir = ".venv.windows"
)

$ErrorActionPreference = "Stop"

& $PythonExe "-$PythonVersion" -m venv $VenvDir
& "$VenvDir\Scripts\python.exe" -m pip install --upgrade pip
& "$VenvDir\Scripts\python.exe" -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
}

Write-Host "Setup complete."
Write-Host "Activate with: .\$VenvDir\Scripts\Activate.ps1"
