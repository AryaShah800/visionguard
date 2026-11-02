$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$visionPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $visionPython)) {
    throw 'Create the project environment using the README setup commands first.'
}
& $visionPython -m streamlit run app.py
