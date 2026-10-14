# Launch the application with the isolated AMD DirectML environment.
$ErrorActionPreference = 'Stop'
$venvPath = Join-Path $PSScriptRoot ".venv-directml"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    $pythonLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -eq $pythonLauncher) {
        throw "Python Launcher (py.exe) is required to create the DirectML environment. Install Python 3.10 and try again."
    }

    Write-Host "Creating the DirectML environment with Python 3.10..."
    & $pythonLauncher.Source -3.10 -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to create the DirectML environment at '$venvPath'."
    }

    Write-Host "Installing DirectML dependencies..."
    & $pythonPath -m pip install -r (Join-Path $PSScriptRoot "requirements-directml.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to install DirectML dependencies."
    }
}
& $pythonPath (Join-Path $PSScriptRoot "main.py") @args
exit $LASTEXITCODE
exit $LASTEXITCODE
