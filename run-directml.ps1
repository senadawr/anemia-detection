# Launch the application with the isolated AMD DirectML environment.
$ErrorActionPreference = 'Stop'
$venvPath = Join-Path $PSScriptRoot '.venv-directml'
$pythonPath = Join-Path $venvPath 'Scripts\python.exe'

if (-not (Test-Path $pythonPath)) {
	if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
		throw 'Python 3.10 is required for DirectML. Install Python 3.10 and rerun this script.'
	}

	& py -3.10 -m venv $venvPath
	if ($LASTEXITCODE -ne 0 -or -not (Test-Path $pythonPath)) {
		throw 'Python 3.10 is required for DirectML. Install Python 3.10 and rerun this script.'
	}

	& $pythonPath -m pip install -r (Join-Path $PSScriptRoot 'requirements-directml.txt')
	if ($LASTEXITCODE -ne 0) {
		throw 'DirectML dependency installation failed.'
	}
}

& $pythonPath (Join-Path $PSScriptRoot 'main.py') @args
