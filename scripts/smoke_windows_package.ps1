$ErrorActionPreference = 'Stop'
# The Python harness owns installation order, prior-version upgrades, and repair.
# Keep one entrypoint so portable smoke cannot register a competing fresh install.
python (Join-Path $PSScriptRoot 'verify_windows_installation.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
