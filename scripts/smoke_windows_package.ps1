$ErrorActionPreference = 'Stop'

function Test-NativeApplication([string] $Executable, [string] $Report) {
    Remove-Item -LiteralPath $Report -ErrorAction SilentlyContinue
    $env:PHOTO_SELECT_SMOKE_TEST_OUTPUT = $Report
    $process = Start-Process -FilePath $Executable -WorkingDirectory (Split-Path $Executable) -PassThru
    if (-not $process.WaitForExit(60000)) {
        Stop-Process -Id $process.Id -Force
        throw "Application installation probe timed out: $Executable"
    }
    $process.Refresh()
    if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $Report)) {
        throw "Application installation probe failed: $Executable (exit $($process.ExitCode))"
    }
    $result = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
    if (-not $result.engineAvailable -or $result.version -ne '0.2.0') {
        throw "Packaged engine was unavailable or the application version was wrong."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $result.lightroomPluginPath 'Info.lua'))) {
        throw "Bundled Lightroom plugin was missing."
    }
    Write-Output ($result | ConvertTo-Json)
    Remove-Item Env:PHOTO_SELECT_SMOKE_TEST_OUTPUT
}

$repository = Split-Path $PSScriptRoot
$packages = Join-Path $repository 'artifacts/windows'
$temporaryRoot = Join-Path $env:RUNNER_TEMP 'Photo Select package verification'
New-Item -ItemType Directory -Path $temporaryRoot -Force | Out-Null
$portable = Get-ChildItem -LiteralPath $packages -Filter '*-Portable.zip'
if ($portable.Count -ne 1) { throw 'Expected exactly one portable package.' }
$extracted = Join-Path $temporaryRoot 'portable'
Expand-Archive -LiteralPath $portable.FullName -DestinationPath $extracted -Force
$portableExecutable = Get-ChildItem -LiteralPath $extracted -Filter 'Photo Select.exe' -Recurse
if ($portableExecutable.Count -ne 1) { throw 'Portable application executable was missing.' }
Test-NativeApplication $portableExecutable.FullName (Join-Path $temporaryRoot 'portable-report.json')

$installer = Get-ChildItem -LiteralPath $packages -Filter '*-Setup.exe'
if ($installer.Count -ne 1) { throw 'Expected exactly one installer.' }
$installed = Join-Path $temporaryRoot 'installed'
# NSIS requires /D to be last; its value includes the rest of the command line.
$installation = Start-Process -FilePath $installer.FullName -ArgumentList "/S /D=$installed" -PassThru
if (-not $installation.WaitForExit(180000)) {
    Stop-Process -Id $installation.Id -Force
    throw 'Silent installer timed out.'
}
$installation.Refresh()
if ($installation.ExitCode -ne 0) { throw "Installer failed with exit $($installation.ExitCode)." }
Test-NativeApplication (Join-Path $installed 'photo-select.exe') (Join-Path $temporaryRoot 'installer-report.json')
Copy-Item -Path (Join-Path $temporaryRoot '*-report.json') -Destination $packages
