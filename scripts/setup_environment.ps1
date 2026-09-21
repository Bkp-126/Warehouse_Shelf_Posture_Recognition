param([ValidateSet('cpu','cu128')][string]$Device='cu128',[string]$EnvironmentName='warehouse-pose')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$existing=(& conda env list --json | ConvertFrom-Json).envs
if ($existing | Where-Object { (Split-Path -Leaf $_) -eq $EnvironmentName }) {
    throw "Environment $EnvironmentName already exists. This setup script only creates new environments."
}
& conda create -n $EnvironmentName python=3.10 pip -y
if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed' }
& conda run -n $EnvironmentName python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url "https://download.pytorch.org/whl/$Device"
if ($LASTEXITCODE -ne 0) { throw 'Torch installation failed' }
& conda run -n $EnvironmentName python -m pip install -r (Join-Path $root 'requirements-dev.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependencies installation failed' }
if ($Device -eq 'cu128') {
    & conda run -n $EnvironmentName python -m pip install -r (Join-Path $root 'requirements.lock.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Locked dependencies installation failed' }
}
& conda run -n $EnvironmentName python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed' }
Write-Output "Ready: conda activate $EnvironmentName"
