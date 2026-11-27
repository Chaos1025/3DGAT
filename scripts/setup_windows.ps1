#requires -Version 5.1
<#
.SYNOPSIS
    Create a Windows-ready conda environment for 3DGAT and build its CUDA extensions.

.DESCRIPTION
    The repository's environment.yml is Linux-only (it pins ld_impl_linux-64,
    _libgcc_mutex, etc. and cannot be solved by conda on Windows). This script
    reproduces the equivalent Windows setup:

      1. create the conda env (python 3.9)
      2. install PyTorch 2.1.1 with the CUDA 12.1 wheels from the official index
      3. install the remaining pinned Python deps (requirements-windows.txt)
      4. build + install the two CUDA extensions:
           submodules/simple-knn
           submodules/us_gaussian_voxelization
      5. verify torch, CUDA and both extensions import cleanly

    Requirements checked up front:
      - conda on PATH
      - NVIDIA driver with a CUDA 12.x capable GPU
      - CUDA Toolkit (nvcc) 12.x        -> -CudaHome if not auto-detected
      - Visual Studio 2022 with "Desktop development with C++" -> vcvars64.bat

.PARAMETER EnvName
    Name of the conda environment to create. Default: 3dgat

.PARAMETER TorchVersion
    PyTorch version to install. Default: 2.1.1 (cu121 wheels)

.PARAMETER CudaHome
    CUDA Toolkit root, e.g. 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'.
    Auto-detected from CUDA_PATH / the default install dir when omitted.

.PARAMETER Recreate
    Remove an existing environment of the same name before creating it.

.EXAMPLE
    pwsh -File scripts\setup_windows.ps1

.EXAMPLE
    pwsh -File scripts\setup_windows.ps1 -EnvName 3dgat -CudaHome 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
#>
[CmdletBinding()]
param(
    [string]$EnvName = "3dgat",
    [string]$TorchVersion = "2.1.1",
    [string]$CudaHome = "",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host "    [ok] $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host "    [warn] $Message" -ForegroundColor Yellow
}

function Fail([string]$Message) {
    Write-Host ""
    Write-Host "ERROR: $Message" -ForegroundColor Red
    exit 1
}

# Repo root = parent of this script's directory
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $RepoRoot "3DGAT.py"))) {
    Fail "Could not locate repo root from script path ($RepoRoot). Run the script from inside the repository."
}

# ---------------------------------------------------------------------------
# 0. pre-flight checks
# ---------------------------------------------------------------------------
Write-Step "Checking prerequisites"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Fail "conda was not found on PATH. Install Miniconda/Anaconda and open a new terminal."
}
Write-Ok "conda: $((Get-Command conda).Source)"

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Fail "nvidia-smi was not found. Install an NVIDIA driver first."
}
$gpuLine = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | Select-Object -First 1)
Write-Ok "GPU: $gpuLine"

# --- CUDA toolkit ---
if ([string]::IsNullOrWhiteSpace($CudaHome)) {
    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) {
        $CudaHome = $env:CUDA_PATH
    }
    else {
        $base = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if (Test-Path $base) {
            $candidate = Get-ChildItem $base -Directory |
                Where-Object { Test-Path (Join-Path $_.FullName "bin\nvcc.exe") } |
                Sort-Object Name -Descending |
                Select-Object -First 1
            if ($candidate) { $CudaHome = $candidate.FullName }
        }
    }
}
if ([string]::IsNullOrWhiteSpace($CudaHome) -or -not (Test-Path (Join-Path $CudaHome "bin\nvcc.exe"))) {
    Fail "Could not find the CUDA Toolkit (nvcc.exe). Install CUDA 12.x or pass -CudaHome '<path>'."
}
$nvccVersion = (& (Join-Path $CudaHome "bin\nvcc.exe") --version | Select-String "release").ToString().Trim()
Write-Ok "CUDA: $CudaHome ($nvccVersion)"

# --- Visual Studio / MSVC ---
# CUDA 12.x rejects MSVC from VS 2026+ ("unsupported Microsoft Visual Studio
# version! Only ... 2017 and 2022"). torch 2.1.1 also cannot forward
# '-allow-unsupported-compiler' via NVCC_PREPEND_FLAGS, so we MUST select a
# 2017..2022 toolchain explicitly instead of trusting `vswhere -latest`.
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vcvars = $null
$vsDesc = ""

function Find-Vcvars {
    param([string]$VersionRange)
    if (-not (Test-Path $vswhere)) { return $null }
    $queryArgs = @("-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath")
    if ($VersionRange) { $queryArgs += @("-version", $VersionRange) }
    $paths = & $vswhere @queryArgs
    foreach ($p in $paths) {
        if ([string]::IsNullOrWhiteSpace($p)) { continue }
        $bat = Join-Path $p "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $bat) { return $bat }
    }
    return $null
}

# Prefer VS 2022 (17.x), then 2019 (16.x), then 2017 (15.x).
foreach ($range in @("[17.0,18.0)", "[16.0,17.0)", "[15.0,16.0)")) {
    $vcvars = Find-Vcvars $range
    if ($vcvars) { $vsDesc = "VS $range"; break }
}

if (-not $vcvars) {
    # Last resort: scan the filesystem, still refusing 18.x+ by name heuristics.
    $fallback = Get-ChildItem "C:\Program Files\Microsoft Visual Studio", "C:\Program Files (x86)\Microsoft Visual Studio" `
        -Recurse -Filter "vcvars64.bat" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match '\\(2017|2019|2022)\\' } |
        Sort-Object FullName -Descending |
        Select-Object -First 1
    if ($fallback) { $vcvars = $fallback.FullName; $vsDesc = "filesystem 2017/2019/2022" }
}

if (-not $vcvars) {
    $found = & $vswhere -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property displayName 2>$null
    $foundText = if ($found) { ($found -join "; ") } else { "none" }
    Fail @"
No CUDA-compatible Visual Studio found (need 2017/2019/2022 with the C++ workload).
Detected installations: $foundText
CUDA 12.x cannot build with VS 2026+ — install Visual Studio 2022 (or 2019) instead.
"@
}
Write-Ok "MSVC ($vsDesc): $vcvars"

# ---------------------------------------------------------------------------
# 1. conda environment
# ---------------------------------------------------------------------------
Write-Step "Preparing conda environment '$EnvName'"

$envList = (& conda env list --json | ConvertFrom-Json).envs
$envPath = $envList | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } | Select-Object -First 1

if ($envPath -and $Recreate) {
    Write-Host "    removing existing env at $envPath"
    & conda env remove -n $EnvName -y | Out-Null
    $envPath = $null
}

if (-not $envPath) {
    & conda create -n $EnvName python=3.9 -y
    if ($LASTEXITCODE -ne 0) { Fail "conda create failed." }
    $envPath = (& conda env list --json | ConvertFrom-Json).envs |
        Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } | Select-Object -First 1
    if (-not $envPath) { Fail "Environment '$EnvName' was not created." }
}
else {
    Write-Warn "Environment '$EnvName' already exists at $envPath (pass -Recreate to rebuild it)."
}
$PyExe = Join-Path $envPath "python.exe"
Write-Ok "python: $PyExe"

# ---------------------------------------------------------------------------
# 2. PyTorch (CUDA 12.1 wheels)
# ---------------------------------------------------------------------------
Write-Step "Installing PyTorch $TorchVersion (cu121)"
& $PyExe -m pip install --upgrade pip
& $PyExe -m pip install `
    "torch==$TorchVersion" `
    "torchvision==0.16.1" `
    "torchaudio==$TorchVersion" `
    --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { Fail "PyTorch installation failed." }

# ---------------------------------------------------------------------------
# 3. remaining Python deps (pins numpy back to 1.24.0, which torch 2.1.1 needs)
# ---------------------------------------------------------------------------
Write-Step "Installing Python dependencies (requirements-windows.txt)"
& $PyExe -m pip install -r (Join-Path $RepoRoot "requirements-windows.txt")
if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed." }

# ---------------------------------------------------------------------------
# 4. build CUDA extensions (need MSVC + nvcc => run under vcvars64)
# ---------------------------------------------------------------------------
function Build-Extension([string]$Name, [string]$RelativePath) {
    Write-Step "Building CUDA extension: $Name"
    $extDir = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path $extDir)) { Fail "Extension directory not found: $extDir" }

    # simple-knn's setup.py packages a 'simple_knn' folder that .gitignore used to
    # exclude entirely; make sure the (tracked) package dir exists before install.
    $pkgDir = $null
    if ($Name -eq "simple-knn") { $pkgDir = Join-Path $extDir "simple_knn" }
    if ($pkgDir -and -not (Test-Path $pkgDir)) {
        New-Item -ItemType Directory -Force -Path $pkgDir | Out-Null
    }
    if ($pkgDir -and -not (Test-Path (Join-Path $pkgDir "__init__.py"))) {
        Set-Content -Path (Join-Path $pkgDir "__init__.py") -Value "" -NoNewline
    }

    # Build through a temporary .bat so each line is expanded AFTER vcvars64 runs.
    # A single `cmd /c "call vcvars && set PATH=...%PATH%..."` would expand %PATH%
    # at parse time (before vcvars), dropping the compiler it just added.
    $batPath = Join-Path ([System.IO.Path]::GetTempPath()) "3dgat_build_$([System.IO.Path]::GetRandomFileName()).bat"
    $batLines = @(
        "@echo off",
        "call `"$vcvars`" >nul",
        "if errorlevel 1 (echo vcvars64 failed & exit /b 1)",
        "set `"CUDA_HOME=$CudaHome`"",
        "set `"CUDA_PATH=$CudaHome`"",
        "set `"PATH=$CudaHome\bin;%PATH%`"",
        "set `"DISTUTILS_USE_SDK=1`"",
        "cd /d `"$extDir`"",
        "`"$PyExe`" -m pip install -e . --no-build-isolation"
    )
    Set-Content -Path $batPath -Value $batLines -Encoding ascii

    try {
        & cmd /c "`"$batPath`""
        $code = $LASTEXITCODE
    }
    finally {
        Remove-Item -LiteralPath $batPath -ErrorAction SilentlyContinue
    }
    if ($code -ne 0) { Fail "Building '$Name' failed." }
    Write-Ok "$Name built"
}

Build-Extension "simple-knn" "submodules\simple-knn"
Build-Extension "us_gaussian_voxelization" "submodules\us_gaussian_voxelization"

# ---------------------------------------------------------------------------
# 5. verification
# ---------------------------------------------------------------------------
Write-Step "Verifying installation"
$verify = @'
import torch
import simple_knn._C
import us_gaussian_voxelization
from us_gaussian_voxelization import _C as vox_C
import lpips, kornia, pytorch_lightning, torchmetrics
import plyfile, yaml, tifffile, skimage, pandas, scipy, matplotlib, tqdm, einops, imageio

print(f"torch           {torch.__version__}")
print(f"cuda available  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device          {torch.cuda.get_device_name(0)}")
print("simple_knn/_C   ok")
print("voxelization    ok")
print("python deps     ok")
'@
& $PyExe -c $verify
if ($LASTEXITCODE -ne 0) { Fail "Verification failed." }

Write-Host ""
Write-Host "Environment '$EnvName' is ready." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  conda activate $EnvName"
Write-Host "  python onestep_wiener.py --root_dir data/flfm/reticular1"
Write-Host ""
