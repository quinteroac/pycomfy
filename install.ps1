# install.ps1 — Install the parallax CLI on Windows
# Usage: irm https://raw.githubusercontent.com/quinteroac/comfy-diffusion/master/install.ps1 | iex
# Env overrides:
#   PARALLAX_VERSION  — install a specific version tag (e.g. v1.2.0)

$ErrorActionPreference = "Stop"

$REPO         = "quinteroac/comfy-diffusion"
$ASSET        = "parallax-windows-x86_64.exe"
$INSTALL_DIR  = Join-Path $env:APPDATA "parallax\bin"
$INSTALL_PATH = Join-Path $INSTALL_DIR "parallax.exe"

# ── version resolution (AC02) ─────────────────────────────────────────────

if ($env:PARALLAX_VERSION) {
    $Version = $env:PARALLAX_VERSION
    Write-Host "[parallax] Using requested version: $Version"
} else {
    $ApiUrl = "https://api.github.com/repos/$REPO/releases/latest"
    Write-Host "[parallax] Fetching latest release from GitHub..."
    $Response = Invoke-WebRequest -Uri $ApiUrl -UseBasicParsing
    $Json = $Response.Content | ConvertFrom-Json
    $Version = $Json.tag_name
    if (-not $Version) {
        Write-Host "Could not fetch latest release. Set PARALLAX_VERSION=vX.X.X to install a specific version."
        exit 1
    }
    Write-Host "[parallax] Latest version: $Version"
}

# Strip leading 'v' to get bare semver (used in success message)
$BareVersion = $Version.TrimStart('v')

# ── download (AC03) ────────────────────────────────────────────────────────

$BaseUrl     = "https://github.com/$REPO/releases/download/$Version"
$BinaryUrl   = "$BaseUrl/$ASSET"
$ChecksumUrl = "$BaseUrl/$ASSET.sha256"

$TmpDir      = [System.IO.Path]::GetTempPath()
$TmpBinary   = Join-Path $TmpDir $ASSET
$TmpChecksum = Join-Path $TmpDir "$ASSET.sha256"

Write-Host "[parallax] Downloading $ASSET..."

# Ensure progress bar is visible even if the caller suppressed it
$ProgressPreference = "Continue"
Invoke-WebRequest -Uri $BinaryUrl   -OutFile $TmpBinary   -UseBasicParsing
Invoke-WebRequest -Uri $ChecksumUrl -OutFile $TmpChecksum -UseBasicParsing

# ── checksum verification (AC04) ──────────────────────────────────────────

Write-Host "[parallax] Verifying checksum..."

$ExpectedHash = (Get-Content $TmpChecksum -Raw).Trim().Split(' ')[0].ToUpper()
$ActualHash   = (Get-FileHash -Path $TmpBinary -Algorithm SHA256).Hash.ToUpper()

if ($ActualHash -ne $ExpectedHash) {
    Remove-Item -Path $TmpBinary -Force
    Write-Host "Checksum verification failed. Aborting."
    exit 1
}

Write-Host "[parallax] Checksum OK."

# ── install (AC05) ─────────────────────────────────────────────────────────

if (-not (Test-Path $INSTALL_DIR)) {
    New-Item -ItemType Directory -Path $INSTALL_DIR -Force | Out-Null
}

Copy-Item -Path $TmpBinary -Destination $INSTALL_PATH -Force

# ── PATH update (AC06) ─────────────────────────────────────────────────────

$CurrentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($CurrentPath -notlike "*$INSTALL_DIR*") {
    $NewPath = "$INSTALL_DIR;$CurrentPath"
    [Environment]::SetEnvironmentVariable("PATH", $NewPath, "User")
    Write-Host "[parallax] Added $INSTALL_DIR to your user PATH."
}

# ── success (AC07) ─────────────────────────────────────────────────────────

Write-Host "parallax $BareVersion installed. Open a new terminal and run: parallax install"
