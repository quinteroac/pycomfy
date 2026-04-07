#!/bin/sh
# install.sh — Install the parallax CLI on Linux or macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/quinteroac/comfy-diffusion/master/install.sh | sh
# Env overrides:
#   PARALLAX_VERSION      — install a specific version tag (e.g. v1.2.0)
#   PARALLAX_INSTALL_DIR  — override install directory (default: ~/.local/bin)

set -e

REPO="quinteroac/comfy-diffusion"
INSTALL_DIR="${PARALLAX_INSTALL_DIR:-$HOME/.local/bin}"
BINARY_NAME="parallax"

# ── helpers ────────────────────────────────────────────────────────────────

info()  { printf '\033[0;32m[parallax]\033[0m %s\n' "$*"; }
warn()  { printf '\033[0;33m[parallax]\033[0m %s\n' "$*" >&2; }
error() { printf '\033[0;31m[parallax]\033[0m %s\n' "$*" >&2; exit 1; }

# ── OS / arch detection (AC01) ─────────────────────────────────────────────

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux)
    case "$ARCH" in
      x86_64) ASSET="parallax-linux-x86_64" ;;
      *) error "Unsupported architecture: $ARCH. Only x86_64 is supported on Linux." ;;
    esac
    ;;
  Darwin)
    ASSET="parallax-macos-arm64"
    ;;
  *)
    error "Unsupported OS: $OS. This installer supports Linux and macOS only."
    ;;
esac

# ── version resolution (AC02 / AC09) ──────────────────────────────────────

if [ -n "${PARALLAX_VERSION:-}" ]; then
  VERSION="$PARALLAX_VERSION"
  info "Using requested version: $VERSION"
else
  API_URL="https://api.github.com/repos/${REPO}/releases/latest"
  info "Fetching latest release from GitHub..."

  if command -v curl >/dev/null 2>&1; then
    API_RESPONSE="$(curl -fsSL "$API_URL" 2>/dev/null)" || API_RESPONSE=""
  elif command -v wget >/dev/null 2>&1; then
    API_RESPONSE="$(wget -qO- "$API_URL" 2>/dev/null)" || API_RESPONSE=""
  else
    error "Neither curl nor wget found. Please install one and retry."
  fi

  if [ -z "$API_RESPONSE" ]; then
    error "Could not fetch latest release. Set PARALLAX_VERSION=vX.X.X to install a specific version."
  fi

  # Extract tag_name from JSON — POSIX sed / awk, no jq required
  VERSION="$(printf '%s' "$API_RESPONSE" | \
    sed 's/,/,\n/g' | \
    grep '"tag_name"' | \
    sed 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')"

  if [ -z "$VERSION" ]; then
    error "Could not fetch latest release. Set PARALLAX_VERSION=vX.X.X to install a specific version."
  fi

  info "Latest version: $VERSION"
fi

# Strip leading 'v' to get bare semver (used in success message)
BARE_VERSION="${VERSION#v}"

# ── detect existing installation (AC08) ────────────────────────────────────

INSTALL_PATH="${INSTALL_DIR}/${BINARY_NAME}"

EXISTING_VERSION=""
if [ -x "$INSTALL_PATH" ]; then
  EXISTING_VERSION="$("$INSTALL_PATH" --version 2>/dev/null | awk '{print $NF}')" || EXISTING_VERSION=""
fi

if [ -n "$EXISTING_VERSION" ] && [ "$EXISTING_VERSION" = "$BARE_VERSION" ]; then
  info "parallax $BARE_VERSION is already installed. Nothing to do."
  exit 0
fi

if [ -n "$EXISTING_VERSION" ]; then
  info "Updating parallax from v${EXISTING_VERSION} to v${BARE_VERSION}."
fi

# ── download (AC03) ────────────────────────────────────────────────────────

BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"
BINARY_URL="${BASE_URL}/${ASSET}"
CHECKSUM_URL="${BASE_URL}/${ASSET}.sha256"

TMP_DIR="$(mktemp -d)"
TMP_BINARY="${TMP_DIR}/${ASSET}"
TMP_CHECKSUM="${TMP_DIR}/${ASSET}.sha256"

# Ensure temp dir is cleaned on exit
_cleanup() { rm -rf "$TMP_DIR"; }
trap _cleanup EXIT

info "Downloading $ASSET..."

if command -v curl >/dev/null 2>&1; then
  curl -fL --progress-bar -o "$TMP_BINARY"   "$BINARY_URL"   || error "Download failed: $BINARY_URL"
  curl -fsSL              -o "$TMP_CHECKSUM" "$CHECKSUM_URL" || error "Download failed: $CHECKSUM_URL"
elif command -v wget >/dev/null 2>&1; then
  wget --show-progress -q -O "$TMP_BINARY"   "$BINARY_URL"   || error "Download failed: $BINARY_URL"
  wget -q              -O "$TMP_CHECKSUM" "$CHECKSUM_URL"    || error "Download failed: $CHECKSUM_URL"
else
  error "Neither curl nor wget found. Please install one and retry."
fi

# ── checksum verification (AC04) ──────────────────────────────────────────

info "Verifying checksum..."

EXPECTED_HASH="$(awk '{print $1}' "$TMP_CHECKSUM")"

case "$OS" in
  Linux)
    ACTUAL_HASH="$(sha256sum "$TMP_BINARY" | awk '{print $1}')"
    ;;
  Darwin)
    ACTUAL_HASH="$(shasum -a 256 "$TMP_BINARY" | awk '{print $1}')"
    ;;
esac

if [ "$EXPECTED_HASH" != "$ACTUAL_HASH" ]; then
  rm -f "$TMP_BINARY"
  error "Checksum verification failed. Aborting."
fi

info "Checksum OK."

# ── install (AC05) ─────────────────────────────────────────────────────────

mkdir -p "$INSTALL_DIR"
cp "$TMP_BINARY" "$INSTALL_PATH"
chmod +x "$INSTALL_PATH"

# ── PATH guidance (AC06) ───────────────────────────────────────────────────

_on_path() {
  case ":${PATH}:" in
    *":${INSTALL_DIR}:"*) return 0 ;;
    *) return 1 ;;
  esac
}

if ! _on_path; then
  warn ""
  warn "$INSTALL_DIR is not on your PATH."
  warn "Add the following line to your shell profile:"
  warn ""
  warn "  export PATH=\"${INSTALL_DIR}:\$PATH\""
  warn ""
  warn "You can do this by running one of:"
  warn "  echo 'export PATH=\"${INSTALL_DIR}:\$PATH\"' >> ~/.bashrc"
  warn "  echo 'export PATH=\"${INSTALL_DIR}:\$PATH\"' >> ~/.zshrc"
  warn "  echo 'export PATH=\"${INSTALL_DIR}:\$PATH\"' >> ~/.profile"
  warn ""
  warn "Then open a new terminal for the change to take effect."
fi

# ── success (AC07) ─────────────────────────────────────────────────────────

info "parallax ${BARE_VERSION} installed. Run: parallax install"
