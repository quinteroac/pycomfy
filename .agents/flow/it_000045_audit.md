# Audit Report — it_000045 / PRD-003
# Installer Scripts: install.sh (Linux/macOS) + install.ps1 (Windows)

## Executive Summary

The implementation for iteration 000045 (PRD index 003) is largely compliant with the product requirement document. Both install scripts (`install.sh` and `install.ps1`) correctly implement all user-story acceptance criteria for US-001 and US-002. Two functional requirements have partial gaps: FR-4 is only satisfied by `install.sh` — `install.ps1` does not honour the `$env:PARALLAX_INSTALL_DIR` override. FR-7 is partially satisfied — the README documents the Linux/macOS one-liner but omits the Windows PowerShell install command. No blocking issues were found.

---

## Verification by FR

| FR | Assessment | Notes |
|----|-----------|-------|
| FR-1 | ✅ comply | `install.sh` uses `#!/bin/sh` with no bash-isms (`[[`, `$BASH_VERSION`, `source` absent). POSIX-compatible constructs throughout. |
| FR-2 | ✅ comply | `install.ps1` uses only built-in PowerShell 5.1+ cmdlets; no external tools required. |
| FR-3 | ✅ comply | `REPO="quinteroac/comfy-diffusion"` defined at top of `install.sh` (line 10); `$REPO = "quinteroac/comfy-diffusion"` at top of `install.ps1` (line 8). |
| FR-4 | ⚠️ partially_comply | `install.sh` honours `PARALLAX_INSTALL_DIR` (line 11). `install.ps1` hardcodes the install dir (lines 10–11) without checking `$env:PARALLAX_INSTALL_DIR`. |
| FR-5 | ✅ comply | `install.sh` writes to `~/.local/bin`; `install.ps1` writes to `$env:APPDATA\parallax\bin` and updates User-scope PATH only. No elevated privileges required. |
| FR-6 | ✅ comply | Both scripts query `https://api.github.com/repos/{REPO}/releases/latest` and extract `tag_name`. |
| FR-7 | ⚠️ partially_comply | README shows the Linux/macOS curl one-liner (lines 39–51) but is missing the Windows PowerShell `irm … | iex` command. |

---

## Verification by US

| US | Assessment | Notes |
|----|-----------|-------|
| US-001 | ✅ comply | All 9 AC verified: OS/arch detection (AC01), version resolution + env override (AC02), curl/wget download with progress (AC03), sha256 checksum with abort (AC04), install to `~/.local/bin` with `chmod +x` (AC05), PATH guidance for `.bashrc`/`.zshrc`/`.profile` (AC06), success message (AC07), update detection message (AC08), API failure message (AC09). |
| US-002 | ✅ comply | All 7 AC verified: always downloads x86_64 exe (AC01), GitHub API with `$env:PARALLAX_VERSION` override (AC02), `Invoke-WebRequest` with progress (AC03), `Get-FileHash` SHA256 with abort (AC04), install to `$env:APPDATA\parallax\bin\parallax.exe` (AC05), User-scope PATH update (AC06), success message (AC07). |

---

## Minor Observations

1. `install.ps1` line 25: the API-failure error branch prints without the `[parallax]` prefix, inconsistent with all other `Write-Host` calls.
2. `install.sh` line 163: the PATH export hint hardcodes `$HOME/.local/bin` even when `PARALLAX_INSTALL_DIR` is set to a custom path. The hint should reference `$INSTALL_DIR`.
3. `install.ps1` has no `try/catch` around `Invoke-WebRequest` — a network error would produce a raw PowerShell exception instead of a friendly message.
4. README `Install` section is missing the Windows one-liner, which is needed to complete FR-7 and improve discoverability.

---

## Conclusions and Recommendations

The implementation is functionally solid. All user-story acceptance criteria pass. Two targeted fixes are recommended:

1. **Fix FR-4 in `install.ps1`** — add `$env:PARALLAX_INSTALL_DIR` support.
2. **Fix FR-7 in `README.md`** — add a Windows install section with the PowerShell one-liner.

Low-priority polish items: fix missing `[parallax]` prefix in `install.ps1` error branch, update the PATH hint in `install.sh` to use `$INSTALL_DIR`, and add a `try/catch` around downloads in `install.ps1`.

---

## Refactor Plan

### Priority 1 — FR-4: Add `$env:PARALLAX_INSTALL_DIR` to `install.ps1`

**File:** `install.ps1`  
**Change:** Replace the hardcoded `$INSTALL_DIR` assignment with an env-aware one:

```powershell
# Before (line 10):
$INSTALL_DIR  = Join-Path $env:APPDATA "parallax\bin"

# After:
$INSTALL_DIR  = if ($env:PARALLAX_INSTALL_DIR) { $env:PARALLAX_INSTALL_DIR } else { Join-Path $env:APPDATA "parallax\bin" }
```

`$INSTALL_PATH = Join-Path $INSTALL_DIR "parallax.exe"` remains unchanged (already derived from `$INSTALL_DIR`).

---

### Priority 2 — FR-7: Add Windows install command to `README.md`

**File:** `README.md`  
**Change:** After the Linux/macOS install block (after line 51), add:

```markdown
**Windows (PowerShell, one command):**
```powershell
irm https://raw.githubusercontent.com/quinteroac/comfy-diffusion/master/install.ps1 | iex
```

Then open a new terminal and run:
```powershell
parallax install
```
```

---

### Priority 3 — Minor polish (low priority)

| File | Location | Fix |
|------|----------|-----|
| `install.ps1` | Line 25 | Add `[parallax]` prefix to the API-failure `Write-Host` message |
| `install.sh` | Line 163 | Replace hardcoded `$HOME/.local/bin` in PATH hint with `$INSTALL_DIR` |
| `install.ps1` | Lines 48–49 | Wrap `Invoke-WebRequest` calls in `try/catch` for friendly network-error messages |
