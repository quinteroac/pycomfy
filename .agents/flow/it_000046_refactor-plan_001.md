# Refactor Completion Report — it_000046 (pass 001)

## Summary of changes

### RF-1 — Add `s` to `_ltx23` explicit signature (`cli/_runners/video.py`)

Updated the `_ltx23` runner signature from:

```python
def _ltx23(*, mdir, prompt, image, w, h, n, f, c, seed, audio=None, **_):
```

to:

```python
def _ltx23(*, mdir, prompt, image, w, h, n, f, s, c, seed, audio=None, **_):
```

This cosmetic change aligns the explicit signature with the PRD FR-6 spec. The parameter `s` is still not forwarded to any ltx23 pipeline branch (t2v / i2v / ia2v), as none of them accept a `steps` argument. Its presence in the signature is purely for API consistency with other runners (`_wan21`, etc.) and for self-documenting purposes.

### RF-2 — Add unit tests for CLI audio guard validations (`tests/test_create_video_audio.py`)

Created a new focused, CPU-only test module covering the three audio guard validation scenarios:

- `TestAudioUnsupportedModelGuard` — `--audio` with a non-ltx23 model (`ltx2`, `wan21`, `wan22`) exits 1 with the expected error message.
- `TestAudioMissingFileGuard` — `--audio` pointing to a non-existent path exits 1 and includes the path in the error message.
- `TestAudioRequiresInputGuard` — `--audio` without `--input` exits 1 with the expected error message.

### Test updates (`tests/test_cli_us002_it000046.py`)

Added the now-required `s=20` keyword argument to all direct `_ltx23()` invocations in the pre-existing test file, since the signature change made `s` a required keyword-only parameter. No test logic was changed — only the call sites were updated to satisfy the new signature.

---

## Quality checks

| Check | Command | Outcome |
|-------|---------|---------|
| Targeted test suite | `uv run pytest tests/test_create_video_audio.py tests/test_cli_us001_it000046.py tests/test_cli_us002_it000046.py -v` | ✅ 36/36 passed |

All three test files pass cleanly with no warnings related to the refactor changes.

---

## Deviations from refactor plan

None.

Both RF-1 and RF-2 were applied exactly as specified in the audit JSON. The only additional change was updating the direct `_ltx23()` call sites in the pre-existing `test_cli_us002_it000046.py` to pass `s=20`, which was a necessary consequence of RF-1 making `s` an explicit required parameter — not a deviation from the plan.
