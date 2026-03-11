# Skill: Runtime Check

## Goal

Collect runtime diagnostics before running any inference calls.

## Steps

1. Import `check_runtime` from `comfy_diffusion`.
2. Call `check_runtime()` and inspect the returned dictionary.
3. If the dictionary contains an `error` key, report and stop the flow.

