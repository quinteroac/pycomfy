---
name: workflow-reader
description: Read and analyze ComfyUI workflow JSON files, expanding subgraph nodes recursively to reveal all primitive node types used — enables pipeline feasibility analysis.
user-invocable: false
---

# workflow-reader — ComfyUI Workflow Analysis

**When you need to decode a ComfyUI workflow (inspect its nodes, expand subgraphs, or check
pipeline feasibility), use the tool at `tools/workflow.py`** — either run it from the CLI or
import its functions directly in a Python script.

The tool is pure Python and requires no ComfyUI runtime.

---

## CLI Usage

Run against one or more workflow JSON files to print a full node-type report:

```bash
python comfy_diffusion/skills/workflow-reader/tools/workflow.py \
    comfyui_official_workflows/video/ltx/ltx2/video_ltx2_t2v.json

# Multiple files at once
python comfy_diffusion/skills/workflow-reader/tools/workflow.py \
    comfyui_official_workflows/video/ltx/ltx2/*.json
```

Output per file:
- Subgraph names (if any)
- All node types after subgraph expansion (sorted)
- Subgraph I/O ports (inputs/outputs that map to pipeline parameters)

---

## Python API

Import the four functions directly from the tool file:

```python
import sys
sys.path.insert(0, "comfy_diffusion/skills/workflow-reader/tools")
from workflow import load_workflow, get_node_types, get_subgraph_names, get_subgraph_io
```

### `load_workflow(path) → dict`

Load a workflow JSON file. Raises `FileNotFoundError` or `ValueError` on bad input.

```python
wf = load_workflow("comfyui_official_workflows/video/ltx/ltx2/video_ltx2_t2v.json")
```

### `get_node_types(workflow) → set[str]`

Return all ComfyUI node class names used, **subgraphs fully expanded recursively**.
Display-only nodes are excluded (`MarkdownNote`, `Note`, `Reroute`, `Primitive*`).

```python
types = get_node_types(wf)
# {'CheckpointLoaderSimple', 'CLIPTextEncode', 'LTXVConditioning', ...}
```

### `get_subgraph_names(workflow) → list[str]`

Human-readable names of all subgraphs defined in the workflow.

```python
get_subgraph_names(wf)
# ['Text to Video (LTX 2.0)']
```

### `get_subgraph_io(workflow) → list[dict]`

Input/output port definitions for each subgraph — these become the pipeline parameters.

```python
io = get_subgraph_io(wf)
# [{'name': 'Text to Video (LTX 2.0)', 'id': '...', 'inputs': [...], 'outputs': [...]}]
```

---

## Pipeline Feasibility Check

Use this pattern to decide whether a workflow is directly convertible to a Python pipeline:

```python
import sys
sys.path.insert(0, "comfy_diffusion/skills/workflow-reader/tools")
from workflow import load_workflow, get_node_types

SUPPORTED = {
    "CheckpointLoaderSimple", "CLIPLoader", "CLIPTextEncode",
    "LTXVConditioning", "LTXVImgToVideo", "EmptyLTXVLatentVideo",
    "LTXVScheduler", "KSamplerSelect", "SamplerCustom",
    "SamplerCustomAdvanced", "CFGGuider", "RandomNoise",
    "ManualSigmas", "VAEDecode", "VAEDecodeTiled",
    "LTXVCropGuides", "LTXVEmptyLatentAudio", "LTXVConcatAVLatent",
    "LTXVSeparateAVLatent", "LTXVAudioVAEDecode", "LTXVLatentUpsampler",
    "LoraLoaderModelOnly", "LoadImage", "LoadVideo", "LoadAudio",
    "SaveVideo", "CreateVideo", "GetImageSize", "ImageScaleBy",
    "ImageFromBatch", "GetVideoComponents",
}

def analyze(path):
    wf = load_workflow(path)
    types = get_node_types(wf)
    missing = types - SUPPORTED
    return {
        "path": path,
        "convertible": len(missing) == 0,
        "missing": sorted(missing),
    }
```

---

## Subgraph Structure Reference

```
workflow["definitions"]["subgraphs"]  →  list of subgraph objects
```

Each subgraph:
- `id` — UUID matching the `type` field of the subgraph node in `workflow["nodes"]`
- `name` — human-readable label
- `inputs` / `outputs` — port definitions (name, type, linkIds)
- `nodes` — inner nodes (same format as top-level; may reference other UUIDs → resolved recursively with cycle detection)
