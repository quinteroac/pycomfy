---
name: workflow-reader
description: Read and analyze ComfyUI workflow JSON files. Expands subgraph nodes recursively and exposes every node with its type, parameters, connections, and execution state. Designed for pipeline analysis — finding which models are needed, tracing execution flow, and detecting bypassed nodes.
user-invocable: true
---

# workflow-reader — ComfyUI Workflow Analysis

**When you need to analyze a ComfyUI workflow, use the tool at `tools/workflow.py`.**

It expands subgraph nodes recursively so you always see primitive ComfyUI node types,
never opaque UUID placeholders.

---

## CLI Usage

```bash
# Single workflow — prints model downloads + all nodes in execution order
python comfy_diffusion/skills/workflow-reader/tools/workflow.py \
    comfyui_official_workflows/video/ltx/ltx23/video_ltx2_3_t2v.json

# Multiple files at once
python comfy_diffusion/skills/workflow-reader/tools/workflow.py \
    comfyui_official_workflows/video/ltx/ltx23/*.json
```

Output per file:
1. Subgraph names and their I/O ports
2. **Model downloads** — which files the workflow needs, grouped by directory, with their loader node and active/bypassed state
3. **All nodes** in execution order — type, params, input/output connections, active/bypassed state

---

## Python API

```python
import sys
sys.path.insert(0, "comfy_diffusion/skills/workflow-reader/tools")
from workflow import load_workflow, get_nodes, get_connections, get_model_downloads, get_subgraph_names, get_subgraph_io
```

---

### `load_workflow(path) → dict`

Load a workflow JSON file.

```python
wf = load_workflow("comfyui_official_workflows/video/ltx/ltx23/video_ltx2_3_t2v.json")
```

---

### `get_nodes(workflow, *, include_display=False, include_bypassed=True) → list[dict]`

Return every node after full subgraph expansion.  Each dict has:

| field | description |
|-------|-------------|
| `id` | node integer ID |
| `type` | ComfyUI class name (e.g. `"CheckpointLoaderSimple"`) |
| `mode` | `"active"`, `"bypassed"`, or `"muted"` |
| `order` | execution order (lower = earlier) |
| `params` | list of widget values — the node's configuration |
| `inputs` | `[{"name", "type", "link"}]` — `link` is the wire ID or `None` |
| `outputs` | `[{"name", "type", "links"}]` — `links` is a list of wire IDs |

Result is sorted by `order`.

```python
nodes = get_nodes(wf)
# Find all loader nodes and their filenames
for n in nodes:
    if "Loader" in n["type"] and n["mode"] == "active":
        print(n["type"], n["params"])

# Find bypassed nodes
bypassed = [n for n in get_nodes(wf, include_bypassed=True) if n["mode"] != "active"]
```

---

### `get_model_downloads(workflow) → list[dict]`

Return every model file the workflow declares as downloadable, deduplicated and
sorted by directory.  Each dict has:

| field | description |
|-------|-------------|
| `name` | filename (e.g. `"ltx-2.3-22b-dev-fp8.safetensors"`) |
| `url` | direct HuggingFace/download URL |
| `directory` | ComfyUI models subdirectory (e.g. `"checkpoints"`, `"loras"`) |
| `node_type` | the loader node that declared this file |
| `node_mode` | `"active"`, `"bypassed"`, or `"muted"` |

Use this to compare a workflow's required files against a pipeline's `manifest()`.

```python
downloads = get_model_downloads(wf)
# Only files from active nodes
required = [m for m in downloads if m["node_mode"] == "active"]
for m in required:
    print(f"{m['directory']}/{m['name']}")
```

---

### `get_connections(workflow) → list[dict]`

Return every wire connection between nodes.  Each dict has:

| field | description |
|-------|-------------|
| `link_id` | integer wire ID |
| `from_node` | source node ID |
| `from_slot` | source output slot index |
| `to_node` | destination node ID |
| `to_slot` | destination input slot index |
| `type` | data type on the wire (e.g. `"MODEL"`, `"LATENT"`) |

```python
connections = get_connections(wf)
# Trace what feeds into node 215
feeds_215 = [c for c in connections if c["to_node"] == 215]
```

---

### `get_subgraph_names(workflow) → list[str]`

Human-readable names of all subgraphs defined in the workflow.

---

### `get_subgraph_io(workflow) → list[dict]`

Input/output port definitions for each subgraph — these become the pipeline parameters.

---

## Typical Analysis Patterns

### Compare workflow models vs pipeline manifest

```python
downloads = get_model_downloads(wf)
active_files = {m["name"] for m in downloads if m["node_mode"] == "active"}

from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import manifest
manifest_files = {Path(e.dest).name for e in manifest()}

missing_from_manifest = active_files - manifest_files
extra_in_manifest     = manifest_files - active_files
```

### Find the two-pass sampling chain

```python
nodes = get_nodes(wf, include_bypassed=False)
samplers = [n for n in nodes if n["type"] == "SamplerCustomAdvanced"]
sigmas   = [n for n in nodes if n["type"] == "ManualSigmas"]
print(f"{len(samplers)} sampling passes, sigmas: {[s['params'] for s in sigmas]}")
```

### List all active LoRA nodes and their strengths

```python
loras = [n for n in get_nodes(wf) if "Lora" in n["type"] and n["mode"] == "active"]
for lora in loras:
    print(lora["type"], lora["params"])  # [filename, strength_model, strength_clip]
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
- `links` — inner wire list
