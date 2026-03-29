"""ComfyUI workflow reader — full graph inspection with subgraph expansion.

Loads a ComfyUI workflow JSON and exposes every node, its parameters
(widget values), its input/output connections, and its execution state
(active vs. bypassed).  Subgraph nodes are expanded recursively so the
result always contains only primitive ComfyUI node types.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Node mode constants (ComfyUI internal values)
# ---------------------------------------------------------------------------

MODE_ACTIVE   = 0
MODE_MUTED    = 2
MODE_BYPASSED = 4

MODE_LABELS = {
    MODE_ACTIVE:   "active",
    MODE_MUTED:    "muted",
    MODE_BYPASSED: "bypassed",
}

# Node types that are UI-only and carry no computational meaning.
DISPLAY_NODE_TYPES = {
    "MarkdownNote",
    "Note",
    "Reroute",
    "PrimitiveInt",
    "PrimitiveFloat",
    "PrimitiveBoolean",
    "PrimitiveString",
    "PrimitiveStringMultiline",
    "PreviewAny",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_uuid(value: str) -> bool:
    return isinstance(value, str) and len(value) == 36 and value.count("-") == 4


def _build_subgraph_index(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return {uuid: subgraph_dict} for every subgraph defined in the workflow."""
    subgraphs = workflow.get("definitions", {}).get("subgraphs", []) or []
    return {s["id"]: s for s in subgraphs if "id" in s}


def _build_link_index(
    nodes: list[dict[str, Any]],
    links_list: list[Any],
) -> dict[int, dict[str, Any]]:
    """Return {link_id: {from_node, from_slot, to_node, to_slot}} for the given node set.

    Handles both formats:
    - Top-level links: ``[link_id, from_node_id, from_slot, to_node_id, to_slot, type]``
    - Subgraph links: ``{"id", "origin_id", "origin_slot", "target_id", "target_slot", "type"}``
    """
    index: dict[int, dict[str, Any]] = {}
    for link in links_list:
        if isinstance(link, list) and len(link) >= 5:
            index[link[0]] = {
                "from_node": link[1],
                "from_slot": link[2],
                "to_node":   link[3],
                "to_slot":   link[4],
                "type":      link[5] if len(link) > 5 else "",
            }
        elif isinstance(link, dict):
            link_id = link.get("id")
            if link_id is not None:
                index[link_id] = {
                    "from_node": link.get("origin_id"),
                    "from_slot": link.get("origin_slot"),
                    "to_node":   link.get("target_id"),
                    "to_slot":   link.get("target_slot"),
                    "type":      link.get("type", ""),
                }
    return index


def _expand_nodes(
    nodes: list[dict[str, Any]],
    sg_index: dict[str, dict[str, Any]],
    links_index: dict[int, dict[str, Any]],
    visited: set[str],
) -> list[dict[str, Any]]:
    """Recursively expand subgraph nodes; return a flat list of primitive nodes."""
    result: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type: str = node.get("type", "")
        if not node_type:
            continue
        if _is_uuid(node_type):
            if node_type in visited:
                continue
            sg = sg_index.get(node_type)
            if sg is None:
                continue
            inner_nodes = sg.get("nodes", [])
            inner_links = sg.get("links", [])
            inner_link_idx = _build_link_index(inner_nodes, inner_links)
            result.extend(
                _expand_nodes(inner_nodes, sg_index, inner_link_idx, visited | {node_type})
            )
        else:
            result.append(node)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_workflow(path: str | Path) -> dict[str, Any]:
    """Load a ComfyUI workflow JSON file.

    Args:
        path: Path to a ``.json`` workflow file.

    Returns:
        The raw workflow dict as parsed from JSON.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file is not valid JSON or not a workflow object.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Workflow file not found: {p}")
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}: {p}")
    return data


def get_nodes(
    workflow: dict[str, Any],
    *,
    include_display: bool = False,
    include_bypassed: bool = True,
) -> list[dict[str, Any]]:
    """Return every node in the workflow as a structured dict.

    Subgraph nodes are recursively expanded.  Each returned dict has:

    - ``id``        — node integer ID
    - ``type``      — ComfyUI node class name (e.g. ``"CheckpointLoaderSimple"``)
    - ``mode``      — ``"active"``, ``"bypassed"``, or ``"muted"``
    - ``order``     — execution order index (lower = earlier)
    - ``params``    — list of widget values (the node's configuration parameters)
    - ``inputs``    — list of ``{"name": str, "type": str, "link": int | None}``
    - ``outputs``   — list of ``{"name": str, "type": str, "links": list[int]}``

    Args:
        workflow: A workflow dict as returned by :func:`load_workflow`.
        include_display: When ``False`` (default), UI-only nodes such as
            ``MarkdownNote``, ``Note``, and ``Reroute`` are excluded.
        include_bypassed: When ``False``, bypassed and muted nodes are excluded.

    Returns:
        List of node dicts sorted by execution ``order``.
    """
    sg_index = _build_subgraph_index(workflow)
    top_links = _build_link_index(
        workflow.get("nodes", []),
        workflow.get("links", []) or [],
    )
    raw_nodes = _expand_nodes(
        workflow.get("nodes", []) or [],
        sg_index,
        top_links,
        visited=set(),
    )

    result: list[dict[str, Any]] = []
    for node in raw_nodes:
        node_type: str = node.get("type", "")
        if not node_type:
            continue
        if not include_display and node_type in DISPLAY_NODE_TYPES:
            continue

        mode_int: int = node.get("mode", MODE_ACTIVE)
        mode_label: str = MODE_LABELS.get(mode_int, f"mode={mode_int}")
        if not include_bypassed and mode_int in (MODE_MUTED, MODE_BYPASSED):
            continue

        inputs = [
            {
                "name": inp.get("name", ""),
                "type": inp.get("type", ""),
                "link": inp.get("link"),
            }
            for inp in (node.get("inputs") or [])
        ]
        outputs = [
            {
                "name": out.get("name", ""),
                "type": out.get("type", ""),
                "links": out.get("links") or [],
            }
            for out in (node.get("outputs") or [])
        ]

        result.append({
            "id":     node.get("id"),
            "type":   node_type,
            "mode":   mode_label,
            "order":  node.get("order", 9999),
            "params": node.get("widgets_values") or [],
            "inputs": inputs,
            "outputs": outputs,
        })

    result.sort(key=lambda n: (n["order"], n["id"] or 0))
    return result


def get_model_downloads(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every model file that the workflow declares as downloadable.

    ComfyUI stores download hints in ``node.properties.models`` — a list of
    ``{"name": str, "url": str, "directory": str}`` dicts attached to loader
    nodes.  This function collects them all across top-level nodes and
    expanded subgraphs, deduplicating by ``name``.

    Each returned dict has:

    - ``name``        — filename (e.g. ``"ltx-2.3-22b-dev-fp8.safetensors"``)
    - ``url``         — direct download URL
    - ``directory``   — ComfyUI models subdirectory (e.g. ``"checkpoints"``)
    - ``node_type``   — the loader node that declared this file
    - ``node_mode``   — ``"active"``, ``"bypassed"``, or ``"muted"``

    Args:
        workflow: A workflow dict as returned by :func:`load_workflow`.

    Returns:
        List of model dicts, deduplicated by ``name``, sorted by ``directory``
        then ``name``.
    """
    sg_index = _build_subgraph_index(workflow)
    top_links = _build_link_index(
        workflow.get("nodes", []),
        workflow.get("links", []) or [],
    )
    raw_nodes = _expand_nodes(
        workflow.get("nodes", []) or [],
        sg_index,
        top_links,
        visited=set(),
    )

    seen: dict[str, dict[str, Any]] = {}
    for node in raw_nodes:
        node_type = node.get("type", "")
        mode_int  = node.get("mode", MODE_ACTIVE)
        mode_label = MODE_LABELS.get(mode_int, f"mode={mode_int}")
        models = node.get("properties", {}).get("models") or []
        for m in models:
            name = m.get("name", "")
            if not name or name in seen:
                continue
            seen[name] = {
                "name":      name,
                "url":       m.get("url", ""),
                "directory": m.get("directory", ""),
                "node_type": node_type,
                "node_mode": mode_label,
            }

    return sorted(seen.values(), key=lambda m: (m["directory"], m["name"]))


def get_connections(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every wire connection between nodes as a list of dicts.

    Each dict has:

    - ``link_id``    — the link's integer ID
    - ``from_node``  — source node ID
    - ``from_slot``  — source output slot index
    - ``to_node``    — destination node ID
    - ``to_slot``    — destination input slot index
    - ``type``       — data type flowing through the wire (e.g. ``"MODEL"``)

    Connections inside subgraphs are included after expansion.

    Args:
        workflow: A workflow dict as returned by :func:`load_workflow`.

    Returns:
        List of connection dicts sorted by link_id.
    """
    sg_index = _build_subgraph_index(workflow)

    all_links: list[Any] = list(workflow.get("links", []) or [])
    # Also collect links from expanded subgraphs
    for sg in sg_index.values():
        all_links.extend(sg.get("links", []) or [])

    result: list[dict[str, Any]] = []
    for link in all_links:
        if isinstance(link, list) and len(link) >= 5:
            result.append({
                "link_id":   link[0],
                "from_node": link[1],
                "from_slot": link[2],
                "to_node":   link[3],
                "to_slot":   link[4],
                "type":      link[5] if len(link) > 5 else "",
            })
        elif isinstance(link, dict) and "id" in link:
            result.append({
                "link_id":   link["id"],
                "from_node": link.get("origin_id"),
                "from_slot": link.get("origin_slot"),
                "to_node":   link.get("target_id"),
                "to_slot":   link.get("target_slot"),
                "type":      link.get("type", ""),
            })

    result.sort(key=lambda c: c["link_id"])
    return result


def get_subgraph_names(workflow: dict[str, Any]) -> list[str]:
    """Return human-readable names of all subgraph definitions in *workflow*.

    Args:
        workflow: A workflow dict as returned by :func:`load_workflow`.

    Returns:
        List of subgraph name strings.
    """
    subgraphs = workflow.get("definitions", {}).get("subgraphs", []) or []
    return [s.get("name", s.get("id", "?")) for s in subgraphs]


def get_subgraph_io(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the inputs/outputs interface of every subgraph in *workflow*.

    Each item describes one subgraph and its exposed ports, which correspond
    to the parameters a pipeline would accept.

    Args:
        workflow: A workflow dict as returned by :func:`load_workflow`.

    Returns:
        List of dicts with keys ``name``, ``id``, ``inputs``, ``outputs``.
    """
    subgraphs = workflow.get("definitions", {}).get("subgraphs", []) or []
    result = []
    for s in subgraphs:
        result.append({
            "name": s.get("name", s.get("id", "?")),
            "id":   s.get("id", ""),
            "inputs": [
                {"name": p.get("name", ""), "type": p.get("type", "")}
                for p in s.get("inputs", [])
            ],
            "outputs": [
                {"name": p.get("name", ""), "type": p.get("type", "")}
                for p in s.get("outputs", [])
            ],
        })
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_workflow(path: str) -> None:
    try:
        wf = load_workflow(path)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR {path}: {e}")
        return

    print(f"\n{'='*60}")
    print(f"  {path}")
    print(f"{'='*60}")

    # Subgraph names and I/O
    names = get_subgraph_names(wf)
    if names:
        print(f"Subgraphs: {', '.join(names)}")
    io = get_subgraph_io(wf)
    if io:
        for s in io:
            ins  = ", ".join(f"{p['name']}:{p['type']}" for p in s["inputs"])
            outs = ", ".join(f"{p['name']}:{p['type']}" for p in s["outputs"])
            print(f"  [{s['name']}]  in=({ins})  out=({outs})")
        print()

    # Model downloads
    downloads = get_model_downloads(wf)
    if downloads:
        print(f"Model downloads ({len(downloads)}):")
        prev_dir = None
        for m in downloads:
            if m["directory"] != prev_dir:
                print(f"  [{m['directory']}/]")
                prev_dir = m["directory"]
            mode_tag = f"  [{m['node_mode'].upper()}]" if m["node_mode"] != "active" else ""
            print(f"    {m['name']}{mode_tag}  ← {m['node_type']}")
        print()

    # All nodes in execution order
    nodes = get_nodes(wf, include_display=False, include_bypassed=True)
    print(f"Nodes ({len(nodes)}) — execution order:\n")
    for node in nodes:
        status = "" if node["mode"] == "active" else f"  [{node['mode'].upper()}]"
        print(f"  #{node['id']:>4}  order={node['order']:>3}  {node['type']}{status}")
        if node["params"]:
            print(f"             params : {node['params']}")
        for inp in node["inputs"]:
            linked = f"← link {inp['link']}" if inp["link"] is not None else "← (unconnected)"
            print(f"             in  {inp['name']}:{inp['type']}  {linked}")
        for out in node["outputs"]:
            linked = f"→ links {out['links']}" if out["links"] else "→ (unused)"
            print(f"             out {out['name']}:{out['type']}  {linked}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python workflow.py <workflow.json> [workflow.json ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        _print_workflow(path)
