"""ComfyUI workflow reader with subgraph expansion.

Provides utilities to load ComfyUI workflow JSON files and resolve all
subgraph nodes (UUID-typed nodes) into their constituent primitive nodes,
enabling downstream analysis of which ComfyUI node classes a workflow uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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


def _build_subgraph_index(workflow: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return a dict mapping subgraph UUID → list of its inner nodes."""
    subgraphs: list[dict[str, Any]] = (
        workflow.get("definitions", {}).get("subgraphs", []) or []
    )
    return {s["id"]: s.get("nodes", []) for s in subgraphs if "id" in s}


def _is_uuid(value: str) -> bool:
    """Return True if *value* looks like a UUID (36-char, 4 hyphens)."""
    return isinstance(value, str) and len(value) == 36 and value.count("-") == 4


def _collect_node_types(
    nodes: list[dict[str, Any]],
    index: dict[str, list[dict[str, Any]]],
    visited: set[str],
) -> set[str]:
    """Recursively collect all primitive node type strings from *nodes*.

    UUID-typed nodes are expanded via *index*; *visited* prevents infinite
    loops if subgraph definitions reference each other.
    """
    result: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type: str = node.get("type", "")
        if not node_type:
            continue
        if _is_uuid(node_type):
            if node_type in visited:
                continue
            inner = index.get(node_type)
            if inner is not None:
                visited = visited | {node_type}
                result |= _collect_node_types(inner, index, visited)
        else:
            result.add(node_type)
    return result


def get_node_types(workflow: dict[str, Any]) -> set[str]:
    """Return the set of all ComfyUI node class names used by *workflow*.

    Subgraph nodes (UUID-typed) are recursively expanded using their
    definitions stored in ``workflow["definitions"]["subgraphs"]``.
    Utility / display-only node types (``MarkdownNote``, ``Note``,
    ``Reroute``, ``PrimitiveInt``, ``PrimitiveFloat``, ``PrimitiveBoolean``,
    ``PrimitiveString``) are excluded from the result.

    Args:
        workflow: A workflow dict as returned by :func:`load_workflow`.

    Returns:
        Set of node type strings, e.g. ``{"CheckpointLoaderSimple",
        "CLIPTextEncode", "KSampler"}``.
    """
    _DISPLAY_NODES = {
        "MarkdownNote",
        "Note",
        "Reroute",
        "PrimitiveInt",
        "PrimitiveFloat",
        "PrimitiveBoolean",
        "PrimitiveString",
    }
    index = _build_subgraph_index(workflow)
    nodes: list[dict[str, Any]] = workflow.get("nodes", [])
    if not isinstance(nodes, list):
        return set()
    types = _collect_node_types(nodes, index, visited=set())
    return types - _DISPLAY_NODES


def get_subgraph_names(workflow: dict[str, Any]) -> list[str]:
    """Return human-readable names of all subgraph definitions in *workflow*.

    Useful for quickly understanding what high-level pipelines a workflow
    encapsulates without expanding them.

    Args:
        workflow: A workflow dict as returned by :func:`load_workflow`.

    Returns:
        List of subgraph name strings.
    """
    subgraphs: list[dict[str, Any]] = (
        workflow.get("definitions", {}).get("subgraphs", []) or []
    )
    return [s.get("name", s.get("id", "?")) for s in subgraphs]


def get_subgraph_io(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the inputs/outputs interface of every subgraph in *workflow*.

    Each item describes one subgraph and its exposed ports, which correspond
    to the parameters a pipeline would accept.

    Args:
        workflow: A workflow dict as returned by :func:`load_workflow`.

    Returns:
        List of dicts with keys ``name``, ``id``, ``inputs``, ``outputs``.
        ``inputs`` and ``outputs`` are lists of port dicts with at least
        ``name`` and ``type`` keys.
    """
    subgraphs: list[dict[str, Any]] = (
        workflow.get("definitions", {}).get("subgraphs", []) or []
    )
    result = []
    for s in subgraphs:
        result.append(
            {
                "name": s.get("name", s.get("id", "?")),
                "id": s.get("id", ""),
                "inputs": [
                    {"name": p.get("name", ""), "type": p.get("type", "")}
                    for p in s.get("inputs", [])
                ],
                "outputs": [
                    {"name": p.get("name", ""), "type": p.get("type", "")}
                    for p in s.get("outputs", [])
                ],
            }
        )
    return result


__all__ = [
    "load_workflow",
    "get_node_types",
    "get_subgraph_names",
    "get_subgraph_io",
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python workflow.py <workflow.json> [workflow.json ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        try:
            wf = load_workflow(path)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR {path}: {e}")
            continue

        names = get_subgraph_names(wf)
        types = get_node_types(wf)
        io = get_subgraph_io(wf)

        print(f"\n=== {path} ===")
        if names:
            print(f"Subgraphs ({len(names)}): {', '.join(names)}")
        print(f"Node types ({len(types)}):")
        for t in sorted(types):
            print(f"  {t}")
        if io:
            print("Subgraph I/O:")
            for s in io:
                inputs = ", ".join(f"{p['name']}:{p['type']}" for p in s["inputs"])
                outputs = ", ".join(f"{p['name']}:{p['type']}" for p in s["outputs"])
                print(f"  [{s['name']}]  in=({inputs})  out=({outputs})")
