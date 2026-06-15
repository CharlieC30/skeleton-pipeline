"""Step 5: analyze the skeleton graph (trunk, branch points, lengths)."""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from pathlib import Path

import numpy as np
import tifffile
from natsort import natsorted

STEP_NAME = "05_analyze"
STEP_DESCRIPTION = "Analyze"

# Visualization colors (R, G, B)
COLOR_TRUNK = (255, 255, 255)  # white - main trunk
COLOR_BRANCH_POINT = (255, 255, 0)  # yellow - branch point markers
COLOR_MAX_PATH = (0, 255, 0)  # bright green - max-length path
BRANCH_COLORS = [(0, 255, 255)]  # cyan - other branch nodes


def parse_swc(
    swc_path: str,
) -> tuple[dict[int, dict], dict[int, list[tuple[int, float]]]]:
    """Parse SWC file and build graph structure.

    Args:
        swc_path: Path to the SWC file.

    Returns:
        nodes: Dict mapping node ID to {z, y, x, radius, parent}.
        adj: Adjacency list mapping node ID to [(neighbor_id, distance), ...].
    """
    nodes = {}
    adj = {}

    with open(swc_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            node_id = int(parts[0])
            # Kimimaro SWC format: ID TYPE Z Y X RADIUS PARENT
            nodes[node_id] = {
                "z": float(parts[2]),
                "y": float(parts[3]),
                "x": float(parts[4]),
                "radius": float(parts[5]),
                "parent": int(parts[6]),
            }
            adj[node_id] = []

    # Build adjacency list with edge weights
    for node_id, node in nodes.items():
        parent_id = node["parent"]
        if parent_id != -1 and parent_id in nodes:
            parent = nodes[parent_id]
            dist = np.sqrt(
                (node["x"] - parent["x"]) ** 2
                + (node["y"] - parent["y"]) ** 2
                + (node["z"] - parent["z"]) ** 2
            )
            adj[node_id].append((parent_id, dist))
            adj[parent_id].append((node_id, dist))

    return nodes, adj


def _bfs_farthest(
    start: int, adj: dict, exclude: set[int] | None = None
) -> tuple[int, float, dict[int, int]]:
    """Find the farthest node from start using BFS (breadth-first search).

    Args:
        start: Starting node ID.
        adj: Adjacency list mapping node ID to [(neighbor_id, distance), ...].
        exclude: Set of node IDs to skip during search.

    Returns:
        Tuple of (farthest_node_id, max_distance, parent_map).
        parent_map is used to trace back the path.
    """
    if exclude is None:
        exclude = set()

    dist = {start: 0.0}
    parent_map = {start: -1}
    queue = deque([start])
    farthest_node = start
    max_dist = 0.0

    while queue:
        curr = queue.popleft()
        for neighbor, edge_dist in adj.get(curr, []):
            if neighbor in exclude or neighbor in dist:
                continue
            new_dist = dist[curr] + edge_dist
            dist[neighbor] = new_dist
            parent_map[neighbor] = curr
            queue.append(neighbor)

            if new_dist > max_dist:
                max_dist = new_dist
                farthest_node = neighbor

    return farthest_node, max_dist, parent_map


def find_connected_components(nodes: dict, adj: dict) -> list[list[int]]:
    """Find all connected components in the skeleton graph.

    Args:
        nodes: Dict mapping node ID to node data.
        adj: Adjacency list mapping node ID to [(neighbor_id, distance), ...].

    Returns:
        List of components, where each component is a list of node IDs.
    """
    visited = set()
    components = []

    for node_id in nodes:
        if node_id in visited:
            continue

        # BFS to find all nodes in this component
        component = []
        queue = deque([node_id])

        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            component.append(curr)

            for neighbor, _ in adj.get(curr, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        components.append(component)

    return components


def find_main_trunk(nodes: dict, adj: dict) -> tuple[list[int], float]:
    """Find main trunk using tree diameter algorithm (two BFS).

    The trunk is found in the LARGEST connected component to handle
    skeletons with multiple disconnected parts.

    Args:
        nodes: Dict mapping node ID to node data.
        adj: Adjacency list mapping node ID to [(neighbor_id, distance), ...].

    Returns:
        Tuple of (trunk_path, trunk_length) where trunk_path is list of node IDs.
    """
    if not nodes:
        return [], 0.0

    # Find all connected components and use the largest one
    components = find_connected_components(nodes, adj)
    largest_component = max(components, key=len)
    start = largest_component[0]
    endpoint_a, _, _ = _bfs_farthest(start, adj)
    endpoint_b, trunk_length, parent_map = _bfs_farthest(endpoint_a, adj)

    trunk_path = []
    curr = endpoint_b
    while curr != -1:
        trunk_path.append(curr)
        curr = parent_map.get(curr, -1)
    trunk_path.reverse()

    return trunk_path, trunk_length


def find_branch_points(trunk_path: list[int], adj: dict) -> list[int]:
    """Find branch points on the main trunk.

    A branch point is a node with degree >= 3 (connected to trunk neighbors
    plus at least one branch).

    Args:
        trunk_path: List of node IDs in the main trunk.
        adj: Adjacency list mapping node ID to [(neighbor_id, distance), ...].

    Returns:
        List of node IDs that are branch points.
    """
    branch_points = []
    for node_id in trunk_path:
        degree = len(adj.get(node_id, []))
        if degree >= 3:
            branch_points.append(node_id)
    return branch_points


def calculate_branch_max_length(
    branch_point: int, trunk_set: set[int], adj: dict
) -> float:
    """Calculate maximum branch length from a branch point.

    Finds all branches extending from the branch point and returns the
    length of the longest one (measured to the farthest leaf node).

    Args:
        branch_point: Node ID of the branch point on the trunk.
        trunk_set: Set of node IDs in the main trunk.
        adj: Adjacency list mapping node ID to [(neighbor_id, distance), ...].

    Returns:
        Maximum branch length in voxels. Returns 0.0 if no branches exist.
    """
    branch_starts = []
    for neighbor, dist in adj.get(branch_point, []):
        if neighbor not in trunk_set:
            branch_starts.append((neighbor, dist))

    if not branch_starts:
        return 0.0

    max_length = 0.0
    for start_node, start_dist in branch_starts:
        _, subtree_max, _ = _bfs_farthest(start_node, adj, exclude=trunk_set)
        total_length = start_dist + subtree_max
        if total_length > max_length:
            max_length = total_length

    return max_length


def find_max_length_path(branch_point: int, trunk_set: set[int], adj: dict) -> set[int]:
    """Find nodes in the max-length path from a branch point.

    Args:
        branch_point: Node ID of the branch point.
        trunk_set: Set of node IDs in the main trunk.
        adj: Adjacency list.

    Returns:
        Set of node IDs in the max-length path (excluding branch_point itself).
    """
    branch_starts = []
    for neighbor, dist in adj.get(branch_point, []):
        if neighbor not in trunk_set:
            branch_starts.append((neighbor, dist))

    if not branch_starts:
        return set()

    max_length = 0.0
    best_path = set()

    for start_node, start_dist in branch_starts:
        farthest, subtree_max, parent_map = _bfs_farthest(
            start_node, adj, exclude=trunk_set
        )
        total_length = start_dist + subtree_max

        if total_length > max_length:
            max_length = total_length
            path = set()
            curr = farthest
            while curr != -1:
                path.add(curr)
                curr = parent_map.get(curr, -1)
            best_path = path

    return best_path


def calculate_position_on_trunk(
    branch_point: int, trunk_path: list[int], adj: dict
) -> float:
    """Calculate position of branch point on trunk as distance from start.

    Args:
        branch_point: Node ID of the branch point.
        trunk_path: Ordered list of node IDs in the main trunk.
        adj: Adjacency list mapping node ID to [(neighbor_id, distance), ...].

    Returns:
        Distance from trunk start to the branch point in voxels.
    """
    position = 0.0
    for i, node_id in enumerate(trunk_path):
        if node_id == branch_point:
            break
        if i < len(trunk_path) - 1:
            next_node = trunk_path[i + 1]
            for neighbor, dist in adj.get(node_id, []):
                if neighbor == next_node:
                    position += dist
                    break
    return position


def calculate_total_length(adj: dict) -> float:
    """Calculate total skeleton length by summing all edge distances.

    Args:
        adj: Adjacency list mapping node ID to [(neighbor_id, distance), ...].

    Returns:
        Total skeleton length in voxels.
    """
    total = 0.0
    for neighbors in adj.values():
        for _, dist in neighbors:
            total += dist
    return total / 2


def analyze_skeleton(swc_path: str) -> dict:
    """Analyze skeleton structure from SWC file.

    Parses the SWC file and computes main trunk, branch points, and
    branch lengths.

    Args:
        swc_path: Path to the SWC file.

    Returns:
        Dict with keys: coordinate_order, summary, main_trunk, branches.
    """
    nodes, adj = parse_swc(swc_path)

    if not nodes:
        return {
            "coordinate_order": "ZYX",
            "summary": {
                "total_nodes": 0,
                "total_length": 0.0,
                "main_trunk_length": 0.0,
                "num_branch_points": 0,
            },
            "main_trunk": {"start": [], "end": [], "length": 0.0},
            "branches": [],
        }

    trunk_path, trunk_length = find_main_trunk(nodes, adj)
    trunk_set = set(trunk_path)
    branch_point_ids = find_branch_points(trunk_path, adj)
    total_length = calculate_total_length(adj)

    start_node = nodes[trunk_path[0]] if trunk_path else {}
    end_node = nodes[trunk_path[-1]] if trunk_path else {}
    start_coords = [
        start_node.get("z", 0),
        start_node.get("y", 0),
        start_node.get("x", 0),
    ]
    end_coords = [end_node.get("z", 0), end_node.get("y", 0), end_node.get("x", 0)]

    branches = []
    for idx, bp_id in enumerate(branch_point_ids, start=1):
        bp_node = nodes[bp_id]
        position = calculate_position_on_trunk(bp_id, trunk_path, adj)
        max_length = calculate_branch_max_length(bp_id, trunk_set, adj)

        branches.append(
            {
                "id": idx,
                "branch_point": [bp_node["z"], bp_node["y"], bp_node["x"]],
                "position_on_trunk": round(position, 2),
                "max_length": round(max_length, 2),
            }
        )

    return {
        "coordinate_order": "ZYX",
        "summary": {
            "total_nodes": len(nodes),
            "total_length": round(total_length, 2),
            "main_trunk_length": round(trunk_length, 2),
            "num_branch_points": len(branch_point_ids),
        },
        "main_trunk": {
            "start": [round(c, 2) for c in start_coords],
            "end": [round(c, 2) for c in end_coords],
            "length": round(trunk_length, 2),
        },
        "branches": branches,
    }


def generate_labeled_tif(
    swc_path: str,
    output_path: str,
    shape: tuple[int, int, int],
    branch_point_radius: int = 2,
) -> None:
    """Generate RGB TIF image showing skeleton with colored labels.

    Colors: White=trunk, Yellow=branch points, Green=longest branch, Cyan=other branches.

    Args:
        swc_path: Path to the SWC file.
        output_path: Path to save the output TIF file.
        shape: Image shape as (Z, Y, X).
        branch_point_radius: Size of branch point markers in voxels.

    Returns:
        None. Saves TIF file to output_path.
    """
    nodes, adj = parse_swc(swc_path)
    if not nodes:
        return

    trunk_path, _ = find_main_trunk(nodes, adj)
    trunk_set = set(trunk_path)
    branch_point_ids = set(find_branch_points(trunk_path, adj))

    # Collect all max-length path nodes for all branch points
    max_path_nodes = set()
    for bp_id in branch_point_ids:
        path_nodes = find_max_length_path(bp_id, trunk_set, adj)
        max_path_nodes.update(path_nodes)

    z_dim, y_dim, x_dim = shape
    rgb_image = np.zeros((z_dim, y_dim, x_dim, 3), dtype=np.uint8)

    # Assign branch IDs to non-trunk, non-max-path nodes
    branch_assignments = {}
    branch_counter = 0
    for bp_id in branch_point_ids:
        for neighbor, _ in adj.get(bp_id, []):
            if neighbor not in trunk_set and neighbor not in branch_assignments:
                visited = {neighbor}
                queue = deque([neighbor])
                while queue:
                    curr = queue.popleft()
                    branch_assignments[curr] = branch_counter
                    for next_node, _ in adj.get(curr, []):
                        if next_node not in trunk_set and next_node not in visited:
                            visited.add(next_node)
                            queue.append(next_node)
                branch_counter += 1

    # Draw skeleton points
    for node_id, node in nodes.items():
        z = int(round(node["z"]))
        y = int(round(node["y"]))
        x = int(round(node["x"]))

        if not (0 <= z < z_dim and 0 <= y < y_dim and 0 <= x < x_dim):
            continue

        if node_id in branch_point_ids:
            for dz in range(-branch_point_radius, branch_point_radius + 1):
                for dy in range(-branch_point_radius, branch_point_radius + 1):
                    for dx in range(-branch_point_radius, branch_point_radius + 1):
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if 0 <= nz < z_dim and 0 <= ny < y_dim and 0 <= nx < x_dim:
                            rgb_image[nz, ny, nx] = COLOR_BRANCH_POINT
        elif node_id in trunk_set:
            rgb_image[z, y, x] = COLOR_TRUNK
        elif node_id in max_path_nodes:
            rgb_image[z, y, x] = COLOR_MAX_PATH
        elif node_id in branch_assignments:
            color_idx = branch_assignments[node_id] % len(BRANCH_COLORS)
            rgb_image[z, y, x] = BRANCH_COLORS[color_idx]

    tifffile.imwrite(output_path, rgb_image, photometric="rgb")


def generate_length_tif(
    swc_path: str, output_path: str, shape: tuple[int, int, int]
) -> None:
    """Generate 8-bit TIF with length values as voxel intensities.

    Trunk voxels = 255 (white), branch voxels = max_length (integer, capped at 254).

    Args:
        swc_path: Path to SWC file.
        output_path: Output TIF file path.
        shape: Image shape as (Z, Y, X).

    Returns:
        None. Saves TIF file to output_path.
    """
    nodes, adj = parse_swc(swc_path)
    if not nodes:
        return

    trunk_path, _ = find_main_trunk(nodes, adj)
    trunk_set = set(trunk_path)
    branch_point_ids = list(find_branch_points(trunk_path, adj))

    # Calculate max_length for each branch point and assign to all nodes in subtree
    branch_node_values = {}
    for bp_id in branch_point_ids:
        max_len = calculate_branch_max_length(bp_id, trunk_set, adj)
        max_len_int = min(254, int(round(max_len)))

        for neighbor, _ in adj.get(bp_id, []):
            if neighbor not in trunk_set and neighbor not in branch_node_values:
                visited = {neighbor}
                queue = deque([neighbor])
                while queue:
                    curr = queue.popleft()
                    branch_node_values[curr] = max_len_int
                    for next_node, _ in adj.get(curr, []):
                        if next_node not in trunk_set and next_node not in visited:
                            visited.add(next_node)
                            queue.append(next_node)

    z_dim, y_dim, x_dim = shape
    length_image = np.zeros((z_dim, y_dim, x_dim), dtype=np.uint8)

    for node_id, node in nodes.items():
        z = int(round(node["z"]))
        y = int(round(node["y"]))
        x = int(round(node["x"]))

        if not (0 <= z < z_dim and 0 <= y < y_dim and 0 <= x < x_dim):
            continue

        if node_id in trunk_set:
            length_image[z, y, x] = 255
        elif node_id in branch_node_values:
            length_image[z, y, x] = branch_node_values[node_id]

    tifffile.imwrite(output_path, length_image)


def _find_cleaned_tif(swc_path: str, cleaned_dir: str | None = None) -> str | None:
    """Find the cleaned TIF file that matches an SWC file.

    Args:
        swc_path: Path to the SWC file.
        cleaned_dir: Directory containing cleaned TIF files. If None, uses default location.

    Returns:
        Path to the cleaned TIF file, or None if not found.
    """
    if cleaned_dir is None:
        cleaned_dir = Path(swc_path).parent.parent / "03_clean"
        if not cleaned_dir.exists():
            return None
        cleaned_dir = str(cleaned_dir)

    swc_stem = Path(swc_path).stem
    if "_label-" in swc_stem:
        base = swc_stem.rsplit("_label-", 1)[0]
    else:
        base = swc_stem

    tif_path = os.path.join(cleaned_dir, f"{base}_clean.tif")
    if os.path.exists(tif_path):
        return tif_path

    return None


def run(input_path: str, output_dir: str, config: dict, logger: logging.Logger) -> str:
    """Run step 5: analyze each skeleton and write JSON, summary, and TIFs.

    Args:
        input_path: Path to input (previous step output directory).
        output_dir: Base output directory.
        config: Configuration dictionary.
        logger: Logger instance.

    Returns:
        Output directory path for this step.
    """
    input_dir = Path(input_path)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_path = Path(output_dir) / STEP_NAME
    output_path.mkdir(parents=True, exist_ok=True)

    # Find cleaned directory for TIF shape
    cleaned_dir = Path(output_dir) / "03_clean"
    if not cleaned_dir.exists():
        cleaned_dir = None

    # Get config parameters
    analysis_config = config.get("analyze", {})
    output_json = analysis_config.get("output_json", True)
    output_labeled_tif = analysis_config.get("output_labeled_tif", True)
    output_length_tif = analysis_config.get("output_length_tif", True)
    branch_point_radius = analysis_config.get("branch_point_radius", 2)

    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_path}")

    # Find SWC files
    swc_files = natsorted(
        [f for f in input_dir.iterdir() if f.suffix.lower() == ".swc"]
    )

    if not swc_files:
        raise ValueError(f"No SWC files found in: {input_dir}")

    logger.info(f"Found {len(swc_files)} SWC files")

    for idx, swc_file in enumerate(swc_files, start=1):
        logger.info(f"[{idx}/{len(swc_files)}] Processing: {swc_file.name}")

        result = analyze_skeleton(str(swc_file))
        input_stem = swc_file.stem

        # Save JSON
        if output_json:
            json_path = output_path / f"{input_stem}.json"
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.debug(f"  Saved JSON: {json_path}")

            # Save summary text file
            summary_path = output_path / f"{input_stem}_summary.txt"
            trunk = result["main_trunk"]
            with open(summary_path, "w") as f:
                f.write("Coordinate order: ZYX\n")
                f.write(
                    f"Trunk: {trunk['start']} -> {trunk['end']}, length={trunk['length']}\n\n"
                )
                f.write("ID  Z      Y      X      MaxLen\n")
                for b in result["branches"]:
                    bp = b["branch_point"]
                    f.write(
                        f"{b['id']:<3} {bp[0]:<6.0f} {bp[1]:<6.0f} {bp[2]:<6.0f} {b['max_length']}\n"
                    )

        # Save TIF outputs
        if output_labeled_tif or output_length_tif:
            cleaned_tif = _find_cleaned_tif(
                str(swc_file), str(cleaned_dir) if cleaned_dir else None
            )
            if cleaned_tif:
                with tifffile.TiffFile(cleaned_tif) as tif:
                    shape = tif.pages[0].shape
                    if len(tif.pages) > 1:
                        shape = (len(tif.pages),) + shape
                    elif tif.series[0].shape[0] > 1:
                        shape = tif.series[0].shape

                if output_labeled_tif:
                    tif_path = output_path / f"{input_stem}_labeled.tif"
                    generate_labeled_tif(
                        str(swc_file), str(tif_path), shape, branch_point_radius
                    )
                    logger.debug(f"  Saved labeled TIF: {tif_path}")

                if output_length_tif:
                    length_tif_path = output_path / f"{input_stem}_length.tif"
                    generate_length_tif(str(swc_file), str(length_tif_path), shape)
                    logger.debug(f"  Saved length TIF: {length_tif_path}")
            else:
                logger.warning("  Cleaned TIF not found, skipping TIF outputs")

        # Log summary
        summary = result["summary"]
        logger.info(
            f"  Trunk length: {summary['main_trunk_length']:.1f}, "
            f"Branch points: {summary['num_branch_points']}, "
            f"Total length: {summary['total_length']:.1f}"
        )

    return str(output_path)
