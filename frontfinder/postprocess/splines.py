"""Turn front probability grids into smooth, typed front lines ("splines").

The steps follow WindBorne's front events, which draw lines along the crests of FrontFinder's probability
bands, give one ID to each continuous boundary, and pick symbol sides from motion, temperature contrast and
cross-front wind (https://api.windbornesystems.com/weather/version_1/events/about/). The cross-front
sampling follows Niebler et al. (2022, https://doi.org/10.5194/wcd-3-113-2022):

1. Add up the probabilities of the front types in a group. Cold, warm, stationary and occluded fronts
   form one group, so a boundary that changes type along its length stays one line.
2. Threshold the summed probability and thin each band to a one-pixel crest with ``skeletonize``.
3. Order each crest's pixels into paths: the longest path first, then any long branches.
4. Cut each path into arms where its branches leave it, so a junction where three fronts meet keeps its
   sharp corner, fit a smoothing spline through each arm and resample it at an even spacing, and pin the
   arms that meet at a junction to one shared vertex, so they have no gap between them.
5. Label each point with its most likely front type and cut the line into typed sections.
6. Pick the side each section's symbols go on, from the temperature contrast across the line, the wind
   across it, and how far it moved since the previous time. Near a junction the samples that fall across
   another arm of the same front are ignored.
7. Track fronts from one time to the next with MODE-style matching (distance, length and type), so a
   front keeps one event ID, and record splits and merges as lineage.

On a global grid the longitudes wrap, so a front that crosses the grid's edge stays one front.

Vendored from ai2es/fronts (branch feature/front-splines, src/fronts/splines.py) without its command-line
entry point, so the serving pipeline does not depend on the training stack (TensorFlow, W&B, ...). The body
is no longer byte-identical to upstream: it carries local junction handling (see ``find_junctions``,
``split_at_corners``, ``pin_end``). Keep the rest in sync with upstream; call ``extract_fronts_over_time``
and ``to_geojson`` from ``front_lines.py``.
"""

import dataclasses
import itertools
import json
from collections.abc import Iterator

import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph
import scipy.spatial
import xarray as xr
from skimage.morphology import skeletonize

EARTH_RADIUS_KM = 6371.0
# Kilometres per degree of latitude (111.19 km), used to read short stretches as flat distances.
KM_PER_DEG = float(np.radians(EARTH_RADIUS_KM))

# GeoJSON names, in the order section features are written (WindBorne's order for the four fronts).
CLASSIFICATIONS: dict[str, str] = {
    "CF": "cold",
    "WF": "warm",
    "SF": "stationary",
    "OF": "occluded",
    "DL": "dryline",
    "TROF": "trough",
    "TT": "tropical_trough",
    "INST": "instability_axis",
}
SECTION_NAMES: dict[str, str] = {
    "CF": "cold_front",
    "WF": "warm_front",
    "SF": "stationary_front",
    "OF": "occluded_front",
    "DL": "dryline",
    "TROF": "trough",
    "TT": "tropical_trough",
    "INST": "instability_axis",
}

# How much each clue counts toward putting the symbols on the left: (warm air on the left, wind blowing
# toward the left, motion toward the left).
SYMBOL_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "CF": (1.0, 1.0, 1.0),  # triangles point into the warm air the front pushes into
    "WF": (-1.0, 1.0, 1.0),  # semicircles point into the cold air the front pushes into
    "SF": (1.0, 0.0, 0.0),  # blue triangles point toward the warm air; the front barely moves
    "OF": (0.0, 1.0, 1.0),  # symbols point the way the front moves
}
DEFAULT_SYMBOL_WEIGHTS = (0.0, 1.0, 1.0)


@dataclasses.dataclass
class SplineConfig:
    """Settings for turning probabilities into front lines.

    Attributes:
        groups: Front types whose probabilities are summed into one boundary. Each line in a group is cut
            into sections by its most likely type.
        threshold: Summed probability a grid point needs to be part of a front band. The probabilities
            are uncalibrated, so tune this against the max-CSI thresholds from ``fronts.evaluate``.
        min_length_km: Free-standing crests shorter than this are dropped, which removes the short spurs
            thinning leaves. Branches use ``min_branch_km`` instead when it is set.
        smoothing_deg: About how far (degrees) the spline may stray from the crest pixels.
        spacing_km: Distance between points on the output lines.
        min_section_km: Type changes shorter than this are folded into the neighbouring section.
        offset_km: How far either side of the line temperature and wind are sampled.
        temperature_scale_k: Temperature contrast (K) at which that clue counts fully.
        wind_scale_ms: Cross-front wind (m/s) at which that clue counts fully.
        motion_scale_km: Motion since the previous time (km) at which that clue counts fully.
        max_motion_km: Farthest a point may be from the previous time's fronts and still count as motion.
        match_distance_km: Mean distance between two fronts at which the distance clue for tracking drops to
            zero. Fronts farther apart than this never match.
        match_threshold: Total interest two fronts at consecutive times need to count as the same front.
        distance_weight: Weight of the distance clue in the total interest.
        length_weight: Weight of the length-ratio clue in the total interest.
        type_weight: Weight of the shared-front-type clue in the total interest.
        min_branch_km: Shortest branch to keep when it joins another path of the same crest. Arms that end
            at a junction are worth keeping even when they are too short to stand on their own, because a
            short third arm is what makes a triple point.
        corner_deg: A path is cut at a junction where its direction turns by at least this many degrees,
            so the corner between the two arms is not smoothed away.
        corner_window_km: How far before and after a junction the turn is measured over.
        pin_taper_km: Distance over which a pinned end blends back into where the spline put it.
        clearance_km: Symbol sampling points closer than this to another arm of the same crest are ignored,
            so a junction does not read the air across its neighbouring arms.
    """

    groups: list[list[str]] = dataclasses.field(
        default_factory=lambda: [["CF", "WF", "SF", "OF"], ["DL"], ["TROF"], ["TT"], ["INST"]]
    )
    threshold: float = 0.3
    min_length_km: float = 250.0
    smoothing_deg: float = 0.25
    spacing_km: float = 25.0
    min_section_km: float = 100.0
    offset_km: float = 150.0
    temperature_scale_k: float = 2.0
    wind_scale_ms: float = 2.0
    motion_scale_km: float = 50.0
    max_motion_km: float = 500.0
    match_distance_km: float = 500.0
    match_threshold: float = 0.6
    distance_weight: float = 4.0
    length_weight: float = 1.0
    type_weight: float = 1.0
    min_branch_km: float = 150.0
    corner_deg: float = 40.0
    corner_window_km: float = 100.0
    pin_taper_km: float = 75.0
    clearance_km: float = 100.0


@dataclasses.dataclass
class FrontSection:
    """One stretch of a boundary with a single front type.

    Attributes:
        front_type: Front type key, e.g. "CF".
        lon: Longitudes of the ordered points, in the probability grid's convention. On a global grid a
            line crossing the grid's edge carries on past it (e.g. 355, 360, 365) instead of jumping back.
        lat: Latitudes of the ordered points.
        probability: Probability of ``front_type`` at each point.
        symbol_side: "left" or "right", seen walking from the first point to the last. Symbols go on this
            side. For stationary fronts the blue triangles go here and the red semicircles go opposite.
    """

    front_type: str
    lon: np.ndarray
    lat: np.ndarray
    probability: np.ndarray
    symbol_side: str = "left"


@dataclasses.dataclass
class FrontBoundary:
    """A continuous boundary: one crest, as one or more lines of typed sections.

    Attributes:
        lines: Each line is a list of sections in order; each section starts where the last one ends. The
            first line is the first arm of the crest's longest path, then its other arms, then the arms of
            each branch off it. Lines that meet at a junction share one endpoint.
        event_id: Tracking ID, the same for the same front at every time. Set by ``track_fronts``.
        parents: IDs of other fronts at the previous time that this one split off from or merged with.
    """

    lines: list[list[FrontSection]]
    event_id: str = ""
    parents: list[str] = dataclasses.field(default_factory=list)


def haversine_km(
    lon1: np.ndarray | float, lat1: np.ndarray | float, lon2: np.ndarray | float, lat2: np.ndarray | float
) -> np.ndarray:
    """Return great-circle distances (km) between two sets of points given in degrees."""
    lon1, lat1, lon2, lat2 = (np.radians(x) for x in (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def line_length_km(lon: np.ndarray, lat: np.ndarray) -> float:
    """Return the length (km) of a line through ordered points."""
    return float(np.sum(haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:])))


def is_periodic(lon: np.ndarray) -> bool:
    """Return whether evenly spaced longitudes go all the way around the globe."""
    return len(lon) > 1 and bool(np.isclose(len(lon) * abs(lon[1] - lon[0]), 360.0))


def crest_mask(probability: np.ndarray, threshold: float, periodic: bool = False) -> np.ndarray:
    """Return the one-pixel-wide crests of the bands where ``probability >= threshold``.

    Args:
        probability: 2-D probability grid, dims (latitude, longitude). NaNs count as zero.
        threshold: Probability a grid point needs to be inside a band.
        periodic: The longitudes go all the way around, so bands carry on across the grid's edge. The grid
            is padded with columns from the other side before thinning, so a band crossing the edge is
            thinned as one band.

    Returns:
        Boolean grid, True on the thinned crest lines.
    """
    band = np.nan_to_num(probability) >= threshold
    if not periodic:
        return skeletonize(band)
    pad = min(32, band.shape[1])
    return skeletonize(np.pad(band, ((0, 0), (pad, pad)), mode="wrap"))[:, pad:-pad]


def skeleton_graph(mask: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, scipy.sparse.csr_array]:
    """Link each crest pixel to its 8 neighbours.

    Args:
        mask: Boolean crest grid, dims (latitude, longitude).
        lat: Latitude of each row.
        lon: Longitude of each column. If they go all the way around, the first and last columns are
            neighbours.

    Returns:
        The (row, col) index of each crest pixel, shape (n, 2), and an (n, n) graph whose edge weights are
        the distances (km) between neighbouring pixels. Each edge is stored once.
    """
    periodic = is_periodic(lon)
    rows, cols = np.nonzero(mask)
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[rows, cols] = np.arange(len(rows))
    starts, ends = [], []
    for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
        r2, c2 = rows + dr, cols + dc
        if periodic:
            c2 %= mask.shape[1]
        inside = (r2 < mask.shape[0]) & (c2 >= 0) & (c2 < mask.shape[1])
        neighbour = np.full(len(rows), -1, dtype=np.int64)
        neighbour[inside] = index[r2[inside], c2[inside]]
        starts.append(np.flatnonzero(neighbour >= 0))
        ends.append(neighbour[neighbour >= 0])
    i, j = np.concatenate(starts), np.concatenate(ends)
    weights = haversine_km(lon[cols[i]], lat[rows[i]], lon[cols[j]], lat[rows[j]])
    graph = scipy.sparse.coo_array((weights, (i, j)), shape=(len(rows), len(rows))).tocsr()
    return np.column_stack([rows, cols]), graph


def farthest_path(tree: scipy.sparse.csr_array, nodes: np.ndarray, start: int) -> tuple[np.ndarray, float]:
    """Return the path from ``start`` to the farthest node, staying within ``nodes``.

    Args:
        tree: Symmetric weighted tree (or forest) over all crest pixels.
        nodes: Sorted node indices the path may use. Must contain ``start``.
        start: Node the path begins at.

    Returns:
        Node indices along the path, starting at ``start``, and its length.
    """
    dist, pred = scipy.sparse.csgraph.dijkstra(
        tree[nodes[:, None], nodes], indices=int(np.searchsorted(nodes, start)), return_predecessors=True
    )
    end = int(np.argmax(np.where(np.isfinite(dist), dist, -1)))
    path = [end]
    while pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    return nodes[path[::-1]], float(dist[end])


def trace_paths(
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    min_length_km: float,
    min_branch_km: float | None = None,
) -> list[list[np.ndarray]]:
    """Order the pixels of each crest into paths.

    Each crest becomes a tree: the minimum spanning tree of its 8-connected pixels, which opens loops and
    drops the extra diagonal links thinning leaves at corners. The longest path through the tree comes
    first. Each branch left over is then followed from where it leaves the paths found so far to its far
    tip. Paths shorter than ``min_length_km`` are dropped, which removes the short spurs thinning leaves.

    Args:
        mask: Boolean crest grid, dims (latitude, longitude).
        lat: Latitude of each row.
        lon: Longitude of each column.
        min_length_km: Shortest crest to keep.
        min_branch_km: Shortest branch to keep, when branches should be held to a different length than
            whole crests. Defaults to ``min_length_km``.

    Returns:
        One list per kept crest, holding (n, 2) arrays of (row, col) indices in path order. Branches start
        at the pixel they leave from, so they touch the path they branch off.
    """
    pixels, graph = skeleton_graph(mask, lat, lon)
    tree = scipy.sparse.csgraph.minimum_spanning_tree(graph)
    tree = scipy.sparse.csr_array(tree + tree.T)
    n_crests, labels = scipy.sparse.csgraph.connected_components(tree, directed=False)
    crests = []
    for label in range(n_crests):
        nodes = np.flatnonzero(labels == label)
        far_end = int(farthest_path(tree, nodes, int(nodes[0]))[0][-1])
        main, length = farthest_path(tree, nodes, far_end)
        if length < min_length_km:
            continue
        shortest_branch = min_length_km if min_branch_km is None else min_branch_km
        paths = [main]
        used = np.zeros(len(pixels), dtype=bool)
        used[main] = True
        pending = [nodes[~used[nodes]]]
        while pending:
            rest = pending.pop()
            if len(rest) == 0:
                continue
            n_branches, branch_labels = scipy.sparse.csgraph.connected_components(
                tree[rest[:, None], rest], directed=False
            )
            for branch_label in range(n_branches):
                branch_nodes = rest[branch_labels == branch_label]
                # In a tree, each leftover branch joins the used pixels by exactly one edge.
                links = scipy.sparse.coo_array(tree[branch_nodes])
                k = int(np.flatnonzero(used[links.col])[0])
                branch, length = farthest_path(tree, branch_nodes, int(branch_nodes[links.row[k]]))
                if length + links.data[k] < shortest_branch:
                    continue
                branch = np.concatenate([[links.col[k]], branch])
                paths.append(branch)
                used[branch] = True
                pending.append(branch_nodes[~used[branch_nodes]])
        crests.append([pixels[path] for path in paths])
    return crests


def path_points(path: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the longitudes and latitudes of a crest path's pixels, with longitudes unwrapped.

    Unwrapping keeps a path that crosses a global grid's edge continuous, so distances and angles read
    along the path instead of jumping 360 degrees.
    """
    return np.unwrap(lon[path[:, 1]], period=360.0), lat[path[:, 0]]


def find_junctions(crest: list[np.ndarray]) -> list[tuple[int, int, int]]:
    """Find where each branch of a crest meets a path found before it.

    A wave cyclone's cold, warm and occluded arms thin to one crest with three paths leaving one pixel, and
    any front that forks does the same, so junctions are found from the topology alone, not from the front
    types.

    Args:
        crest: The paths of one crest, as ``trace_paths`` returns them. Each branch starts at the pixel it
            leaves from, which lies on a path before it.

    Returns:
        (path index, position in that path, branch index) for each branch, in ``trace_paths``' order.
        Branches that start off every earlier path (which ``trace_paths`` does not produce) are skipped.
    """
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    for index, path in enumerate(crest):
        for position, (row, col) in enumerate(path):
            seen.setdefault((int(row), int(col)), (index, position))
    junctions = []
    for branch, path in enumerate(crest[1:], start=1):
        row, col = path[0]
        parent = seen.get((int(row), int(col)))
        if parent is not None and parent[0] != branch:
            junctions.append((parent[0], parent[1], branch))
    return junctions


@dataclasses.dataclass
class Junction:
    """One pixel where arms of a crest meet.

    Attributes:
        pixel: (row, col) index of the crest pixel the arms meet at.
        arms: (arm index, end) for every arm that ends here, with end 0 for the arm's first point and -1
            for its last.
        corner: Whether the path that met here was cut, so the arms that end here form a corner.
        through: Index of the arm that carries on through here, or None when a corner was cut.
    """

    pixel: tuple[int, int]
    arms: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    corner: bool = False
    through: int | None = None


def turn_angle_deg(lon: np.ndarray, lat: np.ndarray, k: int, window_km: float) -> float:
    """Return how far (degrees) the path through ordered points turns at point ``k``.

    Each leg is read from ``k`` out to the nearest point at least ``window_km`` away, clipped to the path's
    ends, and the two legs are compared as flat distances in the local east/north frame.
    """
    dist = np.concatenate([[0.0], np.cumsum(haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:]))])
    back = np.flatnonzero(dist[k] - dist[: k + 1] >= window_km)
    ahead = np.flatnonzero(dist[k:] - dist[k] >= window_km)
    before = int(back[-1]) if back.size else 0
    after = k + int(ahead[0]) if ahead.size else len(lon) - 1
    dlon = (np.diff(lon) + 180.0) % 360.0 - 180.0
    east = np.concatenate([[0.0], np.cumsum(dlon * np.cos(np.radians(lat[:-1] + lat[1:]) / 2) * KM_PER_DEG)])
    north = np.concatenate([[0.0], np.cumsum(np.diff(lat) * KM_PER_DEG)])
    coming = (east[k] - east[before], north[k] - north[before])
    going = (east[after] - east[k], north[after] - north[k])
    scale = float(np.hypot(*coming) * np.hypot(*going))
    if scale == 0.0:
        return 0.0
    cos = (coming[0] * going[0] + coming[1] * going[1]) / scale
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def split_at_corners(
    crest: list[np.ndarray],
    junctions: list[tuple[int, int, int]],
    lat: np.ndarray,
    lon: np.ndarray,
    config: SplineConfig,
) -> tuple[list[np.ndarray], list[Junction]]:
    """Cut each path of a crest into arms where it turns sharply at a junction.

    A path is cut at a junction when its direction turns by at least ``config.corner_deg`` there and both
    sides run at least ``config.min_section_km``, so the corner between the arms survives smoothing. A path
    that carries on through a junction is left whole and the arm that left it is pinned onto it later (see
    ``pin_junctions``). A path may be cut at several junctions, and both arms of a cut hold the pixel the
    cut was made at.

    Args:
        crest: The paths of one crest, as ``trace_paths`` returns them.
        junctions: Junctions of those paths, from ``find_junctions``.
        lat: Latitude of each row.
        lon: Longitude of each column.
        config: Spline settings (corner angle and window, shortest section).

    Returns:
        The arms, as (row, col) arrays in walking order, and one ``Junction`` per place where paths meet.
    """
    cuts: dict[int, list[int]] = {}
    for parent, k, _branch in junctions:
        path_lon, path_lat = path_points(crest[parent], lat, lon)
        if k <= 0 or k >= len(path_lon) - 1:
            continue
        dist = np.concatenate(
            [[0.0], np.cumsum(haversine_km(path_lon[:-1], path_lat[:-1], path_lon[1:], path_lat[1:]))]
        )
        if dist[k] < config.min_section_km or dist[-1] - dist[k] < config.min_section_km:
            continue
        if turn_angle_deg(path_lon, path_lat, k, config.corner_window_km) >= config.corner_deg:
            # Two branches can leave at the same pixel, which is one corner, not two cuts.
            cuts.setdefault(parent, [])
            if k not in cuts[parent]:
                cuts[parent].append(k)
    arms: list[np.ndarray] = []
    # For each path, the arms cut from it as (arm index, first and last position in the path).
    spans: list[list[tuple[int, int, int]]] = []
    for index, path in enumerate(crest):
        bounds = [0, *sorted(cuts.get(index, [])), len(path) - 1]
        segments = list(zip(bounds, bounds[1:]))
        spans.append([(len(arms) + i, start, stop) for i, (start, stop) in enumerate(segments)])
        arms.extend(path[start : stop + 1] for start, stop in segments)
    joined: dict[tuple[int, int], Junction] = {}
    for parent, k, branch in junctions:
        if (parent, k) not in joined:
            corner = k in cuts.get(parent, [])
            ends, through = [], None
            for arm, start, stop in spans[parent]:
                if corner and stop == k:
                    ends.append((arm, -1))
                elif corner and start == k:
                    ends.append((arm, 0))
                elif start <= k <= stop:
                    through = arm
            row, col = crest[parent][k]
            joined[(parent, k)] = Junction(pixel=(int(row), int(col)), arms=ends, corner=corner, through=through)
        joined[(parent, k)].arms.append((next(arm for arm, start, _ in spans[branch] if start == 0), 0))
    return arms, list(joined.values())


def lon_near(lon: float, reference: float) -> float:
    """Return ``lon`` moved by whole turns of 360 degrees to the copy nearest ``reference``.

    Lines are unwrapped one at a time, so two ends of a junction can name the same place 360 degrees apart.
    """
    return float(lon) + 360.0 * round((float(reference) - float(lon)) / 360.0)


def project_to_polyline(lon: np.ndarray, lat: np.ndarray, point_lon: float, point_lat: float) -> tuple[float, float]:
    """Return the point of the line through (lon, lat) closest to a point, as (longitude, latitude)."""
    if len(lon) < 2:
        return float(lon[0]), float(lat[0])
    frame = lon + 360.0 * np.round((point_lon - lon) / 360.0)  # lon_near, for the whole line at once
    cos_lat = np.cos(np.radians(point_lat))
    east = np.radians(frame - point_lon) * EARTH_RADIUS_KM * cos_lat
    north = np.radians(lat - point_lat) * EARTH_RADIUS_KM
    de, dn = np.diff(east), np.diff(north)
    length = np.where(de**2 + dn**2 > 0.0, de**2 + dn**2, 1.0)
    # The query point is the origin of the frame, so its projection on a segment is the segment's start
    # plus how far along it the origin falls.
    step = np.clip(-(east[:-1] * de + north[:-1] * dn) / length, 0.0, 1.0)
    near = east[:-1] + step * de, north[:-1] + step * dn
    i = int(np.argmin(near[0] ** 2 + near[1] ** 2))
    return (
        point_lon + float(np.degrees(near[0][i] / (EARTH_RADIUS_KM * cos_lat))),
        point_lat + float(np.degrees(near[1][i] / EARTH_RADIUS_KM)),
    )


def pin_end(
    lon: np.ndarray, lat: np.ndarray, at_start: bool, target_lon: float, target_lat: float, taper_km: float
) -> tuple[np.ndarray, np.ndarray]:
    """Move one end of a line onto a point, fading the move out along the line.

    The end point lands exactly on the target, so the lines that meet there share one vertex. Every other
    point moves by 0.5 * (1 + cos(pi * d / taper)) of the move, where ``d`` is how far it is along the line
    from that end, so the line eases back to where the spline put it over ``taper_km``, or half its length
    when it is shorter. Points past the taper are left alone.

    Args:
        lon: Longitudes of the line's points.
        lat: Latitudes of the line's points.
        at_start: Move the first point onto the target, instead of the last one.
        target_lon: Longitude to move the end onto, in any of the grid's turns.
        target_lat: Latitude to move the end onto.
        taper_km: Distance over which the move fades out.

    Returns:
        Longitudes and latitudes of the moved line, in the line's own longitude frame.
    """
    end = 0 if at_start else -1
    target_lon = lon_near(target_lon, float(lon[end]))
    dist = np.concatenate([[0.0], np.cumsum(haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:]))])
    taper = min(taper_km, dist[-1] / 2)
    along = dist if at_start else dist[-1] - dist
    scaled = np.divide(along, taper, out=np.zeros_like(along), where=taper > 0)
    weight = np.where(along < taper, 0.5 * (1 + np.cos(np.pi * scaled)), 0.0)
    new_lon = lon + weight * (target_lon - lon[end])
    new_lat = lat + weight * (target_lat - lat[end])
    new_lon[end], new_lat[end] = target_lon, target_lat
    return new_lon, new_lat


def pin_junctions(
    lines: list[tuple[np.ndarray, np.ndarray]], junctions: list[Junction], config: SplineConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pin the ends of the lines that meet at each junction onto one shared vertex.

    At a corner, every line that ends there moves onto the mean of their end points, which keeps the
    corner's angle and closes the gap of about a pixel the separate splines leave. Where one line carries
    on through a junction, it is left where it is and the line that left it ends on the point of it that is
    closest, so the branch lands on the front it came off.

    Args:
        lines: Smoothed lines, as (longitude, latitude) pairs.
        junctions: Junctions of those lines, from ``split_at_corners``.
        config: Spline settings (pin taper).

    Returns:
        The lines with their meeting ends pinned, in the same order.
    """
    moves: list[tuple[int, bool, float, float]] = []
    for junction in junctions:
        ends = [(arm, end, lines[arm][0][end], lines[arm][1][end]) for arm, end in junction.arms]
        if not ends:
            continue
        if junction.corner:
            reference = float(ends[0][2])
            target_lon = float(np.mean([lon_near(lon, reference) for _, _, lon, _ in ends]))
            target_lat = float(np.mean([lat for _, _, _, lat in ends]))
            moves.extend((arm, end == 0, target_lon, target_lat) for arm, end, _, _ in ends)
            continue
        if junction.through is None:
            continue
        parent_lon, parent_lat = lines[junction.through]
        for arm, end, lon, lat in ends:
            moves.append((arm, end == 0, *project_to_polyline(parent_lon, parent_lat, lon, lat)))
    pinned = list(lines)
    for arm, at_start, target_lon, target_lat in moves:
        pinned[arm] = pin_end(*pinned[arm], at_start, target_lon, target_lat, config.pin_taper_km)
    return pinned


def smooth_line(
    lon: np.ndarray, lat: np.ndarray, smoothing_deg: float, spacing_km: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a smoothing spline through ordered points and resample it at an even spacing.

    Args:
        lon: Longitudes of the ordered points.
        lat: Latitudes of the ordered points.
        smoothing_deg: About how far (degrees) the spline may stray from the points.
        spacing_km: Distance between output points.

    Returns:
        Longitudes and latitudes of the resampled line, which runs from near the first point to near the
        last.
    """
    n = len(lon)
    tck, _ = scipy.interpolate.splprep([lon, lat], s=n * smoothing_deg**2, k=min(3, n - 1))
    dense_lon, dense_lat = scipy.interpolate.splev(np.linspace(0, 1, max(200, 10 * n)), tck)
    dist = np.concatenate(
        [[0.0], np.cumsum(haversine_km(dense_lon[:-1], dense_lat[:-1], dense_lon[1:], dense_lat[1:]))]
    )
    targets = np.linspace(0, dist[-1], max(2, round(dist[-1] / spacing_km) + 1))
    return np.interp(targets, dist, dense_lon), np.interp(targets, dist, dense_lat)


def split_sections(
    probabilities: np.ndarray, lon: np.ndarray, lat: np.ndarray, min_section_km: float
) -> list[tuple[int, int, int]]:
    """Cut a line into stretches with one most-likely front type each.

    Stretches shorter than ``min_section_km`` are folded into their longer neighbour, shortest first, so
    a brief flicker between types does not chop the line up.

    Args:
        probabilities: Probability of each front type at each point, shape (n_types, n_points).
        lon: Longitudes of the points.
        lat: Latitudes of the points.
        min_section_km: Shortest stretch to keep as its own section.

    Returns:
        (type index, first point, last point) for each section in order. Each section's last point is the
        next section's first point.
    """
    labels = np.argmax(probabilities, axis=0)
    dist = np.concatenate([[0.0], np.cumsum(haversine_km(lon[:-1], lat[:-1], lon[1:], lat[1:]))])
    runs = [[int(labels[start]), int(start)] for start in np.concatenate([[0], np.flatnonzero(np.diff(labels)) + 1])]
    while len(runs) > 1:
        bounds = [start for _, start in runs] + [len(lon) - 1]
        lengths = [dist[bounds[i + 1]] - dist[bounds[i]] for i in range(len(runs))]
        i = int(np.argmin(lengths))
        if lengths[i] >= min_section_km:
            break
        if i == 0:
            j = 1
        elif i == len(runs) - 1:
            j = i - 1
        else:
            j = i - 1 if lengths[i - 1] >= lengths[i + 1] else i + 1
        runs[i][0] = runs[j][0]
        runs = [run for k, run in enumerate(runs) if k == 0 or run[0] != runs[k - 1][0]]
    bounds = [start for _, start in runs] + [len(lon) - 1]
    return [(label, start, stop) for (label, start), stop in zip(runs, bounds[1:], strict=True)]


def left_normals(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unit vectors (east, north components) pointing left of the direction of travel."""
    east = np.gradient(lon) * np.cos(np.radians(lat))
    north = np.gradient(lat)
    length = np.hypot(east, north)
    return -north / length, east / length


def offset_points(
    lon: np.ndarray, lat: np.ndarray, east_km: np.ndarray, north_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Move points by the given distances (km) east and north."""
    new_lat = lat + np.degrees(north_km / EARTH_RADIUS_KM)
    new_lon = lon + np.degrees(east_km / (EARTH_RADIUS_KM * np.cos(np.radians(lat))))
    return new_lon, new_lat


def fractional_index(coord: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return where ``values`` fall along a monotonic (ascending or descending) coordinate, in index units.

    Values outside the coordinate's range give NaN.
    """
    steps = np.arange(len(coord), dtype=float)
    if coord[0] > coord[-1]:
        coord, steps = coord[::-1], steps[::-1]
    index = np.interp(values, coord, steps)
    return np.where((values >= coord[0]) & (values <= coord[-1]), index, np.nan)


def sample(field: xr.DataArray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate a (latitude, longitude) field at points.

    On a global grid, longitudes wrap, so 365 reads the same place as 5. Otherwise points off the grid give
    NaN.
    """
    grid = field.transpose("latitude", "longitude").values
    grid_lon = field["longitude"].values
    if is_periodic(grid_lon):
        # Repeat the first column past the last one so points between them interpolate across the edge.
        grid = np.concatenate([grid, grid[:, :1]], axis=1)
        grid_lon = np.append(grid_lon, grid_lon[0] + 360.0)
        lon = grid_lon[0] + (lon - grid_lon[0]) % 360.0
    rows = fractional_index(field["latitude"].values, lat)
    cols = fractional_index(grid_lon, lon)
    inside = np.isfinite(rows) & np.isfinite(cols)
    values = np.full(len(lon), np.nan)
    values[inside] = scipy.ndimage.map_coordinates(grid, [rows[inside], cols[inside]], order=1)
    return values


def finite_mean(values: np.ndarray) -> float:
    """Return the mean of the finite values, or 0 if there are none."""
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else 0.0


def masked_mean(values: np.ndarray, masked: np.ndarray) -> float:
    """Return ``finite_mean`` over the unmasked values, or over all of them when all are masked."""
    if masked.all():
        return finite_mean(values)
    return finite_mean(np.where(masked, np.nan, values))


def unit_vectors(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Return points as 3-D unit vectors, shape (n, 3), so straight-line distance grows with distance on Earth."""
    lon, lat = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


class PointIndex:
    """Front points with a k-d tree, for fast nearest-point searches on the sphere.

    Attributes:
        lon: Longitudes of the points.
        lat: Latitudes of the points.
    """

    def __init__(self, lon: np.ndarray, lat: np.ndarray) -> None:
        self.lon = lon
        self.lat = lat
        self.tree = scipy.spatial.KDTree(unit_vectors(lon, lat))

    def nearest(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the distance (km) to, and index of, the nearest indexed point for each query point."""
        chord, index = self.tree.query(unit_vectors(lon, lat))
        return 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2, 0, 1)), index

    def within(self, lon: np.ndarray, lat: np.ndarray, km: float) -> np.ndarray:
        """Return the indices of indexed points within ``km`` of any query point."""
        hits = self.tree.query_ball_point(unit_vectors(lon, lat), r=2 * np.sin(km / (2 * EARTH_RADIUS_KM)))
        return np.unique(np.fromiter(itertools.chain.from_iterable(hits), dtype=np.int64))


def clearance_mask(lon: np.ndarray, lat: np.ndarray, others: PointIndex | None, clearance_km: float) -> np.ndarray:
    """Return which points come within ``clearance_km`` of ``others``, all False when there are none."""
    if others is None:
        return np.zeros(len(lon), dtype=bool)
    return others.nearest(lon, lat)[0] < clearance_km


def other_arm_points(lines: list[tuple[np.ndarray, np.ndarray]], index: int) -> PointIndex | None:
    """Return the points of every line but ``index``, for keeping an arm's samples off its neighbours."""
    rest = [line for other, line in enumerate(lines) if other != index]
    if not rest:
        return None
    return PointIndex(
        np.concatenate([lon for lon, _ in rest]),
        np.concatenate([lat for _, lat in rest]),
    )


def motion_km(
    lon: np.ndarray, lat: np.ndarray, east: np.ndarray, north: np.ndarray, previous: PointIndex, max_km: float
) -> float:
    """Return how far (km) a line moved along its left normal since the previous time.

    Each point is matched to the nearest previous front point within ``max_km``.

    Args:
        lon: Longitudes of the line now.
        lat: Latitudes of the line now.
        east: East component of the left normal at each point.
        north: North component of the left normal at each point.
        previous: Front points at the previous time.
        max_km: Farthest match to count.

    Returns:
        Mean displacement toward the left (negative means it moved right), or 0 with no matches.
    """
    dist, nearest = previous.nearest(lon, lat)
    close = dist <= max_km
    dlon = (lon - previous.lon[nearest] + 180.0) % 360.0 - 180.0
    disp_east = np.radians(dlon) * EARTH_RADIUS_KM * np.cos(np.radians(lat))
    disp_north = np.radians(lat - previous.lat[nearest]) * EARTH_RADIUS_KM
    return finite_mean((disp_east * east + disp_north * north)[close])


def symbol_side(
    section: FrontSection,
    config: SplineConfig,
    temperature: xr.DataArray | None = None,
    u: xr.DataArray | None = None,
    v: xr.DataArray | None = None,
    previous: PointIndex | None = None,
    others: PointIndex | None = None,
) -> str:
    """Pick which side of a section its symbols go on.

    Three clues are each squashed to (-1, 1) with ``tanh(value / scale)``, weighted by
    ``SYMBOL_WEIGHTS`` for the section's front type, and added. A missing input gives a clue of 0. A
    positive (or zero) total puts the symbols on the left.

    - Temperature: mean of (left minus right), sampled ``offset_km`` either side of the line.
    - Wind: mean wind across the line toward the left, sampled on the line and at both offsets
      (the cross-section ordering in Niebler et al. 2022).
    - Motion: how far the line moved toward the left since the previous time.

    Args:
        section: Section to orient.
        config: Spline settings (offset and clue scales).
        temperature: Temperature-like field, dims (latitude, longitude). Equivalent potential temperature
            at 850 hPa works well.
        u: Eastward wind (m/s), dims (latitude, longitude).
        v: Northward wind (m/s), dims (latitude, longitude).
        previous: The same group's front points at the previous time.
        others: The other arms of the same crest. Sampling points that fall within ``clearance_km`` of one
            of them are dropped, since they read the air across a neighbouring arm rather than across this
            section. A clue whose every sample is dropped falls back to reading them all.

    Returns:
        "left" or "right".
    """
    east, north = left_normals(section.lon, section.lat)
    d = config.offset_km
    left = offset_points(section.lon, section.lat, east * d, north * d)
    right = offset_points(section.lon, section.lat, -east * d, -north * d)
    on = (section.lon, section.lat)
    near = [clearance_mask(*pts, others, config.clearance_km) for pts in (left, on, right)]
    clues = np.zeros(3)
    if temperature is not None:
        contrast = sample(temperature, *left) - sample(temperature, *right)
        # A pair only says something when the air on both sides of the line belongs to this section.
        clues[0] = np.tanh(masked_mean(contrast, near[0] | near[2]) / config.temperature_scale_k)
    if u is not None and v is not None:
        cross = np.concatenate([sample(u, *pts) * east + sample(v, *pts) * north for pts in (left, on, right)])
        clues[1] = np.tanh(masked_mean(cross, np.concatenate(near)) / config.wind_scale_ms)
    if previous is not None:
        moved = motion_km(section.lon, section.lat, east, north, previous, config.max_motion_km)
        clues[2] = np.tanh(moved / config.motion_scale_km)
    score = float(np.dot(SYMBOL_WEIGHTS.get(section.front_type, DEFAULT_SYMBOL_WEIGHTS), clues))
    return "left" if score >= 0 else "right"


def extract_fronts(
    probabilities: xr.Dataset,
    config: SplineConfig,
    temperature: xr.DataArray | None = None,
    u: xr.DataArray | None = None,
    v: xr.DataArray | None = None,
    previous: list[FrontBoundary] | None = None,
) -> list[FrontBoundary]:
    """Turn one time's probability grids into front lines.

    Args:
        probabilities: One variable per front type (e.g. "CF"), dims (latitude, longitude). Types not in
            any of ``config.groups`` are ignored, as are groups with no types present.
        config: Spline settings.
        temperature: Optional temperature-like field for orienting symbols, on any grid with the same
            longitude convention as ``probabilities``.
        u: Optional eastward wind (m/s) for orienting symbols.
        v: Optional northward wind (m/s) for orienting symbols.
        previous: Fronts found at the previous time, used to orient symbols by motion.

    Returns:
        Boundaries, grouped by ``config.groups`` in order.
    """
    # Sampling reads these many times, so convert them once.
    probabilities = probabilities.astype(np.float64)
    temperature, u, v = (None if f is None else f.astype(np.float64) for f in (temperature, u, v))
    lat = probabilities["latitude"].values
    lon = probabilities["longitude"].values
    boundaries = []
    for group in config.groups:
        types = [ft for ft in group if ft in probabilities]
        if not types:
            continue
        total = np.sum([np.nan_to_num(probabilities[ft].values) for ft in types], axis=0)
        previous_sections = [s for b in previous or [] for line in b.lines for s in line if s.front_type in types]
        previous_points = (
            PointIndex(
                np.concatenate([s.lon for s in previous_sections]), np.concatenate([s.lat for s in previous_sections])
            )
            if previous_sections
            else None
        )
        mask = crest_mask(total, config.threshold, periodic=is_periodic(lon))
        for traced in trace_paths(mask, lat, lon, config.min_length_km, config.min_branch_km):
            # A spline needs two distinct pixels, and the arms are cut from the paths, so paths that cannot
            # be one are dropped before the junctions are read off the path numbering.
            crest = [path for path in traced if len(np.unique(path, axis=0)) > 1]
            if not crest:
                continue
            arms, junctions = split_at_corners(crest, find_junctions(crest), lat, lon, config)
            smoothed = [
                smooth_line(*path_points(arm, lat, lon), config.smoothing_deg, config.spacing_km) for arm in arms
            ]
            smoothed = pin_junctions(smoothed, junctions, config)
            lines = []
            for arm, (line_lon, line_lat) in enumerate(smoothed):
                # Near a junction the air either side of one arm lies across the other arms, so each arm is
                # told where the rest of its crest is and ignores the samples that fall on them.
                others = other_arm_points(smoothed, arm)
                probs = np.nan_to_num(np.stack([sample(probabilities[ft], line_lon, line_lat) for ft in types]))
                sections = []
                for k, start, stop in split_sections(probs, line_lon, line_lat, config.min_section_km):
                    points = slice(start, stop + 1)
                    section = FrontSection(types[k], line_lon[points], line_lat[points], probs[k, points])
                    section.symbol_side = symbol_side(section, config, temperature, u, v, previous_points, others)
                    sections.append(section)
                lines.append(sections)
            boundaries.append(FrontBoundary(lines))
    return boundaries


def longitude_shift(lon: float) -> float:
    """Return the whole number of turns (in degrees) that moves ``lon`` into [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0 - lon


def line_feature(
    event_id: str, name: str, parts: list[tuple[np.ndarray, np.ndarray]], summary: dict, valid_time: str | None
) -> dict:
    """Build a GeoJSON MultiLineString Feature with WindBorne-style properties."""
    properties = {"event_id": event_id, "name": name, "summary": summary}
    if valid_time is not None:
        properties["valid_time"] = valid_time
    coordinates = [np.column_stack([lon, lat]).round(4).tolist() for lon, lat in parts]
    return {
        "type": "Feature",
        "geometry": {"type": "MultiLineString", "coordinates": coordinates},
        "properties": properties,
    }


def to_geojson(boundaries: list[FrontBoundary], valid_time: str | None = None) -> dict:
    """Write fronts as a GeoJSON FeatureCollection in the layout of WindBorne's front events.

    Each boundary gets these Features, all carrying its ``event_id`` (or ``front_NNNN`` by position if it
    was never tracked):

    - ``overall_geometry``: every line of the boundary. Summary has ``classifications`` and
      ``length_km``.
    - One Feature per front type present (``cold_front``, ``warm_front``, ``stationary_front``,
      ``occluded_front``, ``dryline``, ``trough``, ...), holding that type's sections. Summary has
      ``classification``, ``length_km``, ``symbol_sides`` (one "left" or "right" per line part, seen
      walking the part from its first coordinate to its last) and ``probability_diagnostics``
      (``mean_probability`` and ``max_probability``, uncalibrated).

    Coordinates are WGS84 ``[longitude, latitude]``. Lines are never cut. Each line starts in [-180, 180)
    and its longitudes carry on from there, so a front crossing 180 degrees reads e.g. 175, 185, which
    MapLibre, Leaflet and d3 draw as one continuous line.

    Args:
        boundaries: Fronts from ``extract_fronts``.
        valid_time: Optional ISO 8601 time added to every Feature's properties.

    Returns:
        A GeoJSON FeatureCollection, ready for ``json.dump``.
    """
    features = []
    for number, boundary in enumerate(boundaries):
        event_id = boundary.event_id or f"front_{number:04d}"
        lines = [
            [dataclasses.replace(s, lon=s.lon + longitude_shift(float(line[0].lon[0]))) for s in line]
            for line in boundary.lines
        ]
        sections = [s for line in lines for s in line]
        overall = [
            (
                np.concatenate([line[0].lon, *(s.lon[1:] for s in line[1:])]),
                np.concatenate([line[0].lat, *(s.lat[1:] for s in line[1:])]),
            )
            for line in lines
        ]
        types = [ft for ft in CLASSIFICATIONS if any(s.front_type == ft for s in sections)]
        summary = {
            "classifications": sorted(CLASSIFICATIONS[ft] for ft in types),
            "length_km": round(boundary_length_km(boundary), 2),
        }
        features.append(line_feature(event_id, "overall_geometry", overall, summary, valid_time))
        for ft in types:
            typed = [s for s in sections if s.front_type == ft]
            probability = np.concatenate([s.probability for s in typed])
            summary = {
                "classification": CLASSIFICATIONS[ft],
                "length_km": round(sum(line_length_km(s.lon, s.lat) for s in typed), 2),
                "symbol_sides": [s.symbol_side for s in typed],
                "probability_diagnostics": {
                    "mean_probability": round(float(probability.mean()), 3),
                    "max_probability": round(float(probability.max()), 3),
                },
            }
            parts = [(s.lon, s.lat) for s in typed]
            features.append(line_feature(event_id, SECTION_NAMES[ft], parts, summary, valid_time))
    return {"type": "FeatureCollection", "features": features}


def boundary_points(boundary: FrontBoundary, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return every ``step``-th point of a boundary's sections, as longitudes and latitudes."""
    sections = [s for line in boundary.lines for s in line]
    lon = np.concatenate([s.lon for s in sections])[::step]
    lat = np.concatenate([s.lat for s in sections])[::step]
    return lon, lat


def boundary_length_km(boundary: FrontBoundary) -> float:
    """Return the total length (km) of a boundary's lines."""
    return sum(line_length_km(s.lon, s.lat) for line in boundary.lines for s in line)


def boundary_group(boundary: FrontBoundary, config: SplineConfig) -> int:
    """Return the index in ``config.groups`` of the group a boundary's front types come from."""
    front_type = boundary.lines[0][0].front_type
    return next(i for i, group in enumerate(config.groups) if front_type in group)


def match_interest(a: FrontBoundary, b: FrontBoundary, config: SplineConfig) -> float:
    """Score how likely two fronts at consecutive times are the same front, from 0 to 1.

    This is a small version of the MODE object matching that WindBorne uses (Davis et al. 2006): each clue
    is scored from 0 to 1, and the total interest is their weighted mean.

    - Distance: mean distance from the shorter front's points to the nearest point of the longer one.
      Scores 1 at 0 km and falls to 0 at ``match_distance_km``. Measuring from the shorter front lets a
      piece that split off (or merged in) score well against the whole.
    - Length: shorter length over longer length.
    - Type: shared front types over all front types of the two (cold, warm, ...).

    Fronts from different groups, or farther apart than ``match_distance_km``, score 0.

    Args:
        a: A front at one time.
        b: A front at the next time.
        config: Spline settings (match distance and weights).

    Returns:
        The total interest.
    """
    if boundary_group(a, config) != boundary_group(b, config):
        return 0.0
    step = max(1, round(100.0 / config.spacing_km))
    shorter, longer = sorted((a, b), key=boundary_length_km)
    slon, slat = boundary_points(shorter, step)
    llon, llat = boundary_points(longer, step)
    distance = float(PointIndex(llon, llat).nearest(slon, slat)[0].mean())
    if distance >= config.match_distance_km:
        return 0.0
    types_a = {s.front_type for line in a.lines for s in line}
    types_b = {s.front_type for line in b.lines for s in line}
    clues = np.array(
        [
            1.0 - distance / config.match_distance_km,
            boundary_length_km(shorter) / boundary_length_km(longer),
            len(types_a & types_b) / len(types_a | types_b),
        ]
    )
    weights = np.array([config.distance_weight, config.length_weight, config.type_weight])
    return float(weights @ clues / weights.sum())


def track_fronts(
    previous: list[FrontBoundary], current: list[FrontBoundary], config: SplineConfig, new_ids: Iterator[str]
) -> None:
    """Give each current front an event ID, carrying IDs over from the previous time.

    Pairs with a total interest of at least ``match_threshold`` are matched best first, one to one, and the
    current front takes the previous front's ID. Fronts left over get a new ID. Every other matching pair
    is kept as lineage: when a front splits, the new piece lists the old front as a parent; when fronts
    merge, the merged front lists the ones whose IDs it did not take.

    Args:
        previous: Fronts at the previous time, with IDs set. Empty at the first time.
        current: Fronts at this time. Their ``event_id`` and ``parents`` are set in place.
        config: Spline settings (match distance, weights and threshold).
        new_ids: Source of fresh event IDs.
    """
    pairs = []
    if previous and current:
        # Only pairs with a point within match_distance_km of each other can match, so find those first.
        step = max(1, round(100.0 / config.spacing_km))
        points = [boundary_points(c, step) for c in current]
        owners = np.concatenate([np.full(len(lon), j) for j, (lon, _) in enumerate(points)])
        index = PointIndex(np.concatenate([lon for lon, _ in points]), np.concatenate([lat for _, lat in points]))
        for i, p in enumerate(previous):
            for j in np.unique(owners[index.within(*boundary_points(p, step), config.match_distance_km)]):
                interest = match_interest(p, current[j], config)
                if interest >= config.match_threshold:
                    pairs.append((interest, i, int(j)))
    pairs.sort(reverse=True)
    taken_previous, taken_current = set(), set()
    for _, i, j in pairs:
        if i not in taken_previous and j not in taken_current:
            current[j].event_id = previous[i].event_id
            taken_previous.add(i)
            taken_current.add(j)
    for boundary in current:
        boundary.event_id = boundary.event_id or next(new_ids)
    for _, i, j in pairs:
        if previous[i].event_id != current[j].event_id:
            current[j].parents.append(previous[i].event_id)
    for boundary in current:
        boundary.parents.sort()


def event_index(results: list[tuple[str | None, list[FrontBoundary]]]) -> dict[str, dict]:
    """Summarize each tracked front, like the ``events`` index in WindBorne's GeoJSON.

    Args:
        results: Output of ``extract_fronts_over_time``.

    Returns:
        For each event ID: ``start_time`` and ``end_time`` (first and last time it is seen), ``lineage``
        (``parents`` and ``children`` event IDs) and ``summary`` (``classifications`` it ever had and
        ``max_length_km``).
    """
    events: dict[str, dict] = {}
    for time, boundaries in results:
        for boundary in boundaries:
            event = events.setdefault(
                boundary.event_id,
                {
                    "start_time": time,
                    "end_time": time,
                    "lineage": {"parents": [], "children": []},
                    "summary": {"classifications": [], "max_length_km": 0.0},
                },
            )
            event["end_time"] = time
            for parent in boundary.parents:
                if parent not in event["lineage"]["parents"]:
                    event["lineage"]["parents"].append(parent)
                if boundary.event_id not in events[parent]["lineage"]["children"]:
                    events[parent]["lineage"]["children"].append(boundary.event_id)
            summary = event["summary"]
            types = {CLASSIFICATIONS[s.front_type] for line in boundary.lines for s in line}
            summary["classifications"] = sorted(set(summary["classifications"]) | types)
            summary["max_length_km"] = max(summary["max_length_km"], round(boundary_length_km(boundary), 2))
    for event in events.values():
        event["lineage"]["parents"].sort()
        event["lineage"]["children"].sort()
    return events


def field_or_none(fields: xr.Dataset | None, name: str) -> xr.DataArray | None:
    """Return ``fields[name]`` loaded into memory, or None if there are no fields or no such variable."""
    return fields[name].load() if fields is not None and name in fields else None


def extract_fronts_over_time(
    probabilities: xr.Dataset,
    config: SplineConfig,
    fields: xr.Dataset | None = None,
    temperature_var: str = "equivalent_potential_temperature",
    u_var: str = "u_component_of_wind",
    v_var: str = "v_component_of_wind",
    level: int = 850,
) -> list[tuple[str | None, list[FrontBoundary]]]:
    """Run ``extract_fronts`` at every time and track the fronts from one time to the next.

    Each time's fronts are passed on to the next, both for the motion clue and for ``track_fronts``,
    which keeps a front's event ID the same across times.

    Args:
        probabilities: One variable per front type, dims ([time,] latitude, longitude). Times must be in
            order (e.g. forecast hours) for the motion clue to make sense.
        config: Spline settings.
        fields: Optional temperature and wind for orienting symbols, dims ([time,] [level,] latitude,
            longitude). Missing variables are skipped.
        temperature_var: Name of the temperature-like variable in ``fields``.
        u_var: Name of the eastward wind in ``fields``.
        v_var: Name of the northward wind in ``fields``.
        level: Pressure level (hPa) to read when ``fields`` has a ``level`` dim.

    Returns:
        (ISO 8601 time, or None without a time dim; fronts) for each time in order.
    """
    if fields is not None and "level" in fields.dims:
        fields = fields.sel(level=level)
    times = probabilities["time"].values if "time" in probabilities.dims else [None]
    results = []
    previous: list[FrontBoundary] = []
    new_ids = (f"front_{n:04d}" for n in itertools.count())
    for time in times:
        probs = (probabilities if time is None else probabilities.sel(time=time)).load()
        now = fields.sel(time=time) if fields is not None and time is not None and "time" in fields.dims else fields
        boundaries = extract_fronts(
            probs,
            config,
            temperature=field_or_none(now, temperature_var),
            u=field_or_none(now, u_var),
            v=field_or_none(now, v_var),
            previous=previous,
        )
        track_fronts(previous, boundaries, config, new_ids)
        results.append((None if time is None else str(np.datetime_as_string(time, unit="s")), boundaries))
        previous = boundaries
    return results
