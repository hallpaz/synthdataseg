"""
Vector geometry generators for synthetic segmentation shapes.

Supported shapes:
  - Class 1: 5-pointed regular star (pentagram)
  - Class 2: Basquiat-style 3-peaked crown
  - Class N (N >= 3): Strictly convex N-gons generated via Valtr's algorithm
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def polygon_area(vertices: np.ndarray) -> float:
    """
    Compute the signed or unsigned area of a 2D polygon using the Shoelace formula.

    Args:
        vertices: Array of shape (V, 2) defining polygon vertices in order.

    Returns:
        float: Absolute area of the polygon.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    # Shoelace formula: 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_signed_area(vertices: np.ndarray) -> float:
    """Compute signed area of 2D polygon. Positive indicates counter-clockwise orientation."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(vertices: np.ndarray) -> Tuple[float, float]:
    """
    Compute the geometric centroid (Cx, Cy) of a 2D polygon.

    Args:
        vertices: Array of shape (V, 2).

    Returns:
        Tuple[float, float]: (Cx, Cy).
    """
    signed_a = polygon_signed_area(vertices)
    if np.abs(signed_a) < 1e-9:
        # Fallback to mean of vertices if polygon is degenerate
        return float(np.mean(vertices[:, 0])), float(np.mean(vertices[:, 1]))

    x = vertices[:, 0]
    y = vertices[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    cross = x * y_next - x_next * y
    cx = float(np.sum((x + x_next) * cross) / (6.0 * signed_a))
    cy = float(np.sum((y + y_next) * cross) / (6.0 * signed_a))
    return cx, cy


def center_polygon(vertices: np.ndarray) -> np.ndarray:
    """Center a polygon so its centroid is at (0, 0)."""
    cx, cy = polygon_centroid(vertices)
    centered = vertices.copy()
    centered[:, 0] -= cx
    centered[:, 1] -= cy
    return centered


def normalize_polygon_area(vertices: np.ndarray, target_area: float = 0.05) -> np.ndarray:
    """
    Scale and center a polygon such that its Shoelace area equals target_area.

    Args:
        vertices: Array of shape (V, 2).
        target_area: Target area for the normalized shape.

    Returns:
        np.ndarray: Centered, area-normalized vertices of shape (V, 2).
    """
    centered = center_polygon(vertices)
    current_area = polygon_area(centered)
    if current_area < 1e-9:
        return centered
    scale_factor = np.sqrt(target_area / current_area)
    return centered * scale_factor


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """
    Check if line segment p1-p2 strictly intersects line segment p3-p4.
    """
    def ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = ccw(p1, p2, p3)
    d2 = ccw(p1, p2, p4)
    d3 = ccw(p3, p4, p1)
    d4 = ccw(p3, p4, p2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def is_simple_polygon(vertices: np.ndarray) -> bool:
    """
    Check if a polygon is simple (no non-adjacent edges intersect and no self-intersections).

    Args:
        vertices: Array of shape (V, 2).

    Returns:
        bool: True if simple, False otherwise.
    """
    n = len(vertices)
    if n < 3:
        return False

    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        # Compare with non-adjacent edges (j not i, i-1, i+1)
        for j in range(i + 2, n):
            if (i == 0 and j == n - 1):
                continue
            p3 = vertices[j]
            p4 = vertices[(j + 1) % n]
            if _segments_intersect(p1, p2, p3, p4):
                return False
    return True


def is_convex_polygon(vertices: np.ndarray) -> bool:
    """
    Check if a 2D polygon is strictly convex with consistent vertex ordering.

    Args:
        vertices: Array of shape (V, 2).

    Returns:
        bool: True if convex, False otherwise.
    """
    n = len(vertices)
    if n < 3:
        return False

    # Check cross products of consecutive edge vectors
    signs = []
    for i in range(n):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % n]
        p2 = vertices[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if np.abs(cross) > 1e-9:
            signs.append(cross > 0)

    if not signs:
        return False
    # All cross products must have the identical sign
    return all(s == signs[0] for s in signs)


def generate_convex_ngon(
    n: int,
    rng: Optional[np.random.Generator] = None,
    target_area: float = 0.05,
) -> np.ndarray:
    """
    Generate a strictly convex N-gon with exactly N vertices using Valtr's algorithm.

    Valtr's algorithm (Pavel Valtr, 1995) generates random convex polygons
    with a uniform distribution over convex polygons with N vertices.

    Args:
        n: Number of vertices (must be >= 3).
        rng: Optional numpy RandomGenerator.
        target_area: Target polygon area in normalized [-0.5, 0.5]^2 coords.

    Returns:
        np.ndarray: Array of shape (n, 2) centered at (0, 0).
    """
    if n < 3:
        raise ValueError(f"Convex polygon requires n >= 3, got {n}")
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(50):  # Retry loop in rare case of collinear degenerate points
        # 1. Generate n random coordinates in (0, 1) and sort them
        x_coords = np.sort(rng.uniform(0.0, 1.0, size=n))
        y_coords = np.sort(rng.uniform(0.0, 1.0, size=n))

        # 2. Divide intermediate points into two chains
        min_x, max_x = x_coords[0], x_coords[-1]
        min_y, max_y = y_coords[0], y_coords[-1]

        x_chain1, x_chain2 = [min_x], [min_x]
        for val in x_coords[1:-1]:
            if rng.random() < 0.5:
                x_chain1.append(val)
            else:
                x_chain2.append(val)
        x_chain1.append(max_x)
        x_chain2.append(max_x)

        y_chain1, y_chain2 = [min_y], [min_y]
        for val in y_coords[1:-1]:
            if rng.random() < 0.5:
                y_chain1.append(val)
            else:
                y_chain2.append(val)
        y_chain1.append(max_y)
        y_chain2.append(max_y)

        # 3. Compute vector steps: chain 1 forward, chain 2 backward
        x_steps = []
        for i in range(len(x_chain1) - 1):
            x_steps.append(x_chain1[i + 1] - x_chain1[i])
        for i in range(len(x_chain2) - 1):
            x_steps.append(x_chain2[i] - x_chain2[i + 1])

        y_steps = []
        for i in range(len(y_chain1) - 1):
            y_steps.append(y_chain1[i + 1] - y_chain1[i])
        for i in range(len(y_chain2) - 1):
            y_steps.append(y_chain2[i] - y_chain2[i + 1])

        # 4. Randomly pair x and y step components
        rng.shuffle(y_steps)
        vectors = np.column_stack((x_steps, y_steps))

        # 5. Sort vectors by their polar angle
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        sort_idx = np.argsort(angles)
        sorted_vectors = vectors[sort_idx]

        # 6. Cumulative sum to form closed polygon vertices
        vertices = np.cumsum(sorted_vectors, axis=0)

        # 7. Check convexity and validity
        if len(vertices) == n and is_convex_polygon(vertices) and polygon_area(vertices) > 1e-4:
            cand = normalize_polygon_area(vertices, target_area=target_area)
            if np.max(np.abs(cand)) <= 0.49:
                return cand

    # Fallback to perturbed convex N-gon if Valtr rejects
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=n))
    radii = rng.uniform(0.35, 0.45, size=n)
    vertices = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return normalize_polygon_area(vertices, target_area=target_area)


def generate_star(
    rng: Optional[np.random.Generator] = None,
    target_area: float = 0.05,
    outer_radius: float = 0.45,
    inner_outer_ratio: float = 0.381966,
) -> np.ndarray:
    """
    Generate a 5-pointed regular star (pentagram) with 10 alternating vertices.

    Outer radius R, inner radius r = R * (3 - sqrt(5)) / 2 ~ 0.382 R.

    Args:
        rng: Optional numpy RandomGenerator.
        target_area: Target area to normalize to.
        outer_radius: Nominal outer radius.
        inner_outer_ratio: Ratio of inner to outer radius.

    Returns:
        np.ndarray: Array of shape (10, 2) centered at (0, 0).
    """
    if rng is None:
        rng = np.random.default_rng()

    r_outer = outer_radius
    r_inner = outer_radius * inner_outer_ratio

    # 10 alternating vertices (5 outer tips, 5 inner valleys)
    # Starting at top (pi/2)
    start_angle = np.pi / 2.0
    angles = start_angle + np.arange(10) * (2 * np.pi / 10.0)

    radii = np.empty(10, dtype=np.float64)
    radii[0::2] = r_outer  # Even: tips
    radii[1::2] = r_inner  # Odd: inner vertices

    vertices = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return normalize_polygon_area(vertices, target_area=target_area)


def generate_basquiat_crown(
    rng: Optional[np.random.Generator] = None,
    target_area: float = 0.05,
) -> np.ndarray:
    """
    Generate an iconic Basquiat-style 3-peaked crown.

    Features:
      - Flat or slightly sloped base at the bottom
      - Vertical or subtly flaring outer sides
      - Left, center, and right peaks
      - Two deep V-shaped notches separating the three peaks
      - Organic variation in peak heights and notch depths

    Args:
        rng: Optional numpy RandomGenerator.
        target_area: Target area to normalize to.

    Returns:
        np.ndarray: Array of shape (7, 2) or (8, 2) centered at (0, 0).
    """
    if rng is None:
        rng = np.random.default_rng()

    base_w = rng.uniform(0.70, 0.85)
    base_y = -0.35 + rng.uniform(-0.02, 0.02)
    side_flare = rng.uniform(0.04, 0.12)

    # Peak heights (center peak typically prominent or slightly varied)
    peak_y_left = rng.uniform(0.28, 0.38)
    peak_y_center = rng.uniform(0.35, 0.45)
    peak_y_right = rng.uniform(0.28, 0.38)

    # Notch depths (deep V-notches)
    notch_y_left = base_y + rng.uniform(0.12, 0.22)
    notch_y_right = base_y + rng.uniform(0.12, 0.22)

    # Horizontal notch spacing
    notch_x_left = -base_w * rng.uniform(0.18, 0.26)
    notch_x_right = base_w * rng.uniform(0.18, 0.26)

    # Peak horizontal positions
    peak_x_left = -base_w / 2.0 - side_flare + rng.uniform(-0.02, 0.02)
    peak_x_center = rng.uniform(-0.03, 0.03)
    peak_x_right = base_w / 2.0 + side_flare + rng.uniform(-0.02, 0.02)

    # Counter-clockwise closed polygon path starting from bottom-left
    vertices = np.array([
        [-base_w / 2.0, base_y],                      # Bottom-left base
        [base_w / 2.0, base_y],                       # Bottom-right base
        [peak_x_right, peak_y_right],                 # Right peak
        [notch_x_right, notch_y_right],               # Right notch
        [peak_x_center, peak_y_center],               # Center peak
        [notch_x_left, notch_y_left],                 # Left notch
        [peak_x_left, peak_y_left],                   # Left peak
    ], dtype=np.float64)

    return normalize_polygon_area(vertices, target_area=target_area)


def generate_shape(
    class_id: int,
    rng: Optional[np.random.Generator] = None,
    target_area: float = 0.05,
) -> np.ndarray:
    """
    Generate normalized polygon vertices for a given class ID.

    Class Mapping:
      - 1: 5-pointed regular star
      - 2: Basquiat-style 3-peaked crown
      - N >= 3: Strictly convex N-gon

    Args:
        class_id: Integer class ID (must be >= 1).
        rng: Optional numpy RandomGenerator.
        target_area: Target area in normalized coords.

    Returns:
        np.ndarray: Array of shape (V, 2) centered at (0, 0).
    """
    if class_id < 1:
        raise ValueError(f"Foreground class_id must be >= 1, got {class_id}")
    if class_id == 1:
        return generate_star(rng=rng, target_area=target_area)
    elif class_id == 2:
        return generate_basquiat_crown(rng=rng, target_area=target_area)
    else:
        # Class N >= 3: convex polygon with N vertices
        return generate_convex_ngon(n=class_id, rng=rng, target_area=target_area)
