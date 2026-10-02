"""
2D Affine Transformation Engine with Area-Normalized Shape Scaling.

Implements 3x3 homogeneous affine transformations:
  M = T(dx, dy) * R(theta) * Sh(sxy, syx) * Sc(sx, sy)

Ensures exact target polygon area scaling regardless of shape geometry,
rotation, shear, or aspect ratio jitter.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from synthdataseg.shapes import polygon_area, center_polygon


@dataclass
class AffineTransform:
    """Parameters representing a 2D affine transformation."""
    translation: Tuple[float, float]
    rotation: float
    shear: Tuple[float, float]
    scale: Tuple[float, float]
    matrix: np.ndarray


def compute_area_normalized_scaling(
    base_area: float,
    target_area: float,
    shear: Tuple[float, float],
    aspect_ratio: float,
) -> Tuple[float, float]:
    """
    Compute (sx, sy) such that the transformed polygon area matches target_area.

    The linear transformation determinant is:
      |det(R * Sh * Sc)| = sx * sy * |1 - sxy * syx|
    Given aspect_ratio rho = sy / sx:
      Area_final = Area_base * sx^2 * rho * |1 - sxy * syx| = target_area

    Args:
        base_area: Shoelace area of base shape.
        target_area: Desired pixel area on image canvas.
        shear: (sxy, syx) shear factors.
        aspect_ratio: rho = sy / sx aspect ratio jitter.

    Returns:
        Tuple[float, float]: (sx, sy) scaling factors.
    """
    if base_area <= 1e-9:
        raise ValueError(f"base_area must be positive, got {base_area}")
    if target_area <= 1e-9:
        raise ValueError(f"target_area must be positive, got {target_area}")

    sxy, syx = shear
    det_shear = np.abs(1.0 - sxy * syx)
    if det_shear < 0.1:
        det_shear = 0.1

    sx = np.sqrt(target_area / (base_area * aspect_ratio * det_shear))
    sy = sx * aspect_ratio
    return float(sx), float(sy)


def compute_affine_matrix(
    translation: Tuple[float, float],
    rotation: float,
    shear: Tuple[float, float],
    scale: Tuple[float, float],
) -> np.ndarray:
    """
    Construct 3x3 homogeneous affine transformation matrix:
      M = T(dx, dy) @ R(theta) @ Sh(sxy, syx) @ Sc(sx, sy)

    Args:
        translation: (dx, dy) translation offsets.
        rotation: theta rotation in radians.
        shear: (sxy, syx) shear factors.
        scale: (sx, sy) scale factors.

    Returns:
        np.ndarray: 3x3 transformation matrix.
    """
    dx, dy = translation
    theta = rotation
    sxy, syx = shear
    sx, sy = scale

    # 1. Scale matrix Sc
    sc_mat = np.array([
        [sx,  0.0, 0.0],
        [0.0, sy,  0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # 2. Shear matrix Sh
    sh_mat = np.array([
        [1.0, sxy, 0.0],
        [syx, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # 3. Rotation matrix R
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    r_mat = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0]
    ], dtype=np.float64)

    # 4. Translation matrix T
    t_mat = np.array([
        [1.0, 0.0, dx],
        [0.0, 1.0, dy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Composite: M = T @ R @ Sh @ Sc
    return t_mat @ r_mat @ sh_mat @ sc_mat


def apply_affine_transform(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply 3x3 affine transformation matrix to 2D polygon vertices.

    Args:
        vertices: (V, 2) array of 2D vertices.
        matrix: 3x3 transformation matrix.

    Returns:
        np.ndarray: (V, 2) transformed vertices.
    """
    v_homo = np.column_stack([vertices, np.ones(len(vertices), dtype=np.float64)])
    transformed = (matrix @ v_homo.T).T
    return transformed[:, :2]


def sample_affine_transform(
    vertices: np.ndarray,
    target_area: float,
    canvas_size: Tuple[int, int],
    rng: Optional[np.random.Generator] = None,
    shear_max: float = 0.25,
    aspect_ratio_range: Tuple[float, float] = (0.8, 1.25),
    margin_ratio: float = 0.08,
) -> AffineTransform:
    """
    Sample a random valid affine transformation placing the shape onto the canvas
    with exact area normalization.

    Args:
        vertices: (V, 2) normalized shape vertices centered at (0, 0).
        target_area: Desired pixel area of the shape on the canvas.
        canvas_size: (height, width) of canvas in pixels.
        rng: Optional numpy RandomGenerator.
        shear_max: Maximum shear magnitude.
        aspect_ratio_range: (min_ratio, max_ratio) for aspect ratio jitter.
        margin_ratio: Margin fraction of canvas to keep shape centers well within bounds.

    Returns:
        AffineTransform containing all parameters and the 3x3 matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    height, width = canvas_size
    base_area = polygon_area(vertices)

    # 1. Sample shear
    sxy = rng.uniform(-shear_max, shear_max)
    syx = rng.uniform(-shear_max, shear_max)
    shear = (sxy, syx)

    # 2. Sample aspect ratio jitter rho = sy / sx
    aspect_ratio = rng.uniform(aspect_ratio_range[0], aspect_ratio_range[1])

    # 3. Compute area-normalized scale
    sx, sy = compute_area_normalized_scaling(base_area, target_area, shear, aspect_ratio)
    scale = (sx, sy)

    # 4. Sample rotation theta in [0, 2pi)
    rotation = rng.uniform(0.0, 2 * np.pi)

    # 5. Measure centered extent to place shape cleanly on canvas
    mat_centered = compute_affine_matrix((0.0, 0.0), rotation, shear, scale)
    v_centered = apply_affine_transform(vertices, mat_centered)
    min_x, max_x = np.min(v_centered[:, 0]), np.max(v_centered[:, 0])
    min_y, max_y = np.min(v_centered[:, 1]), np.max(v_centered[:, 1])

    w_shape = max_x - min_x
    h_shape = max_y - min_y

    # Translation range keeping shape predominantly on canvas
    margin_x = max(10.0, width * margin_ratio)
    margin_y = max(10.0, height * margin_ratio)

    low_x = min(width / 2.0, max(margin_x, w_shape / 2.0))
    high_x = max(width / 2.0, width - max(margin_x, w_shape / 2.0))

    low_y = min(height / 2.0, max(margin_y, h_shape / 2.0))
    high_y = max(height / 2.0, height - max(margin_y, h_shape / 2.0))

    dx = rng.uniform(low_x, high_x)
    dy = rng.uniform(low_y, high_y)
    translation = (dx, dy)

    matrix = compute_affine_matrix(translation, rotation, shear, scale)

    return AffineTransform(
        translation=translation,
        rotation=rotation,
        shear=shear,
        scale=scale,
        matrix=matrix,
    )
