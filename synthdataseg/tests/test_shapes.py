"""
Unit tests for vector shape validity, convexity, area normalization, and class registry.
"""

import numpy as np
import pytest

from synthdataseg.config import ClassRegistry
from synthdataseg.shapes import (
    generate_shape,
    generate_star,
    generate_basquiat_crown,
    generate_convex_ngon,
    polygon_area,
    polygon_centroid,
    is_simple_polygon,
    is_convex_polygon,
    normalize_polygon_area,
)
from synthdataseg.transforms import (
    sample_affine_transform,
    apply_affine_transform,
    compute_affine_matrix,
    compute_area_normalized_scaling,
)


def test_class_registry_mapping():
    """Verify class registry mappings for 0, 1, 2, and N >= 3."""
    reg = ClassRegistry(max_n=8)
    assert reg.num_classes == 9
    assert reg.get_class_name(0) == "background"
    assert reg.get_class_name(1) == "star_5"
    assert reg.get_class_name(2) == "crown_basquiat"
    assert reg.get_class_name(3) == "triangle"
    assert reg.get_class_name(4) == "quadrilateral"
    assert reg.get_class_name(5) == "pentagon"
    assert reg.get_class_name(6) == "hexagon"
    assert reg.get_class_name(7) == "heptagon"
    assert reg.get_class_name(8) == "octagon"

    # Out of bounds
    with pytest.raises(ValueError):
        reg.get_class_name(9)


def test_star_shape_validity():
    """Verify 5-pointed regular star has 10 vertices, is non-self-intersecting, and centered."""
    rng = np.random.default_rng(101)
    star = generate_star(rng=rng, target_area=0.05)
    assert len(star) == 10
    assert is_simple_polygon(star), "Star polygon must not self-intersect"

    # Centered at (0, 0)
    cx, cy = polygon_centroid(star)
    assert np.isclose(cx, 0.0, atol=1e-3)
    assert np.isclose(cy, 0.0, atol=1e-3)

    # Area normalized
    assert np.isclose(polygon_area(star), 0.05, rtol=1e-2)


def test_basquiat_crown_shape_validity():
    """Verify Basquiat crown is non-self-intersecting, has 3 peaks, and is centered."""
    rng = np.random.default_rng(202)
    crown = generate_basquiat_crown(rng=rng, target_area=0.05)
    assert len(crown) >= 7
    assert is_simple_polygon(crown), "Basquiat crown polygon must not self-intersect"

    # Centered at (0, 0)
    cx, cy = polygon_centroid(crown)
    assert np.isclose(cx, 0.0, atol=1e-3)
    assert np.isclose(cy, 0.0, atol=1e-3)

    # Area normalized
    assert np.isclose(polygon_area(crown), 0.05, rtol=1e-2)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8, 9, 10])
def test_convex_ngons_strict_convexity(n):
    """Verify Valtr's algorithm produces strictly convex polygons with exactly N vertices."""
    rng = np.random.default_rng(300 + n)
    for _ in range(10):
        ngon = generate_convex_ngon(n=n, rng=rng, target_area=0.05)
        assert len(ngon) == n, f"Expected {n} vertices, got {len(ngon)}"
        assert is_convex_polygon(ngon), f"Polygon with n={n} must be strictly convex"
        assert is_simple_polygon(ngon), f"Polygon with n={n} must not self-intersect"

        cx, cy = polygon_centroid(ngon)
        assert np.isclose(cx, 0.0, atol=1e-3)
        assert np.isclose(cy, 0.0, atol=1e-3)
        assert np.isclose(polygon_area(ngon), 0.05, rtol=1e-2)


def test_area_normalized_scaling_under_affine_transform():
    """Verify affine transform scales shapes to target pixel area regardless of shear or aspect ratio."""
    rng = np.random.default_rng(404)
    canvas_size = (256, 256)
    target_pixel_area = 5000.0

    for class_id in [1, 2, 3, 5, 8]:
        shape = generate_shape(class_id, rng=rng)
        tf = sample_affine_transform(
            shape,
            target_area=target_pixel_area,
            canvas_size=canvas_size,
            rng=rng,
            shear_max=0.25,
            aspect_ratio_range=(0.8, 1.25),
        )
        transformed = apply_affine_transform(shape, tf.matrix)
        actual_area = polygon_area(transformed)
        assert np.isclose(actual_area, target_pixel_area, rtol=1e-3), (
            f"Class {class_id}: expected area {target_pixel_area}, got {actual_area}"
        )
