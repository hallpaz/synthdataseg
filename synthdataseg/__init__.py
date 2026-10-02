"""
synthdataseg: Procedural Synthetic Dataset Generator for Semantic Segmentation.
"""

from synthdataseg.config import (
    ClassRegistry,
    DatasetConfig,
    ShapeConfig,
    TextureConfig,
    TransformConfig,
    CompositorConfig,
    GenerationConfig,
    load_config,
    save_config,
)
from synthdataseg.shapes import (
    generate_shape,
    generate_convex_ngon,
    generate_star,
    generate_basquiat_crown,
    polygon_area,
    polygon_centroid,
    normalize_polygon_area,
    is_simple_polygon,
)
from synthdataseg.textures import (
    generate_texture,
    generate_plain_texture,
    generate_wood_texture,
    generate_marble_texture,
    generate_fur_texture,
    TextureType,
)
from synthdataseg.transforms import (
    AffineTransform,
    compute_affine_matrix,
    apply_affine_transform,
    compute_area_normalized_scaling,
)
from synthdataseg.compositor import (
    SyntheticCompositor,
    CompositeResult,
    StratifiedClassSampler,
)
from synthdataseg.dataset import (
    SyntheticSegmentationDataset,
)

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in ("generate_dataset", "generate_sample"):
        from synthdataseg.generate import generate_dataset, generate_sample
        mapping = {"generate_dataset": generate_dataset, "generate_sample": generate_sample}
        return mapping[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "ClassRegistry",
    "DatasetConfig",
    "ShapeConfig",
    "TextureConfig",
    "TransformConfig",
    "CompositorConfig",
    "GenerationConfig",
    "load_config",
    "save_config",
    "generate_shape",
    "generate_convex_ngon",
    "generate_star",
    "generate_basquiat_crown",
    "polygon_area",
    "polygon_centroid",
    "normalize_polygon_area",
    "is_simple_polygon",
    "generate_texture",
    "generate_plain_texture",
    "generate_wood_texture",
    "generate_marble_texture",
    "generate_fur_texture",
    "TextureType",
    "AffineTransform",
    "compute_affine_matrix",
    "apply_affine_transform",
    "compute_area_normalized_scaling",
    "SyntheticCompositor",
    "CompositeResult",
    "StratifiedClassSampler",
    "SyntheticSegmentationDataset",
    "generate_dataset",
    "generate_sample",
]
