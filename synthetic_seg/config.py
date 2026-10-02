"""
Configuration and Class Registry module for synthetic segmentation.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import yaml
import numpy as np


class ClassRegistry:
    """
    Extensible class registry for synthetic semantic segmentation.

    Class Mapping:
      - Class 0: Background
      - Class 1: 5-pointed regular star (pentagram)
      - Class 2: Basquiat-style 3-peaked crown
      - Class N (N >= 3): Convex polygon with N vertices (Class 3 = Triangle, ..., Class N_max)
    """

    def __init__(self, max_n: int = 8) -> None:
        """
        Initialize the class registry.

        Args:
            max_n: Maximum number of vertices for convex N-gons (must be >= 3).
        """
        if max_n < 3:
            raise ValueError(f"max_n must be >= 3, got {max_n}")
        self.max_n = max_n
        self._classes: Dict[int, str] = {
            0: "background",
            1: "star_5",
            2: "crown_basquiat",
        }
        for n in range(3, self.max_n + 1):
            if n == 3:
                name = "triangle"
            elif n == 4:
                name = "quadrilateral"
            elif n == 5:
                name = "pentagon"
            elif n == 6:
                name = "hexagon"
            elif n == 7:
                name = "heptagon"
            elif n == 8:
                name = "octagon"
            elif n == 9:
                name = "nonagon"
            elif n == 10:
                name = "decagon"
            else:
                name = f"convex_{n}gon"
            self._classes[n] = name

        self._color_palette = self._generate_palette()

    @property
    def num_classes(self) -> int:
        """Total number of classes including background."""
        return len(self._classes)

    @property
    def max_class_id(self) -> int:
        """Maximum class index (N_max)."""
        return self.max_n

    def get_class_name(self, class_id: int) -> str:
        """Return the descriptive name for a given class ID."""
        if class_id not in self._classes:
            raise ValueError(f"Class ID {class_id} not in registry [0, {self.max_n}]")
        return self._classes[class_id]

    def get_class_ids(self) -> List[int]:
        """Return list of all registered class IDs including background."""
        return sorted(list(self._classes.keys()))

    def get_foreground_class_ids(self) -> List[int]:
        """Return list of foreground class IDs (1 to N_max)."""
        return [cid for cid in self.get_class_ids() if cid != 0]

    def _generate_palette(self) -> np.ndarray:
        """
        Generate distinct RGB colors for each class ID.
        Class 0 is dark gray/black. Other classes have high-contrast distinct hues.
        """
        palette = np.zeros((self.num_classes, 3), dtype=np.uint8)
        # Background: dark charcoal
        palette[0] = [25, 25, 28]

        # Preset vibrant colors for standard classes
        preset_colors = [
            [255, 215, 0],   # Class 1 (Star): Vivid Gold
            [220, 20, 60],   # Class 2 (Basquiat Crown): Crimson Red
            [30, 144, 255],  # Class 3 (Triangle): Dodger Blue
            [50, 205, 50],   # Class 4 (Quadrilateral): Lime Green
            [255, 127, 80],  # Class 5 (Pentagon): Coral Orange
            [138, 43, 226],  # Class 6 (Hexagon): Blue Violet
            [0, 206, 209],   # Class 7 (Heptagon): Dark Turquoise
            [255, 105, 180], # Class 8 (Octagon): Hot Pink
            [244, 164, 96],  # Class 9 (Nonagon): Sandy Brown
            [0, 250, 154],   # Class 10 (Decagon): Medium Spring Green
        ]

        for cid in range(1, self.num_classes):
            idx = cid - 1
            if idx < len(preset_colors):
                palette[cid] = preset_colors[idx]
            else:
                # Golden ratio hue distribution for extended classes
                hue = (idx * 0.618033988749895) % 1.0
                # Convert HSV (hue, 0.85, 0.95) to RGB
                import colorsys
                r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
                palette[cid] = [int(r * 255), int(g * 255), int(b * 255)]

        return palette

    def get_color_palette(self) -> np.ndarray:
        """Return the color palette array of shape (num_classes, 3)."""
        return self._color_palette.copy()

    def colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert an integer class mask of shape (H, W) into an RGB image (H, W, 3).
        """
        h, w = mask.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        for cid in self.get_class_ids():
            color_img[mask == cid] = self._color_palette[cid]
        return color_img


@dataclass
class ShapeConfig:
    """Configuration for shape geometry generation."""
    target_area_normalized: float = 0.05  # Base normalized area in canonical coords
    star_inner_outer_ratio: float = 0.381966  # (3 - sqrt(5)) / 2 ~ 0.382
    crown_base_width: float = 0.8
    crown_height: float = 0.7


@dataclass
class TextureConfig:
    """Configuration for procedural texture synthesizers."""
    types: List[str] = field(default_factory=lambda: ["plain", "wood", "marble", "fur"])
    perlin_grid_size: int = 8
    fbm_octaves: int = 4
    fbm_persistence: float = 0.5
    wood_frequency: float = 24.0
    wood_turbulence: float = 8.0
    marble_frequency: Tuple[float, float] = (12.0, 12.0)
    marble_turbulence: float = 6.0
    fur_frequency: float = 40.0
    fur_stretch_ratio: float = 6.0


@dataclass
class TransformConfig:
    """Configuration for 2D Affine transformations."""
    rotation_min: float = 0.0
    rotation_max: float = 2 * np.pi
    shear_max: float = 0.25
    aspect_ratio_min: float = 0.75
    aspect_ratio_max: float = 1.33


@dataclass
class CompositorConfig:
    """Configuration for multi-shape canvas compositing."""
    image_size: Tuple[int, int] = (256, 256)
    min_shapes: int = 2
    max_shapes: int = 5
    target_coverage_min: float = 0.20
    target_coverage_max: float = 0.45
    bg_dir: Optional[str] = None
    stratified_sampling: bool = True


@dataclass
class GenerationConfig:
    """Configuration for dataset generation CLI and splits."""
    splits: Dict[str, int] = field(default_factory=lambda: {"train": 100, "val": 20, "test": 20})
    output_dir: str = "./data/synthetic_dataset"
    seed: Optional[int] = 42
    tolerance_threshold: float = 0.15  # Maximum +/- 15% class pixel deviation from target
    max_n: int = 8
    image_size: Tuple[int, int] = (256, 256)
    save_format: str = "png"  # "png" for image and mask


@dataclass
class DatasetConfig:
    """Master configuration combining all sub-configurations."""
    max_n: int = 8
    shapes: ShapeConfig = field(default_factory=ShapeConfig)
    textures: TextureConfig = field(default_factory=TextureConfig)
    transforms: TransformConfig = field(default_factory=TransformConfig)
    compositor: CompositorConfig = field(default_factory=CompositorConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass hierarchy to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetConfig:
        """Construct DatasetConfig from dictionary."""
        max_n = data.get("max_n", 8)
        shapes_data = data.get("shapes", {})
        textures_data = data.get("textures", {})
        transforms_data = data.get("transforms", {})
        compositor_data = data.get("compositor", {})
        generation_data = data.get("generation", {})

        # Convert tuples if loaded as lists from YAML
        if "marble_frequency" in textures_data and isinstance(textures_data["marble_frequency"], list):
            textures_data["marble_frequency"] = tuple(textures_data["marble_frequency"])
        if "image_size" in compositor_data and isinstance(compositor_data["image_size"], list):
            compositor_data["image_size"] = tuple(compositor_data["image_size"])
        if "image_size" in generation_data and isinstance(generation_data["image_size"], list):
            generation_data["image_size"] = tuple(generation_data["image_size"])

        return cls(
            max_n=max_n,
            shapes=ShapeConfig(**shapes_data),
            textures=TextureConfig(**textures_data),
            transforms=TransformConfig(**transforms_data),
            compositor=CompositorConfig(**compositor_data),
            generation=GenerationConfig(**generation_data),
        )


def load_config(path: Union[str, Path]) -> DatasetConfig:
    """Load configuration from a YAML file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return DatasetConfig.from_dict(data or {})


def save_config(config: DatasetConfig, path: Union[str, Path]) -> None:
    """Save configuration to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def get_default_config() -> DatasetConfig:
    """Get default DatasetConfig instance."""
    return DatasetConfig()
