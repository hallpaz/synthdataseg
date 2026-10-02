"""
Canvas Compositor and Multi-Shape Painter's Algorithm Engine.

Handles:
  1. Real or procedural background loading and aspect-ratio cropping
  2. Stratified class sampling with global/batch class frequency balancing
  3. Controlled foreground coverage fraction (20% to 45%)
  4. Multi-layer Painter's Algorithm rendering with integer segmentation masks
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image, ImageDraw

from synthdataseg.config import ClassRegistry, CompositorConfig, TextureConfig
from synthdataseg.shapes import generate_shape, polygon_area
from synthdataseg.textures import generate_texture, PerlinNoise2D
from synthdataseg.transforms import sample_affine_transform, apply_affine_transform


@dataclass
class ShapeLayerInfo:
    """Metadata for an individual rendered shape layer."""
    class_id: int
    class_name: str
    texture_type: str
    target_area: float
    actual_area: float
    visible_pixels: int
    vertices: np.ndarray


@dataclass
class CompositeResult:
    """Output of composite rendering."""
    image: np.ndarray             # (H, W, 3) uint8
    mask: np.ndarray              # (H, W) int64
    class_counts: Dict[int, int]  # Pixel count per class
    instance_counts: Dict[int, int]  # Placed instance count per class
    foreground_coverage: float    # Fraction of canvas occupied by foreground
    layers: List[ShapeLayerInfo] = field(default_factory=list)


class StratifiedClassSampler:
    """
    Maintains class frequency tracker to ensure balanced class distributions
    across images in a dataset.
    """

    def __init__(self, class_registry: ClassRegistry) -> None:
        self.registry = class_registry
        self.foreground_classes = self.registry.get_foreground_class_ids()
        # Frequency counts: track instances and total visible pixels
        self.instance_counts: Dict[int, int] = {cid: 0 for cid in self.foreground_classes}
        self.pixel_counts: Dict[int, int] = {cid: 0 for cid in self.foreground_classes}

    def sample_classes(self, k: int, rng: Optional[np.random.Generator] = None) -> List[int]:
        """
        Sample K foreground class IDs, giving higher probability to underrepresented classes.

        Args:
            k: Number of classes to sample.
            rng: Optional numpy RandomGenerator.

        Returns:
            List[int]: Sampled class IDs.
        """
        if rng is None:
            rng = np.random.default_rng()

        classes = np.array(self.foreground_classes)
        # Compute weights inversely proportional to instance count
        counts = np.array([self.instance_counts[cid] for cid in classes], dtype=np.float64)
        min_count = np.min(counts)

        # Softmax-like or inverse frequency weighting
        # Temperature scaled so underrepresented classes are strongly favored
        inv_counts = 1.0 / (counts - min_count + 1.0)
        probs = inv_counts / np.sum(inv_counts)

        # Sample without replacement if k <= len(classes), else with replacement
        replace = k > len(classes)
        chosen = rng.choice(classes, size=k, replace=replace, p=probs)
        return [int(c) for c in chosen]

    def record_placed(self, class_id: int, visible_pixels: int) -> None:
        """Update tracker with placed instance and its visible pixels."""
        if class_id in self.instance_counts:
            self.instance_counts[class_id] += 1
            self.pixel_counts[class_id] += visible_pixels

    def reset(self) -> None:
        """Reset frequency tracking counts."""
        self.instance_counts = {cid: 0 for cid in self.foreground_classes}
        self.pixel_counts = {cid: 0 for cid in self.foreground_classes}

    def get_instance_distribution(self) -> Dict[int, int]:
        """Return copy of instance counts."""
        return dict(self.instance_counts)

    def get_pixel_distribution(self) -> Dict[int, int]:
        """Return copy of pixel counts."""
        return dict(self.pixel_counts)


class SyntheticCompositor:
    """
    Multi-shape layer composer & ground truth mask painter.
    """

    def __init__(
        self,
        registry: Optional[ClassRegistry] = None,
        config: Optional[CompositorConfig] = None,
        texture_config: Optional[TextureConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config or CompositorConfig()
        self.registry = registry or ClassRegistry()
        self.texture_config = texture_config or TextureConfig()
        self.rng = rng or np.random.default_rng()
        self.sampler = StratifiedClassSampler(self.registry)
        self.bg_paths: List[Path] = []
        self.bg_files: List[Path] = []
        self._scan_bg_dir()

    def _scan_bg_dir(self) -> None:
        """Scan background directory for image files if specified."""
        if self.config.bg_dir:
            bg_dir = Path(self.config.bg_dir)
            if bg_dir.exists() and bg_dir.is_dir():
                extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                self.bg_paths = sorted(
                    [p for p in Path(bg_dir).iterdir() if p.suffix.lower() in extensions],
                    key=lambda p: p.name.lower()
                )
                self.bg_files = self.bg_paths

    def generate_procedural_background(
        self, height: int, width: int, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Generate high quality procedural backdrop when no background images are provided.

        Styles:
          - Gradient studio backdrops
          - Subtle textured architectural surfaces / smooth noise
        """
        if rng is None:
            rng = self.rng
        noise = PerlinNoise2D(seed=int(rng.integers(1, 100000)))
        ys = np.linspace(0.0, 1.0, height)
        xs = np.linspace(0.0, 1.0, width)
        x_grid, y_grid = np.meshgrid(xs, ys)

        # Style 1: Ambient Studio Gradient (smooth vignette / subtle wall)
        col1 = rng.uniform(40, 120, size=3)
        col2 = rng.uniform(160, 230, size=3)
        vignette = np.sqrt((x_grid - 0.5) ** 2 + (y_grid - 0.5) ** 2) * 1.4
        vignette = np.clip(vignette, 0.0, 1.0)[..., None]

        # Subtle noise texture
        n = noise.fbm(x_grid * 3.0, y_grid * 3.0, octaves=3) * 15.0
        bg = (1.0 - vignette) * col2 + vignette * col1 + n[..., None]
        return np.clip(bg, 0, 255).astype(np.uint8)

    def load_background(
        self, height: int, width: int, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Load a random image from background directory or generate procedural backdrop.
        """
        if rng is None:
            rng = self.rng

        if self.bg_paths:
            try:
                bg_path = self.bg_paths[int(rng.integers(len(self.bg_paths)))]
                with Image.open(bg_path) as img:
                    img = img.convert("RGB")
                    # Aspect ratio preserving crop and resize
                    iw, ih = img.size
                    scale = max(width / iw, height / ih)
                    nw, nh = int(round(iw * scale)), int(round(ih * scale))
                    img = img.resize((nw, nh), Image.Resampling.BILINEAR)
                    # Center crop or random crop
                    x_off = int(rng.integers(0, nw - width + 1))
                    y_off = int(rng.integers(0, nh - height + 1))
                    cropped = img.crop((x_off, y_off, x_off + width, y_off + height))
                    return np.array(cropped, dtype=np.uint8)
            except Exception:
                pass  # Fall back to procedural background on read error

        return self.generate_procedural_background(height, width, rng=rng)

    def render(
        self,
        rng: Optional[np.random.Generator] = None,
        track_history: bool = False,
    ) -> CompositeResult:
        """
        Render a full composite image and corresponding segmentation mask.

        Steps:
          1. Load background
          2. Sample K shapes via StratifiedClassSampler
          3. Distribute target area to achieve target foreground coverage (20%-45%)
          4. Sequentially render each shape from back to front (Painter's algorithm)
          5. Update frequency tracker and compute statistics
        """
        if rng is None:
            rng = self.rng

        height, width = self.config.image_size
        canvas_rgb = self.load_background(height, width, rng=rng)
        canvas_mask = np.zeros((height, width), dtype=np.int64)

        # 1. Sample number of shapes K
        k = int(rng.integers(self.config.min_shapes, self.config.max_shapes + 1))

        # 2. Sample classes with stratified balancing
        if self.config.stratified_sampling:
            class_ids = self.sampler.sample_classes(k, rng=rng)
        else:
            fg_classes = self.registry.get_foreground_class_ids()
            class_ids = [int(c) for c in rng.choice(fg_classes, size=k)]

        # 3. Target total foreground coverage fraction
        total_pixels = height * width
        target_coverage = rng.uniform(
            self.config.target_coverage_min, self.config.target_coverage_max
        )
        total_fg_pixels = target_coverage * total_pixels

        # Partition total foreground area among K shapes (with slight variation)
        weights = rng.uniform(0.7, 1.3, size=k)
        weights /= np.sum(weights)

        # Compensate for expected occlusion (~15-25% overlap factor)
        occlusion_factor = 1.15
        shape_target_areas = weights * total_fg_pixels * occlusion_factor

        # 4. Sequentially render each shape (Painter's algorithm: back to front)
        layer_infos: List[ShapeLayerInfo] = []
        texture_types = self.texture_config.types

        for i, cid in enumerate(class_ids):
            target_area = shape_target_areas[i]
            # Generate base normalized shape centered at (0, 0)
            base_shape = generate_shape(cid, rng=rng)
            base_area = polygon_area(base_shape)

            # Sample affine transformation with area normalization
            transform = sample_affine_transform(
                base_shape,
                target_area=target_area,
                canvas_size=(height, width),
                rng=rng,
                shear_max=0.20,
                aspect_ratio_range=(0.85, 1.20),
            )
            v_canvas = apply_affine_transform(base_shape, transform.matrix)
            actual_area = polygon_area(v_canvas)

            # Rasterize polygon mask into a binary PIL buffer
            poly_mask_img = Image.new("L", (width, height), 0)
            poly_draw = ImageDraw.Draw(poly_mask_img)
            # Explicit integer casting for pixel-exact rasterization
            coords = np.round(v_canvas).astype(np.int32)
            poly_draw.polygon([tuple(pt) for pt in coords], fill=255)
            poly_mask = np.array(poly_mask_img, dtype=bool)

            # Synthesize texture for this shape
            tt = str(rng.choice(texture_types))
            # Determine bounding box to synthesize texture efficiently
            ys, xs = np.where(poly_mask)
            if len(ys) == 0:
                continue

            ymin, ymax = int(np.min(ys)), int(np.max(ys)) + 1
            xmin, xmax = int(np.min(xs)), int(np.max(xs)) + 1
            bbox_h = ymax - ymin
            bbox_w = xmax - xmin

            texture_patch = generate_texture(
                tt, bbox_h, bbox_w, rng=rng, config=self.texture_config
            )

            # Overwrite RGB and Mask buffer within the polygon footprint
            patch_mask = poly_mask[ymin:ymax, xmin:xmax]
            canvas_rgb[ymin:ymax, xmin:xmax][patch_mask] = texture_patch[patch_mask]
            canvas_mask[ymin:ymax, xmin:xmax][patch_mask] = cid

            layer_infos.append(ShapeLayerInfo(
                class_id=cid,
                class_name=self.registry.get_class_name(cid),
                texture_type=tt,
                target_area=float(target_area),
                actual_area=float(actual_area),
                visible_pixels=0,  # Computed after all layers are painted
                vertices=v_canvas,
            ))

        # 5. Compute final visible pixels per class and update sampler
        class_counts: Dict[int, int] = {cid: 0 for cid in self.registry.get_class_ids()}
        unique_classes, counts = np.unique(canvas_mask, return_counts=True)
        for u_cid, count in zip(unique_classes, counts):
            class_counts[int(u_cid)] = int(count)

        instance_counts: Dict[int, int] = {cid: 0 for cid in self.registry.get_foreground_class_ids()}
        for layer in layer_infos:
            instance_counts[layer.class_id] += 1
            layer.visible_pixels = class_counts.get(layer.class_id, 0)
            if track_history:
                self.sampler.record_placed(layer.class_id, layer.visible_pixels)

        fg_pixels = sum(class_counts[cid] for cid in self.registry.get_foreground_class_ids())
        foreground_coverage = float(fg_pixels / total_pixels)

        return CompositeResult(
            image=canvas_rgb,
            mask=canvas_mask,
            class_counts=class_counts,
            instance_counts=instance_counts,
            foreground_coverage=foreground_coverage,
            layers=layer_infos,
        )
