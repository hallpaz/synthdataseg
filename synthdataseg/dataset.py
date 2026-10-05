"""
PyTorch Dataset implementation for synthetic semantic segmentation.

Supports:
  1. On-the-fly procedural generation in memory per __getitem__ call
  2. Disk-backed loading from pre-generated dataset splits
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, List
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from synthdataseg.config import ClassRegistry, CompositorConfig, TextureConfig
from synthdataseg.compositor import SyntheticCompositor


# Standard ImageNet normalization statistics
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SyntheticSegmentationDataset(Dataset):
    """
    PyTorch Dataset for procedural synthetic segmentation.

    Operating Modes:
      - 'on_the_fly' (or 'online'): Dynamically generates image/mask pairs in memory.
      - 'disk' (or 'disk_backed'): Loads paired images and masks from disk.
    """

    def __init__(
        self,
        mode: str = "on_the_fly",
        length: int = 100,
        data_dir: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        normalize_mode: str = "unit",  # "unit" for [0, 1], "imagenet" for ImageNet normalization
        min_shapes: Optional[int] = None,
        max_shapes: Optional[int] = None,
        registry: Optional[ClassRegistry] = None,
        compositor_config: Optional[CompositorConfig] = None,
        texture_config: Optional[TextureConfig] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        num_samples: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        max_n: Optional[int] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            mode: 'on_the_fly' or 'disk'.
            length: Virtual length of the dataset in 'on_the_fly' mode.
            data_dir: Root directory of generated dataset for 'disk' mode.
            split: Dataset split ('train', 'val', 'test') if data_dir contains split subfolders.
            normalize_mode: 'unit' for [0.0, 1.0] float tensor, 'imagenet' for ImageNet mean/std.
            registry: ClassRegistry instance.
            compositor_config: CompositorConfig for on-the-fly rendering.
            texture_config: TextureConfig for on-the-fly rendering.
            transform: Optional transformation for image tensor.
            target_transform: Optional transformation for mask tensor.
            seed: Optional seed for reproducible generation.
            num_samples: Alias for length.
            image_size: Convenience tuple (H, W) to configure canvas size.
            max_n: Convenience integer to configure max N-gon class in ClassRegistry.
        """
        self.mode = mode.lower()
        if self.mode not in {"on_the_fly", "online", "disk", "disk_backed"}:
            raise ValueError(f"Invalid mode: '{mode}'. Must be 'on_the_fly' or 'disk'.")

        if num_samples is not None:
            length = num_samples

        if registry is None and max_n is not None:
            registry = ClassRegistry(max_n=max_n)

        if image_size is not None:
            if compositor_config is None:
                compositor_config = CompositorConfig(image_size=image_size)
            else:
                compositor_config.image_size = image_size
        if min_shapes is not None:
            compositor_config.min_shapes = min_shapes
        if max_shapes is not None:
            compositor_config.max_shapes = max_shapes

        self.length = length
        self.normalize_mode = normalize_mode
        self.registry = registry or ClassRegistry()
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed

        if self.mode in {"on_the_fly", "online"}:
            self.compositor = SyntheticCompositor(
                registry=self.registry,
                config=compositor_config,
                texture_config=texture_config,
                rng=np.random.default_rng(seed) if seed is not None else None,
            )
            self.image_files: List[Path] = []
            self.mask_files: List[Path] = []
        else:
            # Disk mode: discover files
            if data_dir is None:
                raise ValueError("data_dir must be provided when mode='disk'")

            base_path = Path(data_dir)
            if split:
                base_path = base_path / split

            img_dir = base_path / "images"
            mask_dir = base_path / "masks"

            if not img_dir.exists() or not mask_dir.exists():
                raise FileNotFoundError(
                    f"Expected 'images' and 'masks' directories inside {base_path}"
                )

            # Discover image files sorted deterministically by filename
            valid_exts = {".png", ".jpg", ".jpeg"}
            self.image_files = sorted(
                [p for p in img_dir.iterdir() if p.suffix.lower() in valid_exts],
                key=lambda p: p.name.lower()
            )
            self.mask_files = []

            for img_p in self.image_files:
                # Mask might be .png or .npy
                mask_p_png = mask_dir / f"{img_p.stem}.png"
                mask_p_npy = mask_dir / f"{img_p.stem}.npy"
                if mask_p_png.exists():
                    self.mask_files.append(mask_p_png)
                elif mask_p_npy.exists():
                    self.mask_files.append(mask_p_npy)
                else:
                    raise FileNotFoundError(f"Missing mask for image: {img_p.name} in {mask_dir}")

            self.length = len(self.image_files)
            self.compositor = None

    def __len__(self) -> int:
        return self.length

    def _normalize_image(self, img_tensor: torch.FloatTensor) -> torch.FloatTensor:
        """Normalize (3, H, W) tensor according to normalize_mode."""
        if self.normalize_mode == "imagenet":
            return (img_tensor - IMAGENET_MEAN) / IMAGENET_STD
        # Default: already in [0.0, 1.0]
        return img_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Fetch an (image, mask) pair.

        Returns:
            Tuple[torch.FloatTensor, torch.LongTensor]:
              - image: (3, H, W) float tensor
              - target: (H, W) long tensor with class indices in [0, N_max]
        """
        if self.mode in {"on_the_fly", "online"}:
            rng = None
            if self.seed is not None:
                seed_seq = np.random.SeedSequence(self.seed, spawn_key=(idx,))
                rng = np.random.default_rng(seed_seq)

            result = self.compositor.render(rng=rng)
            image_np = result.image  # (H, W, 3) uint8
            mask_np = result.mask    # (H, W) int64

        else:
            # Disk mode
            img_path = self.image_files[idx]
            mask_path = self.mask_files[idx]

            with Image.open(img_path) as pil_img:
                image_np = np.array(pil_img.convert("RGB"), dtype=np.uint8)

            if mask_path.suffix.lower() == ".npy":
                mask_np = np.load(mask_path).astype(np.int64)
            else:
                with Image.open(mask_path) as pil_mask:
                    mask_np = np.array(pil_mask, dtype=np.int64)

        # Convert to PyTorch tensors
        # image: (H, W, 3) uint8 -> (3, H, W) float32 in [0, 1]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = self._normalize_image(image_tensor)

        # target: (H, W) int64 -> torch.LongTensor
        target_tensor = torch.from_numpy(mask_np).long()

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        if self.target_transform is not None:
            target_tensor = self.target_transform(target_tensor)

        return image_tensor, target_tensor
