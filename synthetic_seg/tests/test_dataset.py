"""
Unit tests for PyTorch Dataset integration, dataloaders, and mask validation.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from PIL import Image

from synthetic_seg.config import ClassRegistry, CompositorConfig, TextureConfig
from synthetic_seg.dataset import SyntheticSegmentationDataset
from synthetic_seg.compositor import SyntheticCompositor


def test_on_the_fly_dataset_and_mask_bounds():
    """Verify on-the-fly dataset returns correct tensor shapes and mask values in [0, N_max]."""
    max_n = 8
    registry = ClassRegistry(max_n=max_n)
    comp_config = CompositorConfig(image_size=(128, 128), min_shapes=2, max_shapes=4)

    dataset = SyntheticSegmentationDataset(
        mode="on_the_fly",
        length=10,
        registry=registry,
        compositor_config=comp_config,
        normalize_mode="unit",
        seed=123,
    )

    assert len(dataset) == 10

    for i in range(5):
        img_tensor, mask_tensor = dataset[i]

        # Shape check
        assert img_tensor.shape == (3, 128, 128)
        assert mask_tensor.shape == (128, 128)

        # Type check
        assert img_tensor.dtype == torch.float32
        assert mask_tensor.dtype == torch.int64

        # Value bounds
        assert img_tensor.min() >= 0.0
        assert img_tensor.max() <= 1.0

        # Mask strictly within [0, max_n]
        assert mask_tensor.min() >= 0
        assert mask_tensor.max() <= max_n
        unique_vals = torch.unique(mask_tensor).tolist()
        for v in unique_vals:
            assert 0 <= v <= max_n, f"Unexpected class value {v} outside [0, {max_n}]"


def test_pytorch_dataloader_batching():
    """Verify DataLoader batches tensors correctly into (B, 3, H, W) and (B, H, W)."""
    registry = ClassRegistry(max_n=6)
    comp_config = CompositorConfig(image_size=(64, 64), min_shapes=1, max_shapes=3)

    dataset = SyntheticSegmentationDataset(
        mode="on_the_fly",
        length=12,
        registry=registry,
        compositor_config=comp_config,
        normalize_mode="imagenet",
        seed=999,
    )

    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    batches_tested = 0
    for images, targets in loader:
        assert images.shape == (batch_size, 3, 64, 64)
        assert targets.shape == (batch_size, 64, 64)
        assert images.dtype == torch.float32
        assert targets.dtype == torch.int64
        assert targets.min() >= 0
        assert targets.max() <= 6
        batches_tested += 1

    assert batches_tested == 3


def test_disk_backed_dataset():
    """Verify disk-backed mode correctly discovers and loads images and masks from filesystem."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        # Create 3 synthetic sample pairs
        for idx in range(3):
            dummy_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            dummy_mask = np.random.randint(0, 5, (64, 64), dtype=np.uint8)

            Image.fromarray(dummy_img).save(img_dir / f"{idx:05d}.png")
            Image.fromarray(dummy_mask).save(mask_dir / f"{idx:05d}.png")

        dataset = SyntheticSegmentationDataset(
            mode="disk",
            data_dir=tmp_path,
            normalize_mode="unit",
        )

        assert len(dataset) == 3
        img, mask = dataset[0]
        assert img.shape == (3, 64, 64)
        assert mask.shape == (64, 64)
        assert img.dtype == torch.float32
        assert mask.dtype == torch.int64
