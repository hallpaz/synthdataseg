import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from synthdataseg.config import ClassRegistry, CompositorConfig, TextureConfig
from synthdataseg.compositor import SyntheticCompositor
from synthdataseg.dataset import SyntheticSegmentationDataset

def test_single_generation():
    registry = ClassRegistry(max_n=8)
    comp_cfg = CompositorConfig(
        image_size=(384, 384),
        min_shapes=3,
        max_shapes=6,
        bg_dir=None,  # Or Path to your background images folder
    )
    tex_cfg = TextureConfig()
    rng = np.random.default_rng(42)

    compositor = SyntheticCompositor(
        registry=registry,
        config=comp_cfg,
        texture_config=tex_cfg,
        rng=rng,
    )

    # 1. Generate one composite
    result = compositor.render(rng=rng)
    color_mask = registry.colorize_mask(result.mask)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(result.image)
    axes[0].set_title(f"Synthetic RGB (Foreground: {result.foreground_coverage:.1%})")
    axes[0].axis("off")

    axes[1].imshow(color_mask)
    classes_present = [f"{c}:{registry.get_class_name(c)}" for c in np.unique(result.mask) if c != 0]
    axes[1].set_title(f"Classes: {', '.join(classes_present)}")
    axes[1].axis("off")
    plt.tight_layout()
    import os
    os.makedirs("artifacts", exist_ok=True)
    preview_path = "artifacts/demo_single_generation.png"
    plt.savefig(preview_path, bbox_inches="tight", dpi=150)
    print(f"Saved preview to {preview_path}")
    plt.show()

def test_pytorch_dataloader():
    # 2. Test PyTorch on-the-fly integration
    dataset = SyntheticSegmentationDataset(
        mode="on_the_fly",
        num_samples=16,
        image_size=(256, 256),
        max_n=8,
        seed=1337,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    images, targets = next(iter(loader))
    print(f"\nDataLoader Batch Verification:")
    print(f"  Images Tensor:  {images.shape}, dtype={images.dtype}, range=[{images.min():.2f}, {images.max():.2f}]")
    print(f"  Targets Tensor: {targets.shape}, dtype={targets.dtype}, classes={torch.unique(targets).tolist()}")

if __name__ == "__main__":
    test_single_generation()
    test_pytorch_dataloader()