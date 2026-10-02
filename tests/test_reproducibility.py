"""
Verification test suite for 100% bit-for-bit deterministic reproducibility.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
from PIL import Image

from synthetic_seg.config import ClassRegistry, CompositorConfig, DatasetConfig
from synthetic_seg.compositor import SyntheticCompositor
from synthetic_seg.dataset import SyntheticSegmentationDataset
from synthetic_seg.generate import generate_dataset, generate_sample


def test_reproducibility_run1_vs_run2():
    """
    Generate 5 composite images and masks twice using seed=42.
    Assert np.array_equal(run1_img, run2_img) and np.array_equal(run1_mask, run2_mask) for all samples.
    """
    master_seed = 42
    n_samples = 5
    canvas_size = (128, 128)

    cfg = DatasetConfig()
    cfg.compositor.image_size = canvas_size

    # Run 1
    run1_images = []
    run1_masks = []
    seed_seq1 = np.random.SeedSequence(master_seed)
    child_seeds1 = seed_seq1.spawn(n_samples)
    comp1 = SyntheticCompositor(
        registry=ClassRegistry(max_n=cfg.max_n),
        config=cfg.compositor,
        texture_config=cfg.textures,
    )
    for i in range(n_samples):
        rng = np.random.default_rng(child_seeds1[i])
        res = comp1.render(rng=rng)
        run1_images.append(res.image)
        run1_masks.append(res.mask)

    # Run 2
    run2_images = []
    run2_masks = []
    seed_seq2 = np.random.SeedSequence(master_seed)
    child_seeds2 = seed_seq2.spawn(n_samples)
    comp2 = SyntheticCompositor(
        registry=ClassRegistry(max_n=cfg.max_n),
        config=cfg.compositor,
        texture_config=cfg.textures,
    )
    for i in range(n_samples):
        rng = np.random.default_rng(child_seeds2[i])
        res = comp2.render(rng=rng)
        run2_images.append(res.image)
        run2_masks.append(res.mask)

    # Assert bit-for-bit identity for all 5 samples
    for i in range(n_samples):
        assert np.array_equal(
            run1_images[i], run2_images[i]
        ), f"Sample {i} RGB image differs between run 1 and run 2!"
        assert np.array_equal(
            run1_masks[i], run2_masks[i]
        ), f"Sample {i} segmentation mask differs between run 1 and run 2!"


def test_background_paths_order_invariance():
    """
    Assert background paths list is identical regardless of filesystem directory iteration order.
    Forces case-insensitive lexical sorting.
    """
    file_names = [
        "zeta.JPG",
        "Alpha.png",
        "BETA.jpeg",
        "01_first.webp",
        "gamma.bmp",
        "DELTA.PNG",
    ]
    # Expected lexical sort by name.lower()
    expected_order = [
        "01_first.webp",
        "Alpha.png",
        "BETA.jpeg",
        "DELTA.PNG",
        "gamma.bmp",
        "zeta.JPG",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        bg_dir = Path(tmpdir)
        # Create dummy image files
        for name in file_names:
            img_file = bg_dir / name
            img = Image.new("RGB", (64, 64), color=(100, 150, 200))
            img.save(img_file)

        comp = SyntheticCompositor(config=CompositorConfig(bg_dir=str(bg_dir)))
        scanned_names = [p.name for p in comp.bg_paths]

        assert scanned_names == expected_order, (
            f"Expected sorted background filenames {expected_order}, got {scanned_names}"
        )
        assert comp.bg_paths == comp.bg_files


def test_sample_42_batch_vs_standalone():
    """
    Ensure sample #00042 is identical whether generated as part of a batch of 100
    or generated standalone by index.
    """
    master_seed = 42
    target_idx = 42
    n_samples = 100

    cfg = DatasetConfig()
    cfg.generation.seed = master_seed
    cfg.generation.splits = {"train": n_samples, "val": 0, "test": 0}
    cfg.compositor.image_size = (128, 128)

    # 1. Generate standalone via generate_sample helper
    solo_res = generate_sample(
        idx=target_idx,
        seed=master_seed,
        config=cfg,
        num_samples=n_samples,
    )

    # 2. Generate batch of 100 and extract sample #00042
    child_seeds = np.random.SeedSequence(master_seed).spawn(n_samples)
    comp_batch = SyntheticCompositor(
        registry=ClassRegistry(max_n=cfg.max_n),
        config=cfg.compositor,
        texture_config=cfg.textures,
    )
    batch_42_res = None
    for idx in range(n_samples):
        rng = np.random.default_rng(child_seeds[idx])
        res = comp_batch.render(rng=rng)
        if idx == target_idx:
            batch_42_res = res

    assert batch_42_res is not None

    # Assert bit-for-bit identical
    assert np.array_equal(
        solo_res.image, batch_42_res.image
    ), "Sample #00042 RGB image differs between standalone and batch!"
    assert np.array_equal(
        solo_res.mask, batch_42_res.mask
    ), "Sample #00042 segmentation mask differs between standalone and batch!"

    # 3. Check on-the-fly SyntheticSegmentationDataset matches as well
    dataset = SyntheticSegmentationDataset(
        mode="on_the_fly",
        length=n_samples,
        registry=ClassRegistry(max_n=cfg.max_n),
        compositor_config=cfg.compositor,
        texture_config=cfg.textures,
        normalize_mode="unit",
        seed=master_seed,
    )
    ds_img_tensor, ds_mask_tensor = dataset[target_idx]
    expected_img_tensor = (solo_res.image / 255.0).astype(np.float32)
    assert np.allclose(
        ds_img_tensor.permute(1, 2, 0).numpy(), expected_img_tensor, atol=1e-5
    ), "Dataset item #00042 tensor differs from standalone sample!"
    assert np.array_equal(
        ds_mask_tensor.numpy(), solo_res.mask
    ), "Dataset item #00042 mask differs from standalone sample!"


def test_independent_of_global_random_state():
    """
    Verify that mutating Python's global random or NumPy's legacy global random state
    does not affect the deterministic output when seed is provided.
    """
    import random

    seed = 12345
    comp = SyntheticCompositor(config=CompositorConfig(image_size=(64, 64)))

    # Run with global state undisturbed
    rng1 = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
    res1 = comp.render(rng=rng1)

    # Mess up global random states
    random.seed(999999)
    np.random.seed(888888)
    _ = [random.random() for _ in range(50)]
    _ = np.random.rand(50)

    # Run again with same seed sequence
    rng2 = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
    res2 = comp.render(rng=rng2)

    assert np.array_equal(res1.image, res2.image)
    assert np.array_equal(res1.mask, res2.mask)
