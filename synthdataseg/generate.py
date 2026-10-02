"""
CLI tool for procedural synthetic semantic segmentation dataset generation.

Generates:
  1. Train / Val / Test dataset splits with paired RGB images and integer masks
  2. Dataset Balance Report (dataset_balance.json)
  3. Class Balance Histogram (artifacts/class_balance_histogram.png)
  4. Visual Verification Grid (artifacts/sample_verification.png)
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthdataseg.config import (
    ClassRegistry,
    DatasetConfig,
    GenerationConfig,
    CompositorConfig,
    load_config,
)
from synthdataseg.compositor import SyntheticCompositor, CompositeResult


def create_verification_grid(
    samples: list[CompositeResult],
    registry: ClassRegistry,
    output_path: Path,
) -> None:
    """
    Generate a visual verification grid with 8 samples:
    Original RGB on left, color-coded ground truth mask on right with legend.
    """
    n_samples = min(8, len(samples))
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 3.2 * n_samples))
    fig.patch.set_facecolor("#18181b")

    if n_samples == 1:
        axes = np.array([axes])

    palette = registry.get_color_palette()

    for i in range(n_samples):
        res = samples[i]
        rgb = res.image
        mask = res.mask
        color_mask = registry.colorize_mask(mask)

        # Plot RGB image
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(
            f"Sample #{i+1} - Composite RGB (K={len(res.layers)} shapes, Cov={res.foreground_coverage:.1%})",
            color="white",
            fontsize=11,
            pad=6,
        )
        axes[i, 0].axis("off")

        # Plot Color-Coded Mask
        axes[i, 1].imshow(color_mask)
        present_classes = sorted(list(np.unique(mask)))
        class_str = ", ".join([f"{cid}:{registry.get_class_name(cid)}" for cid in present_classes if cid != 0])
        axes[i, 1].set_title(
            f"Ground Truth Mask [{class_str}]",
            color="white",
            fontsize=10,
            pad=6,
        )
        axes[i, 1].axis("off")

    # Add legend at bottom
    handles = []
    labels = []
    for cid in registry.get_class_ids():
        col_norm = [c / 255.0 for c in palette[cid]]
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=col_norm, edgecolor="white", linewidth=0.5)
        handles.append(patch)
        labels.append(f"{cid}: {registry.get_class_name(cid)}")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(labels)),
        facecolor="#27272a",
        edgecolor="#52525b",
        labelcolor="white",
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_balance_histogram(
    report: Dict[str, Any],
    registry: ClassRegistry,
    output_path: Path,
    tolerance: float = 0.15,
) -> None:
    """
    Plot and save class balance histogram:
      - Left: Instance counts per foreground class
      - Right: Pixel count and percentage with tolerance boundaries
    """
    fg_classes = registry.get_foreground_class_ids()
    class_names = [f"C{cid}: {registry.get_class_name(cid)}" for cid in fg_classes]

    fg_instances = [report["classes"][str(cid)]["instance_count"] for cid in fg_classes]
    fg_pixels = [report["classes"][str(cid)]["pixel_count"] for cid in fg_classes]
    mean_pixels = float(np.mean(fg_pixels)) if fg_pixels else 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#18181b")

    palette = registry.get_color_palette()
    bar_colors = [[c / 255.0 for c in palette[cid]] for cid in fg_classes]

    # Subplot 1: Instance counts
    bars1 = ax1.bar(class_names, fg_instances, color=bar_colors, edgecolor="white", alpha=0.9, width=0.6)
    ax1.set_facecolor("#27272a")
    ax1.set_title("Total Instances per Foreground Class", color="white", fontsize=13, pad=10)
    ax1.set_ylabel("Instance Count", color="white", fontsize=11)
    ax1.tick_params(colors="white", labelsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.3, color="#a1a1aa")
    plt.setp(ax1.get_xticklabels(), rotation=25, ha="right")

    for bar, count in zip(bars1, fg_instances):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            color="white",
            fontweight="bold",
            fontsize=9,
        )

    # Subplot 2: Pixel counts with tolerance band
    bars2 = ax2.bar(class_names, fg_pixels, color=bar_colors, edgecolor="white", alpha=0.9, width=0.6)
    ax2.set_facecolor("#27272a")
    ax2.set_title(
        f"Total Foreground Pixels per Class (Mean = {mean_pixels:,.0f}, ±{tolerance:.0%} Band)",
        color="white",
        fontsize=13,
        pad=10,
    )
    ax2.set_ylabel("Pixel Count", color="white", fontsize=11)
    ax2.tick_params(colors="white", labelsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.3, color="#a1a1aa")
    plt.setp(ax2.get_xticklabels(), rotation=25, ha="right")

    # Mean line and tolerance shaded band
    ax2.axhline(mean_pixels, color="#38bdf8", linestyle="-", linewidth=1.8, label=f"Target Mean ({mean_pixels:,.0f})")
    ax2.axhspan(
        mean_pixels * (1.0 - tolerance),
        mean_pixels * (1.0 + tolerance),
        color="#38bdf8",
        alpha=0.15,
        label=f"±{tolerance:.0%} Tolerance Band",
    )
    ax2.legend(facecolor="#3f3f46", edgecolor="#71717a", labelcolor="white", loc="upper right")

    for bar, px in zip(bars2, fg_pixels):
        dev = ((px - mean_pixels) / mean_pixels) * 100
        sign = "+" if dev >= 0 else ""
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (mean_pixels * 0.02),
            f"{px:,.0f}\n({sign}{dev:.1f}%)",
            ha="center",
            va="bottom",
            color="white",
            fontsize=8,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def generate_sample(
    idx: int,
    seed: Optional[int] = 42,
    config: Optional[DatasetConfig] = None,
    num_samples: int = 100,
) -> CompositeResult:
    """
    Generate a single composite sample deterministically by index.
    Matches the sample produced at index `idx` in batch generation with `seed`.
    """
    if config is None:
        config = DatasetConfig()

    registry = ClassRegistry(max_n=config.max_n)
    compositor = SyntheticCompositor(
        registry=registry,
        config=config.compositor,
        texture_config=config.textures,
    )

    if seed is not None:
        child_seed = np.random.SeedSequence(seed).spawn(max(idx + 1, num_samples))[idx]
        sample_rng = np.random.default_rng(child_seed)
    else:
        sample_rng = None

    return compositor.render(rng=sample_rng)


def generate_dataset(
    config: DatasetConfig,
    fail_on_imbalance: bool = False,
    artifact_dirs: Optional[list[Path]] = None,
) -> Dict[str, Any]:
    """
    Execute generation across all configured splits and compile balance report.
    """
    output_root = Path(config.generation.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    registry = ClassRegistry(max_n=config.max_n)
    master_seed = config.generation.seed

    compositor = SyntheticCompositor(
        registry=registry,
        config=config.compositor,
        texture_config=config.textures,
        rng=np.random.default_rng(master_seed) if master_seed is not None else None,
    )

    total_instances_dataset: Dict[int, int] = {cid: 0 for cid in registry.get_foreground_class_ids()}
    total_pixels_dataset: Dict[int, int] = {cid: 0 for cid in registry.get_class_ids()}
    all_sample_results: list[CompositeResult] = []

    print(f"\n========================================================")
    print(f" Procedural Synthetic Dataset Generator")
    print(f" Output Directory: {output_root.resolve()}")
    print(f" Classes: {registry.num_classes} (0=Background, 1=Star, 2=Crown, 3..{config.max_n}=Convex N-gons)")
    print(f" Canvas Size: {config.compositor.image_size[0]}x{config.compositor.image_size[1]}")
    print(f"========================================================\n")

    for split_idx, (split_name, count) in enumerate(config.generation.splits.items()):
        if count <= 0:
            continue
        print(f"Generating split '{split_name}' ({count} samples)...")
        split_dir = output_root / split_name
        img_dir = split_dir / "images"
        mask_dir = split_dir / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        # Derive independent, deterministic child seeds for each sample index
        if master_seed is not None:
            if split_name == "train":
                split_seq = np.random.SeedSequence(master_seed)
            else:
                split_seq = np.random.SeedSequence(master_seed, spawn_key=(split_idx,))
            child_seeds = split_seq.spawn(count)
        else:
            child_seeds = None

        for idx in range(count):
            sample_rng = np.random.default_rng(child_seeds[idx]) if child_seeds is not None else None
            result = compositor.render(rng=sample_rng)

            # Save sample for visual verification if from train or first split
            if len(all_sample_results) < 8:
                all_sample_results.append(result)

            # Save RGB image as PNG
            img_file = img_dir / f"{idx:05d}.png"
            Image.fromarray(result.image).save(img_file, format="PNG")

            # Save Mask: 8-bit PNG where pixel values are exact integer class IDs
            mask_file = mask_dir / f"{idx:05d}.png"
            palette = registry.get_color_palette().flatten().tolist()
            mask_img = Image.fromarray(result.mask.astype(np.uint8), mode="P")
            mask_img.putpalette(palette)
            mask_img.save(mask_file, format="PNG")
            # Image.fromarray(result.mask.astype(np.uint8)).save(mask_file, format="PNG")

            # Accumulate dataset-wide stats
            for cid, px_cnt in result.class_counts.items():
                total_pixels_dataset[cid] += px_cnt
            for cid, inst_cnt in result.instance_counts.items():
                total_instances_dataset[cid] += inst_cnt

            if (idx + 1) % max(1, count // 5) == 0 or (idx + 1) == count:
                print(f"  [{split_name}] Completed {idx + 1}/{count} samples")

    # Compute Statistical Balance Report
    total_pixels_all = sum(total_pixels_dataset.values())
    fg_classes = registry.get_foreground_class_ids()
    fg_pixel_counts = [total_pixels_dataset[cid] for cid in fg_classes]
    target_mean_fg_pixels = float(np.mean(fg_pixel_counts)) if fg_pixel_counts else 0.0

    class_reports: Dict[str, Any] = {}
    max_fg_deviation = 0.0
    imbalanced_classes = []

    for cid in registry.get_class_ids():
        px = total_pixels_dataset[cid]
        pct = (px / total_pixels_all) * 100.0 if total_pixels_all > 0 else 0.0
        inst = total_instances_dataset.get(cid, 0)

        dev_from_mean = 0.0
        if cid != 0 and target_mean_fg_pixels > 0:
            dev_from_mean = (px - target_mean_fg_pixels) / target_mean_fg_pixels
            abs_dev = abs(dev_from_mean)
            if abs_dev > max_fg_deviation:
                max_fg_deviation = abs_dev
            if abs_dev > config.generation.tolerance_threshold:
                imbalanced_classes.append((cid, registry.get_class_name(cid), dev_from_mean))

        class_reports[str(cid)] = {
            "class_id": cid,
            "name": registry.get_class_name(cid),
            "instance_count": inst,
            "pixel_count": int(px),
            "pixel_percentage": round(pct, 2),
            "deviation_from_target_mean": round(dev_from_mean * 100.0, 2) if cid != 0 else None,
        }

    balance_report = {
        "summary": {
            "total_pixels": int(total_pixels_all),
            "target_mean_foreground_pixels": round(target_mean_fg_pixels, 1),
            "max_foreground_deviation_percent": round(max_fg_deviation * 100.0, 2),
            "tolerance_threshold_percent": round(config.generation.tolerance_threshold * 100.0, 2),
            "is_balanced": len(imbalanced_classes) == 0,
        },
        "classes": class_reports,
    }

    # Print Formatted Report Table
    print("\n" + "=" * 80)
    print(f"{'Class ID':<10}{'Class Name':<20}{'Instances':<12}{'Pixels':<15}{'Pixel %':<12}{'Dev vs Mean FG':<15}")
    print("-" * 80)
    for cid in registry.get_class_ids():
        c_info = class_reports[str(cid)]
        dev_str = f"{c_info['deviation_from_target_mean']:+.1f}%" if c_info['deviation_from_target_mean'] is not None else "-"
        print(f"{c_info['class_id']:<10}{c_info['name']:<20}{c_info['instance_count']:<12}{c_info['pixel_count']:<15,}{c_info['pixel_percentage']:<12.2f}{dev_str:<15}")
    print("=" * 80)
    print(f"Target Mean Foreground Pixels: {target_mean_fg_pixels:,.0f}")
    print(f"Max Foreground Class Deviation: {max_fg_deviation * 100.0:.2f}% (Tolerance: {config.generation.tolerance_threshold * 100.0:.1f}%)")

    if imbalanced_classes:
        warn_msg = f"WARNING: {len(imbalanced_classes)} class(es) exceeded the {config.generation.tolerance_threshold*100:.1f}% deviation threshold: " + \
                   ", ".join([f"Class {c[0]} ({c[1]}): {c[2]*100:+.1f}%" for c in imbalanced_classes])
        print(f"\n[!] {warn_msg}\n")
        if fail_on_imbalance:
            raise ValueError(warn_msg)
    else:
        print(f"\n[OK] All foreground classes are strictly within the +/-{config.generation.tolerance_threshold*100:.1f}% balance threshold!\n")

    # Save reports and artifacts
    json_path = output_root / "dataset_balance.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(balance_report, f, indent=2)

    # Determine artifact destination paths
    art_paths = [Path("artifacts")]
    if artifact_dirs:
        art_paths.extend(artifact_dirs)

    for art_dir in art_paths:
        art_dir.mkdir(parents=True, exist_ok=True)
        # 1. Copy JSON report
        shutil.copy2(json_path, art_dir / "dataset_balance.json")
        # 2. Plot and save balance histogram
        plot_balance_histogram(
            balance_report,
            registry,
            art_dir / "class_balance_histogram.png",
            tolerance=config.generation.tolerance_threshold,
        )
        # 3. Generate and save verification grid
        if all_sample_results:
            create_verification_grid(
                all_sample_results,
                registry,
                art_dir / "sample_verification.png",
            )

    print(f"Reports & Visualizations successfully exported to:")
    for art_dir in art_paths:
        print(f"  - {art_dir.resolve()}/sample_verification.png")
        print(f"  - {art_dir.resolve()}/class_balance_histogram.png")
        print(f"  - {art_dir.resolve()}/dataset_balance.json")

    return balance_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Procedural Synthetic Dataset Generator for Semantic Segmentation"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--output-dir", type=str, default="./data/synthetic_dataset", help="Output directory")
    parser.add_argument("--num-train", type=int, default=100, help="Number of train samples")
    parser.add_argument("--num-val", type=int, default=20, help="Number of validation samples")
    parser.add_argument("--num-test", type=int, default=20, help="Number of test samples")
    parser.add_argument("--image-size", type=int, nargs="+", default=[256, 256], help="Canvas height and width")
    parser.add_argument("--max-n", type=int, default=8, help="Maximum N for convex N-gon classes")
    parser.add_argument("--bg-dir", type=str, default=None, help="Directory containing background images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--tolerance", type=float, default=0.15, help="Balance deviation tolerance threshold")
    parser.add_argument("--fail-on-imbalance", action="store_true", help="Exit with error if tolerance is exceeded")
    parser.add_argument("--artifact-dir", type=str, default="artifacts", help="Directory for verification artifacts")
    parser.add_argument("--sample-idx", type=int, default=None, help="Generate a single sample standalone by index")

    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = DatasetConfig(max_n=args.max_n)

    # CLI parameter overrides
    cfg.generation.output_dir = args.output_dir
    cfg.generation.seed = args.seed
    cfg.generation.tolerance_threshold = args.tolerance
    cfg.max_n = args.max_n

    if len(args.image_size) == 1:
        img_size = (args.image_size[0], args.image_size[0])
    else:
        img_size = (args.image_size[0], args.image_size[1])

    cfg.compositor.image_size = img_size
    cfg.generation.image_size = img_size
    if args.bg_dir:
        cfg.compositor.bg_dir = args.bg_dir

    if args.sample_idx is not None:
        result = generate_sample(args.sample_idx, seed=cfg.generation.seed, config=cfg)
        print(f"Sample #{args.sample_idx:05d} generated standalone: {len(result.layers)} shapes, coverage {result.foreground_coverage:.1%}")
        return

    cfg.generation.splits = {
        "train": args.num_train,
        "val": args.num_val,
        "test": args.num_test,
    }

    # Also include the Antigravity Ide brain artifact directory if it exists
    antigravity_artifact_dir = Path("/Users/hallpaz/.gemini/antigravity-ide/brain/47260149-3b0a-4107-a037-78dda62c8712")
    art_dirs = [Path(args.artifact_dir)]
    if antigravity_artifact_dir.exists():
        art_dirs.append(antigravity_artifact_dir)

    try:
        generate_dataset(
            cfg,
            fail_on_imbalance=args.fail_on_imbalance,
            artifact_dirs=art_dirs,
        )
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
