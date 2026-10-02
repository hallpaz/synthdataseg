# TASK_SPEC: Procedural Synthetic Dataset Generator for Semantic Segmentation

## 1. Executive Summary & Objective

This project implements a modular, high-performance Python package (`synthdataseg`) and command-line interface (`generate.py`) for procedurally generating synthetic semantic segmentation datasets. It composites mathematically defined, procedurally textured, geometrically transformed 2D vector shapes onto real (or synthetic procedural) backgrounds, outputting paired RGB images and ground truth integer segmentation masks, along with a full PyTorch `Dataset` implementation supporting both dynamic in-memory generation and disk-backed loading.

---

## 2. Architecture & File Structure

```text
synth-segmentation/
├── prompt.md
├── TASK_SPEC.md
├── synthdataseg/
│   ├── __init__.py
│   ├── config.py             # Dataclasses, ClassRegistry, YAML configuration loader
│   ├── shapes.py             # Vector geometry generators (Convex N-gons via Valtr's algorithm, Star, Basquiat Crown)
│   ├── textures.py           # Vectorized procedural texture synthesizers (Plain, Perlin wood, marble, fur)
│   ├── transforms.py         # 3x3 Homogeneous affine transformation engine & area normalization
│   ├── compositor.py         # Stratified class sampler, layer composer & ground truth mask painter
│   ├── dataset.py            # PyTorch Dataset (on-the-fly & disk-backed) & transforms
│   ├── generate.py           # CLI tool for dataset generation, balance verification, & reporting
│   └── tests/
│       ├── __init__.py
│       ├── test_shapes.py    # Shape geometry, convexity, non-self-intersection, area normalization
│       └── test_dataset.py   # Dataset modes, mask value bounds [0, N_max], DataLoader batching
├── tests/                    # Top-level test runner discovery pointing to test suites
│   ├── test_shapes.py
│   └── test_dataset.py
└── artifacts/
    ├── sample_verification.png      # 8-sample side-by-side visual verification grid
    ├── class_balance_histogram.png  # Instance & pixel distribution across classes
    └── dataset_balance.json         # Statistical report of dataset balance
```

---

## 3. Detailed Specifications

### A. Class Schema (`config.py`)
- **Class 0:** Background
- **Class 1:** 5-pointed regular star (pentagram)
- **Class 2:** Basquiat-style 3-peaked crown
- **Class $N$ ($N \ge 3$):** Convex polygon with $N$ vertices (Class 3 = Triangle, Class 4 = Quadrilateral, ..., Class $N_{\max}$).
- Extensible registry pattern allowing arbitrary $N_{\max} \ge 3$.
- Dataclasses for shapes, textures, affine transformations, canvas compositing, and dataset splits.
- Full YAML serialization and deserialization support.

### B. Geometry Generation (`shapes.py`)
- Centered at $(0, 0)$ within normalized coordinates $[-0.5, 0.5]^2$.
- **Convex $N$-gons:** Generated via **Valtr's Algorithm** (Pavel Valtr, 1995), guaranteed to produce strictly convex, non-self-intersecting $N$-gons with exactly $N$ vertices uniformly distributed.
- **Regular Star (Pentagram):** 10 alternating vertices with outer radius $R$ and inner radius $r = R \cdot \frac{3 - \sqrt{5}}{2} \approx 0.382 R$.
- **Basquiat Crown:** Iconic 3-peaked crown with flat base, flaring or vertical sides, and two deep V-shaped notches separating left, center, and right peaks.
- **Area Calculation & Normalization:** Shoelace formula to compute exact polygon area $A = \frac{1}{2} |\sum (x_i y_{i+1} - x_{i+1} y_i)|$ and centroid centering $(C_x, C_y)$.

### C. Procedural Texture Engine (`textures.py`)
- Fully vectorized NumPy implementations of 2D Perlin noise and fractional Brownian motion (fBm) turbulence.
- **Plain Colors:** Palette-based and uniform RGB sampling with optional subtle gradients.
- **Wood:** Concentric ring function $I(x, y) = \sin(k \cdot r + \alpha \cdot \text{Noise}(x, y))$ mapped to rich wood tones.
- **Marble:** Directional wave turbulence $I(x, y) = \sin(k_x x + k_y y + \beta \cdot \text{Turbulence}(x, y))$ with realistic veining.
- **Fur/Fibers:** Anisotropic noise stretched along a flow angle with local angular perturbation and hair-like striations.

### D. Transformations (`transforms.py`)
- $3 \times 3$ Homogeneous Affine transformation matrix:
  $$\mathbf{M} = \mathbf{T}(\Delta x, \Delta y) \cdot \mathbf{R}(\theta) \cdot \mathbf{Sh}(s_{xy}, s_{yx}) \cdot \mathbf{Sc}(s_x, s_y)$$
- **Area-Normalized Shape Scaling:**
  To guarantee that every shape (regardless of whether it is a star, crown, or 8-gon) occupies approximately the same target pixel area $A_{\text{target}}$, the scaling factor is derived from the linear determinant:
  $$|\det(\mathbf{R} \cdot \mathbf{Sh} \cdot \mathbf{Sc})| = s_x s_y |1 - s_{xy} s_{yx}|$$
  $$s = \sqrt{\frac{A_{\text{target}}}{A_{\text{base}} \cdot |1 - s_{xy} s_{yx}| \cdot \rho}}$$
  where $\rho = s_y / s_x$ is the aspect ratio jitter.

### E. Compositing & Mask Generation (`compositor.py`)
- Real background loading with automatic aspect ratio cropping/resizing, with seamless fallback to procedural background synthesis.
- **Stratified Class Sampling:** Global frequency tracker prioritizing underrepresented classes across images to prevent dataset bias.
- **Target Total Foreground Coverage:** Enforces total foreground area occupies $20\%$ to $45\%$ of canvas pixels.
- **Painter's Algorithm:** Renders textured polygon layers from back to front, writing RGB values and stamping integer class IDs into `mask[y, x] = class_id`.

### F. PyTorch Dataset (`dataset.py`)
- `SyntheticSegmentationDataset` with:
  1. `mode="on_the_fly"`: Infinite dynamic procedural generation in memory.
  2. `mode="disk"`: Loads pre-generated paired PNG images and masks.
- Output shapes:
  - `image`: `torch.FloatTensor` of shape `(3, H, W)` normalized to $[0.0, 1.0]$ or ImageNet statistics.
  - `target`: `torch.LongTensor` of shape `(H, W)` with class indices in $[0, N_{\max}]$.

### G. Balance Report & CLI (`generate.py`)
- CLI with rich parameters for split sizing, image resolution, background directories, and seeds.
- Statistical computation of class instance count and pixel count distributions.
- Tolerance threshold validation: Warnings or failures if any foreground class deviates by more than $15\%$ from target mean pixel count.
- Exports `dataset_balance.json` and generates `artifacts/class_balance_histogram.png`.
