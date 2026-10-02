# Antigravity Task: Procedural Synthetic Dataset Generator for Semantic Segmentation

## 1. Objective

Build a modular Python package and CLI tool that generates synthetic semantic segmentation datasets by compositing procedurally textured, geometrically transformed 2D shapes onto real background images, accompanied by ground truth segmentation masks and a PyTorch `Dataset` implementation.

---

## 2. Architecture & File Structure

Ensure the implementation follows this modular directory layout:

```text
synthetic_seg/
├── __init__.py
├── config.py             # Dataclasses / YAML configuration loader
├── shapes.py             # Vector geometry generators (N-gons, Star, Basquiat Crown)
├── textures.py           # Solid color & procedural texture synthesizers (Perlin wood, marble, fur)
├── transforms.py         # 2D Affine transformation engine (translation, rotation, shear, scale)
├── compositor.py         # Multi-shape layer composer & ground truth mask painter
├── dataset.py            # PyTorch Dataset & torchvision transform pipelines
├── generate.py           # CLI script to generate and save datasets to disk
└── tests/
    ├── test_shapes.py    # Unit tests for shape validity and class mapping
    └── test_dataset.py   # Unit tests for PyTorch dataloader integration

```

---

## 3. Module Specifications

### A. Class Schema (`config.py`)

Implement an extensible class registry:

* **Class 0:** Background
* **Class 1:** 5-pointed regular star (pentagram)
* **Class 2:** Basquiat-style 3-peaked crown
* **Class $N$ ($N \ge 3$):** Convex polygon with $N$ vertices (e.g., Class 3 = Triangle, Class 4 = Quadrilateral, ..., Class $N_{\max}$).

### B. Geometry Generation (`shapes.py`)

All shapes must be generated as normalized closed 2D polygon vertices centered at $(0, 0)$ within $[-0.5, 0.5]^2$:

1. **Convex $N$-gons:** Sample $N$ random angles $\theta_i \in [0, 2\pi)$, sort them, generate random radii within $[r_{\min}, r_{\max}]$, and build vertices ensuring strict convexity (e.g., via Valtr's algorithm or taking the 2D convex hull of random points).
2. **Regular Star (Pentagram):** 10 alternating vertices (5 outer tips at radius $R$, 5 inner vertices at radius $r \approx 0.382 R$).
3. **Basquiat Crown:** 3-peaked crown with a flat base, vertical/flaring sides, and two deep V-shaped notches separating three peaks (left, center, right).

### C. Procedural Texture Engine (`textures.py`)

Implement a texture synthesizer capable of filling a shape's bounding box using NumPy:

1. **Plain Colors:** Uniform RGB sampling or palette-based sampling.
2. **Wood:** Concentric rings generated via distance to an offset origin perturbed by 2D Perlin/Simplex noise: $I(x, y) = \sin(k \cdot r + \alpha \cdot \text{Noise}(x, y))$.
3. **Marble:** Sine wave turbulence: $I(x, y) = \sin(k_x x + k_y y + \beta \cdot \text{Turbulence}(x, y))$.
4. **Fur/Fibers:** High-frequency directional noise stretched along one axis (anisotropic noise) with slight local angular variation.

### D. Transformations (`transforms.py`)

Represent all transformations using $3 \times 3$ homogeneous affine matrices:


$$\mathbf{M} = \mathbf{T}(\Delta x, \Delta y) \cdot \mathbf{R}(\theta) \cdot \mathbf{Sh}(s_{xy}, s_{yx}) \cdot \mathbf{Sc}(s_x, s_y)$$

* **Translation:** Random $(x, y)$ inside image boundaries.
* **Rotation:** $\theta \sim \mathcal{U}(0, 2\pi)$.
* **Shearing:** Independent shears $s_{xy}, s_{yx} \sim \mathcal{U}(-s_{\max}, s_{\max})$.
* **Scaling:** Isotropic and anisotropic scaling $(s_x, s_y)$ with aspect ratio jitter.

### E. Compositing & Mask Generation (`compositor.py`)

1. **Background Selection:** Randomly sample an image from the user-specified background directory, cropping or resizing it to the target resolution $(H, W)$.
2. **Multi-Shape Layering:** Sample $K \sim \mathcal{U}(K_{\min}, K_{\max})$ shapes per image. Apply the **Painter's Algorithm** (sequential rendering from back to front):
* Render texture inside the transformed polygon path onto the RGB buffer.
* Draw the corresponding integer class ID onto an integer mask buffer `mask[y, x] = class_id` (dtype `np.int64` or `np.uint8`). Overlapping shapes must overwrite earlier shapes on both RGB and mask buffers to maintain consistent occlusion boundaries.



### F. PyTorch Dataset (`dataset.py`)

Implement `SyntheticSegmentationDataset(torch.utils.data.Dataset)`:

* Support two operating modes:
1. **On-the-fly:** Generates images and masks dynamically in memory per `__getitem__` call (ideal for infinite data augmentation without disk I/O bottlenecks).
2. **Disk-backed:** Loads pre-generated `.png` (image) and `.png`/`.npy` (mask) pairs from a generated dataset directory.


* Return tensors:
* `image`: `torch.FloatTensor` of shape `(3, H, W)` normalized to $[0.0, 1.0]$ or ImageNet statistics.
* `target`: `torch.LongTensor` of shape `(H, W)` with class indices in $[0, N_{\max}]$.



---

## 4. Execution & Verification Steps for Antigravity

Instruct the agent to carry out the following validation sequence in the workspace:

1. **Environment Setup:** Install `numpy`, `pillow`, `scipy`, `torch`, `torchvision`, and `pytest`.
2. **Core Implementation:** Implement all modules following the architecture above.
3. **Automated Unit Tests:** Run `pytest tests/` in the terminal to verify:
* Polygon generation produces valid non-self-intersecting contours.
* Segmentation masks strictly contain class values in the range $[0, N_{\max}]$.
* PyTorch `DataLoader` batches tensors correctly with shapes `(B, 3, H, W)` and `(B, H, W)`.


4. **Visual Verification Artifact:** Generate a sample batch of 8 images with their corresponding color-coded masks side-by-side and save them to `artifacts/sample_verification.png` for inspection in Antigravity's artifact viewer.

### Balancing & Verification Requirements to Add to TASK_SPEC.md

1. Area-Normalized Shape Scaling (`shapes.py` & `transforms.py`):
   - Rather than scaling shapes by arbitrary bounding box dimensions, scale each shape such that its polygon area $A \approx \text{target\_area} \pm \epsilon$.
   - This ensures a 3-pointed crown, a star, and an 8-gon contribute roughly the same number of pixels to the segmentation mask when placed at identical scale levels.

2. Stratified Class Sampling (`compositor.py`):
   - Do not sample class IDs with uniform independent probabilities per image.
   - Maintain a global or batch-level class frequency tracker so underrepresented classes are prioritized in subsequent images.
   - Target total foreground coverage: Enforce that total shape area per image occupies a controlled fraction of the canvas (e.g., 20% to 45% of pixels) to prevent extreme background class dominance.

3. Dataset Balance Report CLI (`generate.py`):
   - After generating a dataset split, compute and print:
     * Total instance count per class.
     * Total pixel count and percentage distribution per class (including Background Class 0).
   - Export this report to `dataset_balance.json` and save a bar chart to `artifacts/class_balance_histogram.png`.
   - Add a tolerance threshold: Warn or fail if any foreground class deviates by more than 15% from the target mean pixel count.