"""
Procedural texture synthesis engine using NumPy.

Generates:
  1. Plain colors (uniform & palette-based sampling with subtle gradients)
  2. Wood (concentric rings perturbed by 2D Perlin noise)
  3. Marble (sine wave turbulence)
  4. Fur/Fibers (anisotropic directional noise with angular perturbation)
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Tuple, Union
import numpy as np
from synthdataseg.config import TextureConfig


class TextureType(str, Enum):
    PLAIN = "plain"
    WOOD = "wood"
    MARBLE = "marble"
    FUR = "fur"


class PerlinNoise2D:
    """
    Vectorized 2D Perlin / Gradient Noise synthesizer in pure NumPy.
    """

    def __init__(self, seed: Optional[Union[int, np.random.Generator, np.random.SeedSequence]] = None) -> None:
        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed)
        # Precompute 256 random 2D unit gradient vectors
        angles = self.rng.uniform(0.0, 2 * np.pi, size=256)
        self.gradients = np.column_stack([np.cos(angles), np.sin(angles)])
        # Permutation table
        p = np.arange(256, dtype=int)
        self.rng.shuffle(p)
        self.perm = np.tile(p, 4)

    def _fade(self, t: np.ndarray) -> np.ndarray:
        """Quintic polynomial interpolation curve: 6t^5 - 15t^4 + 10t^3."""
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate 2D Perlin noise at grid points (x, y).

        Args:
            x: 2D array of x coordinates.
            y: 2D array of y coordinates.

        Returns:
            np.ndarray: Noise values in approximately [-1, 1].
        """
        xi0 = np.floor(x).astype(int) & 255
        yi0 = np.floor(y).astype(int) & 255
        xi1 = (xi0 + 1) & 255
        yi1 = (yi0 + 1) & 255

        dx = x - np.floor(x)
        dy = y - np.floor(y)

        # Hash corners
        g00_idx = self.perm[self.perm[xi0] + yi0]
        g10_idx = self.perm[self.perm[xi1] + yi0]
        g01_idx = self.perm[self.perm[xi0] + yi1]
        g11_idx = self.perm[self.perm[xi1] + yi1]

        g00 = self.gradients[g00_idx]
        g10 = self.gradients[g10_idx]
        g01 = self.gradients[g01_idx]
        g11 = self.gradients[g11_idx]

        # Dot products with displacement vectors
        dot00 = g00[..., 0] * dx + g00[..., 1] * dy
        dot10 = g10[..., 0] * (dx - 1.0) + g10[..., 1] * dy
        dot01 = g01[..., 0] * dx + g01[..., 1] * (dy - 1.0)
        dot11 = g11[..., 0] * (dx - 1.0) + g11[..., 1] * (dy - 1.0)

        # Smooth weights
        u = self._fade(dx)
        v = self._fade(dy)

        # Bilinear interpolation
        nx0 = dot00 + u * (dot10 - dot00)
        nx1 = dot01 + u * (dot11 - dot01)
        return nx0 + v * (nx1 - nx0)

    def fbm(
        self,
        x: np.ndarray,
        y: np.ndarray,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> np.ndarray:
        """Fractal Brownian Motion (fBm) multi-octave noise."""
        total = np.zeros_like(x, dtype=np.float64)
        frequency = 1.0
        amplitude = 1.0
        max_amp = 0.0

        for _ in range(octaves):
            total += amplitude * self.sample(x * frequency, y * frequency)
            max_amp += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        return total / max_amp

    def turbulence(
        self,
        x: np.ndarray,
        y: np.ndarray,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> np.ndarray:
        """Turbulence: sum of absolute value octave noise."""
        total = np.zeros_like(x, dtype=np.float64)
        frequency = 1.0
        amplitude = 1.0
        max_amp = 0.0

        for _ in range(octaves):
            total += amplitude * np.abs(self.sample(x * frequency, y * frequency))
            max_amp += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        return total / max_amp


def _make_grid(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create normalized (y, x) coordinate grids in [0, 1]."""
    ys = np.linspace(0.0, 1.0, height, endpoint=False)
    xs = np.linspace(0.0, 1.0, width, endpoint=False)
    return np.meshgrid(xs, ys)


def generate_plain_texture(
    height: int,
    width: int,
    rng: Optional[np.random.Generator] = None,
    config: Optional[TextureConfig] = None,
) -> np.ndarray:
    """
    Generate plain/solid color texture with optional subtle gradient.

    Returns:
        np.ndarray: (height, width, 3) uint8 array.
    """
    if rng is None:
        rng = np.random.default_rng()

    palettes = [
        # Vibrant modern
        [[235, 87, 87], [242, 153, 74], [242, 201, 76], [39, 174, 96], [47, 128, 237]],
        # Earth & Terracotta
        [[184, 115, 51], [205, 133, 63], [139, 69, 19], [160, 82, 45], [210, 105, 30]],
        # Pastels
        [[255, 179, 186], [255, 223, 186], [255, 255, 186], [186, 255, 201], [186, 225, 255]],
        # Deep Neons
        [[255, 0, 128], [0, 255, 255], [128, 0, 255], [0, 255, 128], [255, 255, 0]],
    ]

    chosen_palette = palettes[rng.integers(len(palettes))]
    base_color = np.array(chosen_palette[rng.integers(len(chosen_palette))], dtype=np.float64)

    # Subtle gradient or uniform
    if rng.random() < 0.6:
        target_color = np.array(chosen_palette[rng.integers(len(chosen_palette))], dtype=np.float64)
        angle = rng.uniform(0, 2 * np.pi)
        gx, gy = np.cos(angle), np.sin(angle)
        x_grid, y_grid = _make_grid(height, width)
        grad = (x_grid * gx + y_grid * gy)
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        # Blend lightly (up to 30% shift)
        alpha = rng.uniform(0.1, 0.35)
        blended = (1.0 - alpha * grad[..., None]) * base_color + (alpha * grad[..., None]) * target_color
        return np.clip(blended, 0, 255).astype(np.uint8)
    else:
        tex = np.full((height, width, 3), base_color, dtype=np.uint8)
        return tex


def generate_wood_texture(
    height: int,
    width: int,
    rng: Optional[np.random.Generator] = None,
    config: Optional[TextureConfig] = None,
) -> np.ndarray:
    """
    Generate wood texture using concentric rings perturbed by 2D Perlin noise:
    I(x, y) = sin(k * r + alpha * Noise(x, y)).

    Returns:
        np.ndarray: (height, width, 3) uint8 array.
    """
    if rng is None:
        rng = np.random.default_rng()
    if config is None:
        config = TextureConfig()

    x_grid, y_grid = _make_grid(height, width)

    # Offset origin for rings
    ox = rng.uniform(-0.5, 1.5)
    oy = rng.uniform(-0.5, 1.5)
    r = np.sqrt((x_grid - ox) ** 2 + (y_grid - oy) ** 2)

    # Noise perturbation
    noise_gen = PerlinNoise2D(seed=int(rng.integers(1, 100000)))
    k = config.wood_frequency * rng.uniform(0.8, 1.3)
    alpha = config.wood_turbulence * rng.uniform(0.7, 1.4)

    noise_field = noise_gen.fbm(x_grid * 4.0, y_grid * 4.0, octaves=config.fbm_octaves)
    grain = np.sin(k * r + alpha * noise_field)  # in [-1, 1]
    norm_grain = 0.5 * (grain + 1.0)  # in [0, 1]

    # Wood color profiles (light sapwood, golden oak, dark walnut)
    wood_palettes = [
        # Golden oak
        (np.array([218, 160, 102], dtype=np.float64), np.array([120, 68, 33], dtype=np.float64)),
        # Rich Walnut
        (np.array([150, 105, 75], dtype=np.float64), np.array([65, 38, 22], dtype=np.float64)),
        # Warm Cedar
        (np.array([210, 140, 90], dtype=np.float64), np.array([140, 65, 35], dtype=np.float64)),
    ]
    color_light, color_dark = wood_palettes[rng.integers(len(wood_palettes))]

    # Interpolate colors with sharp grain rings
    t = norm_grain[..., None]
    tex = (1.0 - t) * color_light + t * color_dark

    # Fine micro-grain
    micro = noise_gen.sample(x_grid * 20.0, y_grid * 60.0) * 12.0
    tex += micro[..., None]

    return np.clip(tex, 0, 255).astype(np.uint8)


def generate_marble_texture(
    height: int,
    width: int,
    rng: Optional[np.random.Generator] = None,
    config: Optional[TextureConfig] = None,
) -> np.ndarray:
    """
    Generate marble texture with sine wave turbulence:
    I(x, y) = sin(k_x * x + k_y * y + beta * Turbulence(x, y)).

    Returns:
        np.ndarray: (height, width, 3) uint8 array.
    """
    if rng is None:
        rng = np.random.default_rng()
    if config is None:
        config = TextureConfig()

    x_grid, y_grid = _make_grid(height, width)

    kx, ky = config.marble_frequency
    kx *= rng.uniform(0.8, 1.2)
    ky *= rng.uniform(0.8, 1.2)
    beta = config.marble_turbulence * rng.uniform(0.8, 1.3)

    noise_gen = PerlinNoise2D(seed=int(rng.integers(1, 100000)))
    turb = noise_gen.turbulence(x_grid * 3.0, y_grid * 3.0, octaves=config.fbm_octaves)

    sine_val = np.sin(kx * x_grid + ky * y_grid + beta * turb)
    # Marble vein curve: narrow vein valleys
    vein = np.abs(sine_val) ** 0.85
    t = vein[..., None]

    marble_palettes = [
        # Carrara White & Gray Veins
        (np.array([245, 245, 248], dtype=np.float64), np.array([60, 65, 75], dtype=np.float64)),
        # Nero Marquina (Black & White Veins)
        (np.array([28, 28, 32], dtype=np.float64), np.array([230, 230, 240], dtype=np.float64)),
        # Jade Green Marble
        (np.array([45, 95, 75], dtype=np.float64), np.array([190, 220, 205], dtype=np.float64)),
        # Gold Vein Calacatta
        (np.array([242, 240, 235], dtype=np.float64), np.array([195, 145, 60], dtype=np.float64)),
    ]
    base_col, vein_col = marble_palettes[rng.integers(len(marble_palettes))]

    tex = t * base_col + (1.0 - t) * vein_col
    return np.clip(tex, 0, 255).astype(np.uint8)


def generate_fur_texture(
    height: int,
    width: int,
    rng: Optional[np.random.Generator] = None,
    config: Optional[TextureConfig] = None,
) -> np.ndarray:
    """
    Generate fur/fibers texture using high-frequency directional anisotropic noise
    stretched along one axis with slight local angular variation.

    Returns:
        np.ndarray: (height, width, 3) uint8 array.
    """
    if rng is None:
        rng = np.random.default_rng()
    if config is None:
        config = TextureConfig()

    x_grid, y_grid = _make_grid(height, width)
    noise_gen = PerlinNoise2D(seed=int(rng.integers(1, 100000)))

    # Base directional angle
    flow_angle = rng.uniform(0.0, 2 * np.pi)

    # Angular perturbation map
    angle_noise = noise_gen.sample(x_grid * 3.0, y_grid * 3.0) * 0.4
    theta = flow_angle + angle_noise

    # Coordinates rotated along flow
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    u = x_grid * cos_t + y_grid * sin_t
    v = -x_grid * sin_t + y_grid * cos_t

    # Stretch along u (fibers length) and high frequency along v (fibers density)
    stretch = config.fur_stretch_ratio * rng.uniform(0.8, 1.2)
    freq = config.fur_frequency * rng.uniform(0.8, 1.2)

    fiber_noise = noise_gen.fbm(u * (freq / stretch), v * freq, octaves=3, persistence=0.6)
    fine_noise = noise_gen.sample(u * 8.0, v * (freq * 2.0))

    combined = 0.7 * fiber_noise + 0.3 * fine_noise
    norm_fur = 0.5 * (combined + 1.0)
    norm_fur = np.clip(norm_fur, 0.0, 1.0)[..., None]

    fur_palettes = [
        # Tawny / Lion Gold
        (np.array([125, 75, 35], dtype=np.float64), np.array([225, 175, 110], dtype=np.float64)),
        # Silver / Wolf Gray
        (np.array([60, 60, 65], dtype=np.float64), np.array([210, 215, 220], dtype=np.float64)),
        # Sable / Dark Brown
        (np.array([40, 25, 18], dtype=np.float64), np.array([115, 80, 55], dtype=np.float64)),
        # Polar Cream
        (np.array([200, 195, 185], dtype=np.float64), np.array([250, 248, 242], dtype=np.float64)),
    ]
    undercoat, guard_hair = fur_palettes[rng.integers(len(fur_palettes))]

    tex = (1.0 - norm_fur) * undercoat + norm_fur * guard_hair
    return np.clip(tex, 0, 255).astype(np.uint8)


def generate_texture(
    texture_type: str,
    height: int,
    width: int,
    rng: Optional[np.random.Generator] = None,
    config: Optional[TextureConfig] = None,
) -> np.ndarray:
    """
    Synthesize texture of given type and dimensions.

    Args:
        texture_type: 'plain', 'wood', 'marble', or 'fur'.
        height: Image height in pixels.
        width: Image width in pixels.
        rng: Optional numpy RandomGenerator.
        config: Optional TextureConfig.

    Returns:
        np.ndarray: (height, width, 3) uint8 RGB array.
    """
    tt = texture_type.lower()
    if tt == TextureType.PLAIN.value:
        return generate_plain_texture(height, width, rng=rng, config=config)
    elif tt == TextureType.WOOD.value:
        return generate_wood_texture(height, width, rng=rng, config=config)
    elif tt == TextureType.MARBLE.value:
        return generate_marble_texture(height, width, rng=rng, config=config)
    elif tt == TextureType.FUR.value:
        return generate_fur_texture(height, width, rng=rng, config=config)
    else:
        raise ValueError(f"Unknown texture type: '{texture_type}'. Supported: plain, wood, marble, fur.")
