# Tile Culling in the CUDA Rasterizer

During the preprocessing pass, the rasterizer needs to decide which screen-space tiles each
Gaussian splat overlaps — this determines how many sort keys to emit and therefore how much
work the subsequent alpha-compositing kernel will do. The implementation uses a two-stage
approach: a fast coarse filter based on the Gaussian's longest axis, followed by a
per-tile test that accounts for the actual ellipse shape.

---

## Stage 1 — Coarse bounding rect from the largest eigenvalue

**File:** `src/rasterizer/cuda_rasterizer/forward.cu`, lines 296–301

```cpp
float mid = 0.5f * (cov.x + cov.z);
float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));  // largest eigenvalue of 2D cov
float my_radius = ceil(3.f * sqrt(lambda1));              // 3σ along the longest axis
float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
uint2 rect_min, rect_max;
getRect(point_image, my_radius, rect_min, rect_max, grid);
```

`cov` is the 2×2 projected covariance matrix (stored as three floats: `cov.x`, `cov.y`,
`cov.z` for the upper triangle). Its largest eigenvalue `lambda1` gives the variance along
the direction of maximum spread. Taking `3 * sqrt(lambda1)` gives a 3σ radius along that
axis — a circle large enough to enclose the entire Gaussian ellipse.

`getRect` (`auxiliary.h`, line 46) turns this circular radius into a rectangular range of
tile indices by intersecting a square of side `2 * my_radius` centred on the projected
mean with the tile grid. The result is a conservative axis-aligned set of tiles: every
tile that could possibly overlap the Gaussian is included, but many corner tiles will not
actually receive meaningful contribution because the Gaussian is elongated rather than
circular.

This is the step that the original 3DGS paper uses as its sole tile-assignment criterion.

---

## Stage 2 — Per-tile culling using the actual Gaussian ellipse

**File:** `src/rasterizer/cuda_rasterizer/forward.cu`, line 303

```cpp
const int tile_count = computeTilebasedCullingTileCount(
    active, co, point_image, opacity_factor_threshold, rect_min, rect_max);
```

`computeTilebasedCullingTileCount` iterates over every tile in the coarse rect and calls
`max_contrib_power_rect_gaussian_float` on each one. Only tiles that pass this test are
counted, and `tiles_touched[idx]` is set to `tile_count` rather than the full coarse rect
area.

### What `max_contrib_power_rect_gaussian_float` computes

**File:** `src/rasterizer/cuda_rasterizer/forward.h`, lines 39–78

The function answers the question: *what is the maximum opacity contribution this Gaussian
can make to any pixel inside this tile?*

The Gaussian's opacity at pixel `(x, y)` is:

```
α(x, y) = opacity * exp(-power)
power    = 0.5 * (Σ⁻¹₀₀ * dx² + Σ⁻¹₂₂ * dy²) + Σ⁻¹₀₁ * dx * dy
```

where `(dx, dy) = (x - μₓ, y - μᵧ)` and `Σ⁻¹` is the 2D conic (inverse covariance),
stored as `co = {Σ⁻¹₀₀, Σ⁻¹₀₁, Σ⁻¹₂₂, opacity}`. The exponent is maximised (opacity is
highest) where `power` is smallest, which is at the mean itself. So the maximum
contribution to a tile is achieved at the pixel closest to the Gaussian's projected mean
in the covariance-weighted metric.

The function works as follows:

**1. Determine the closest point on the tile to the Gaussian mean.**

```cpp
const float x_left       = x_min_diff > 0.0f;   // mean is left of tile
const float not_in_x_range = x_left + (mean.x > rect_max.x);

const float y_above      = y_min_diff > 0.0f;   // mean is above tile
const float not_in_y_range = y_above + (mean.y > rect_max.y);
```

If `not_in_x_range + not_in_y_range == 0`, the mean falls inside the tile, so `power = 0`
and the Gaussian fully contributes — no culling is possible, and the function returns 0
immediately (the early-exit `if` is skipped).

If the mean is outside in at least one axis, the closest corner of the tile is identified:

```cpp
const float px = x_left * rect_min.x + (1.0f - x_left) * rect_max.x;
const float py = y_above * rect_min.y + (1.0f - y_above) * rect_max.y;
```

**2. Slide along the tile edges to find the true closest point.**

The corner `(px, py)` is not necessarily the closest point — the closest point may lie on
an edge rather than a corner. The function solves for the optimal position along each edge
by minimising `power` subject to the constraint that the point stays on the tile boundary:

```cpp
const float tx = not_in_y_range * saturate((dx * co.x * diffx + dx * co.y * diffy) / (dx*dx*co.x));
const float ty = not_in_x_range * saturate((dy * co.y * diffx + dy * co.z * diffy) / (dy*dy*co.z));
max_pos = {px + tx * dx, py + ty * dy};
```

`tx` and `ty` are clamped to `[0, 1]` by `__saturatef`, so the result always stays within
the tile boundary. When the mean is outside in only one axis (e.g. above the tile but
within the x-range), only that axis's term is non-zero, and the closest point slides along
the corresponding edge.

**3. Evaluate the opacity exponent at that closest point.**

```cpp
const float2 max_pos_diff = {mean.x - max_pos.x, mean.y - max_pos.y};
max_contrib_power = evaluate_opacity_factor(max_pos_diff.x, max_pos_diff.y, co);
```

`evaluate_opacity_factor` computes `power = 0.5*(co.x*dx² + co.z*dy²) + co.y*dx*dy`.

### The culling decision

Back in `computeTilebasedCullingTileCount`:

```cpp
tile_count += (max_opac_factor <= opacity_power_threshold);
```

`opacity_power_threshold = log(opacity / OPACITY_THRESHOLD)`. If `max_contrib_power`
exceeds this threshold, then even the closest pixel in the tile would have
`exp(-power) < OPACITY_THRESHOLD / opacity` — i.e. the Gaussian is effectively
transparent across the entire tile — and the tile is excluded from `tile_count`.

Because this test respects the full conic shape, elongated Gaussians shed all the tiles
that fall in the "corners" of the coarse circular bounding rect but outside the ellipse.
For highly anisotropic splats this can eliminate a large fraction of the candidate tiles.

---

## Cooperative warp execution for large rects

When the coarse rect contains more than `SEQUENTIAL_TILE_THRESH = 32` tiles,
`computeTilebasedCullingTileCount` switches to a warp-cooperative loop. The per-Gaussian
data (`co`, `xy`, thresholds, rect bounds) are broadcast to all 32 lanes via
`__shfl_sync`, and the remaining tiles are distributed across lanes. Each lane tests one
tile per iteration and the results are aggregated with `__ballot_sync` / `__popc`. This
avoids serialising the per-tile work for large splats while keeping the computation on a
single warp.

---

## Summary

| Step | Location | What it computes |
|---|---|---|
| Eigenvalue radius | `forward.cu:296-298` | 3σ radius along longest axis of Gaussian |
| Coarse tile rect | `auxiliary.h:46-56` | Conservative axis-aligned tile range |
| Per-tile cull | `forward.h:39-78` | Max opacity contribution to each tile; drops tiles below threshold |
| Tile count output | `forward.cu:318` | Only truly-contributing tiles counted |
