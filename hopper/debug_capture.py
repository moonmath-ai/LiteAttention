"""
Offline utilities for loading and rendering LiteAttention debug capture files.

Usage::

    from lite_attention.debug_capture import load_capture, render_skip_images

    data = load_capture("debug.pt")
    render_skip_images(data, output_dir="./vis/")
"""

import os
from pathlib import Path

import torch


def load_capture(path: str | Path) -> dict:
    """Load a debug capture file produced by ``LiteAttentionRegistry.save()``.

    Args:
        path: Path to the ``.pt`` file.

    Returns:
        Dictionary with ``"metadata"`` and ``"modules"`` keys.
        See the capture file format specification for the full schema.
    """
    return torch.load(path, weights_only=True)


def _decode_skip_list_ranges(row: torch.Tensor):
    """Decode one skip list row into a list of (start_tile, end_tile) ranges.

    Args:
        row: 1-D tensor of shape ``[ktiles + 1]``.
            Format: ``[length, r1, r2, r1, r2, ..., uninit...]``

    Returns:
        List of ``(start_tile, end_tile)`` tuples where ``start_tile < end_tile``
        and the range is inclusive of start, exclusive of end (in tile units).
    """
    length = int(row[0].item())
    if length < 2:
        return []

    entries = row[1 : length + 1]
    # Detect step: first pair tells us the direction
    r1, r2 = entries[0].item(), entries[1].item()
    step = 1 if r1 > r2 else 0

    # Truncate to even number of entries (pairs of start/end)
    n_even = len(entries) - len(entries) % 2
    pairs = (entries[:n_even] + step).view(-1, 2)
    ranges = []
    for r1, r2 in pairs:
        lo = min(int(r1.item()), int(r2.item()))
        hi = max(int(r1.item()), int(r2.item()))
        ranges.append((lo, hi))
    return ranges


def skip_list_to_mask(skip_list_row: torch.Tensor, ktiles: int) -> torch.Tensor:
    """Decode a 2-D skip list slice into a binary mask.

    Args:
        skip_list_row: Tensor of shape ``[qtiles, ktiles + 1]`` — one skip list
            for a single (batch, head) pair.
        ktiles: Number of key tiles (``skip_list_row.shape[-1] - 1``).

    Returns:
        Boolean tensor of shape ``[qtiles, ktiles]`` where ``True`` means the
        tile was computed (not skipped).
    """
    qtiles = skip_list_row.shape[0]
    mask = torch.zeros(qtiles, ktiles, dtype=torch.bool)
    for qi in range(qtiles):
        for lo, hi in _decode_skip_list_ranges(skip_list_row[qi]):
            mask[qi, lo:hi] = True
    return mask


def render_skip_images(
    data: dict,
    output_dir: str | Path,
    max_res: int = 520,
) -> None:
    """Render captured skip list data as PNG images with attention map overlays.

    For each ``(module, timestep, batch, head)`` combination, produces a PNG
    showing the downsampled attention map (viridis colormap) with white
    semi-transparent rectangles overlaid for computed tiles and a tile grid.

    Args:
        data: Dictionary returned by :func:`load_capture`.
        output_dir: Root directory for output PNGs.  Directory tree is::

            {output_dir}/{module_name}/batch_{b}/head_{h}/t_{t:04d}.png

        max_res: Resolution for the output image (default 520).
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    for mod_name, mod_data in data["modules"].items():
        kBlockM = mod_data["kBlockM"]
        kBlockN = mod_data["kBlockN"]
        timesteps = mod_data["timesteps"]
        heads = mod_data["heads"]
        batch_indices = mod_data["batch_indices"]
        skip_lists = mod_data[
            "skip_lists"
        ]  # [n_captured, n_batch, n_heads, qtiles, ktiles+1]
        pct_per_head = mod_data["pct_per_head"]  # [n_captured, n_batch, n_heads]
        if "attn_maps" not in mod_data:
            raise ValueError(
                f"Module {mod_name!r} has no attention maps. "
                "Re-capture with attn_map_res > 0 to include them."
            )
        attn_maps = mod_data["attn_maps"]  # [n_captured, n_batch, n_heads, res, res]
        seq_len_q = mod_data["seq_len_q"]
        seq_len_k = mod_data["seq_len_k"]

        # Grid geometry (map tile coordinates to pixel coordinates)
        height, width = max_res, max_res
        ratio_h = height / seq_len_q
        ratio_w = width / seq_len_k
        grid_h = kBlockM * ratio_h
        grid_w = kBlockN * ratio_w

        y_positions = [
            b * grid_h for b in range(int(height / grid_h) + 1) if b * grid_h <= height
        ]
        x_positions = [
            b * grid_w for b in range(int(width / grid_w) + 1) if b * grid_w <= width
        ]

        for ti, t in enumerate(timesteps):
            t_val = int(t.item())
            for bi_idx, b in enumerate(batch_indices):
                b_val = int(b.item())
                for hi_idx, h in enumerate(heads):
                    h_val = int(h.item())

                    img_dir = output_dir / mod_name / f"batch_{b_val}" / f"head_{h_val}"
                    os.makedirs(img_dir, exist_ok=True)

                    attn_map = attn_maps[ti, bi_idx, hi_idx].float()
                    pct = float(pct_per_head[ti, bi_idx, hi_idx].item()) * 100

                    plt.figure(figsize=(6, 6))
                    plt.imshow(
                        attn_map.numpy(), cmap="viridis", interpolation="nearest"
                    )
                    plt.title(
                        f"{mod_name} | t={t_val} | batch {b_val} | head {h_val} | {pct:.1f}% computed"
                    )

                    for y in y_positions:
                        plt.axhline(y=y - 0.5, color="black", linewidth=0.2, alpha=0.7)
                    for x in x_positions:
                        plt.axvline(x=x - 0.5, color="black", linewidth=0.2, alpha=0.7)

                    # Overlay computed tile rectangles
                    row_skip = skip_lists[ti, bi_idx, hi_idx]  # [qtiles, ktiles+1]
                    for qi, row in enumerate(row_skip):
                        for lo, hi in _decode_skip_list_ranges(row):
                            rect = plt.Rectangle(
                                (lo * grid_w, qi * grid_h),
                                (hi - lo) * grid_w,
                                grid_h,
                                facecolor="white",
                                edgecolor="none",
                                linewidth=0.4,
                                alpha=0.3,
                            )
                            plt.gca().add_patch(rect)

                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(img_dir / f"t_{t_val:04d}.png", dpi=150)
                    plt.close()


def to_xarray(data: dict):
    """Convert capture data to an ``xarray.Dataset`` with labeled dimensions.

    Requires ``xarray`` to be installed. Dimensions are
    ``(module, timestep, batch, head, qtile, ktile_entry)``.

    Args:
        data: Dictionary returned by :func:`load_capture`.

    Returns:
        ``xarray.Dataset`` with ``skip_lists``, ``pct_per_head``, and
        ``attn_maps`` as data variables.
    """
    import xarray as xr

    datasets = {}
    for mod_name, mod_data in data["modules"].items():
        timesteps = mod_data["timesteps"].numpy()
        heads = mod_data["heads"].numpy()
        batch_indices = mod_data["batch_indices"].numpy()

        data_vars = {
            "skip_lists": xr.DataArray(
                mod_data["skip_lists"].numpy(),
                dims=["timestep", "batch", "head", "qtile", "ktile_entry"],
                coords={"timestep": timesteps, "batch": batch_indices, "head": heads},
            ),
            "pct_per_head": xr.DataArray(
                mod_data["pct_per_head"].numpy(),
                dims=["timestep", "batch", "head"],
                coords={"timestep": timesteps, "batch": batch_indices, "head": heads},
            ),
        }
        if "attn_maps" in mod_data:
            data_vars["attn_maps"] = xr.DataArray(
                mod_data["attn_maps"].numpy(),
                dims=["timestep", "batch", "head", "attn_y", "attn_x"],
                coords={"timestep": timesteps, "batch": batch_indices, "head": heads},
            )

        ds = xr.Dataset(
            data_vars,
            attrs={
                "module_name": mod_name,
                "seq_len_q": mod_data["seq_len_q"],
                "seq_len_k": mod_data["seq_len_k"],
                "head_dim": mod_data["head_dim"],
                "use_int8": mod_data["use_int8"],
            },
        )
        datasets[mod_name] = ds

    return datasets
