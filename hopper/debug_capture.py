# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib>=3.10.8",
#     "torch>=2.10.0",
# ]
# ///
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
        Dictionary with ``"modules"`` key. Each module has ``pct_per_head``
        (all timesteps/heads) and optionally ``attn_maps``/``skip_lists``
        for a filtered subset.
    """
    return torch.load(path, weights_only=True, map_location="cpu")


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
) -> None:
    """Render captured skip list data as PNG images with attention map overlays.

    Only renders modules that have attn maps (tier 2 capture). Modules with
    only pct_per_head are silently skipped.

    For each ``(module, timestep, batch, head)`` combination that has attn maps,
    produces a PNG showing the downsampled attention map (viridis colormap) with
    white semi-transparent rectangles overlaid for computed tiles and a tile grid.

    Args:
        data: Dictionary returned by :func:`load_capture`.
        output_dir: Root directory for output PNGs.  Directory tree is::

            {output_dir}/{module_name}/batch_{b}/head_{h}/t_{t:04d}.png

    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    for mod_name, mod_data in data["modules"].items():
        if "attn_maps" not in mod_data:
            continue

        kBlockM = mod_data["kBlockM"]
        kBlockN = mod_data["kBlockN"]
        map_timesteps = mod_data["map_timesteps"]
        map_heads = mod_data["map_heads"]
        map_batch_indices = mod_data["map_batch_indices"]
        skip_lists = mod_data["skip_lists"]
        attn_maps = mod_data["attn_maps"]
        seq_len_q = mod_data["seq_len_q"]
        seq_len_k = mod_data["seq_len_k"]

        # For pct overlay: look up from the full pct_per_head
        all_timesteps = mod_data["timesteps"]
        pct_per_head_all = mod_data["pct_per_head"]  # [T_all, B, H_all]

        # Grid geometry — use the actual attn map resolution so that imshow
        # pixel coordinates match the grid/rectangle coordinates exactly
        # (same coordinate system as visualize_skips).
        height, width = attn_maps.shape[-2], attn_maps.shape[-1]
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

        for ti, t in enumerate(map_timesteps):
            t_val = int(t.item())
            for bi_idx, b in enumerate(map_batch_indices):
                b_val = int(b.item())
                for hi_idx, h in enumerate(map_heads):
                    h_val = int(h.item())

                    img_dir = output_dir / mod_name / f"batch_{b_val}" / f"head_{h_val}"
                    os.makedirs(img_dir, exist_ok=True)

                    attn_map = attn_maps[ti, bi_idx, hi_idx].float()

                    # Look up pct from pct_per_head_all
                    t_all_idx = (all_timesteps == t).nonzero(as_tuple=True)[0]
                    if t_all_idx.numel() > 0:
                        pct = (
                            float(pct_per_head_all[t_all_idx[0], b_val, h_val].item())
                            * 100
                        )
                    else:
                        pct = 0.0

                    plt.figure(figsize=(6, 6))
                    plt.imshow(
                        attn_map.numpy(), cmap="viridis", interpolation="nearest"
                    )
                    plt.title(
                        f"{mod_name}\nt={t_val} | batch {b_val} | head {h_val} | {pct:.1f}% computed"
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
    """Convert capture data to ``xarray.Dataset`` objects with labeled dimensions.

    Returns a dict mapping module names to datasets. Each dataset always has
    ``pct_per_head`` (all timesteps/heads). If attn maps were captured for the
    module, the dataset also contains ``attn_maps`` and ``skip_lists`` with
    their own coordinate arrays (``map_timestep``, ``map_head``, ``map_batch``).

    Args:
        data: Dictionary returned by :func:`load_capture`.

    Returns:
        Dict mapping module names to ``xarray.Dataset`` instances.
    """
    import xarray as xr

    datasets = {}
    for mod_name, mod_data in data["modules"].items():
        timesteps = mod_data["timesteps"].numpy()
        n_batch = mod_data["pct_per_head"].shape[1]
        n_heads = mod_data["pct_per_head"].shape[2]

        data_vars = {
            "pct_per_head": xr.DataArray(
                mod_data["pct_per_head"].numpy(),
                dims=["timestep", "batch", "head"],
                coords={
                    "timestep": timesteps,
                    "batch": list(range(n_batch)),
                    "head": list(range(n_heads)),
                },
            ),
        }

        if "attn_maps" in mod_data:
            map_ts = mod_data["map_timesteps"].numpy()
            map_heads = mod_data["map_heads"].numpy()
            map_batch = mod_data["map_batch_indices"].numpy()

            data_vars["skip_lists"] = xr.DataArray(
                mod_data["skip_lists"].numpy(),
                dims=["map_timestep", "map_batch", "map_head", "qtile", "ktile_entry"],
                coords={
                    "map_timestep": map_ts,
                    "map_batch": map_batch,
                    "map_head": map_heads,
                },
            )
            data_vars["attn_maps"] = xr.DataArray(
                mod_data["attn_maps"].numpy(),
                dims=["map_timestep", "map_batch", "map_head", "attn_y", "attn_x"],
                coords={
                    "map_timestep": map_ts,
                    "map_batch": map_batch,
                    "map_head": map_heads,
                },
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render LiteAttention capture .pt files to PNG images."
    )
    parser.add_argument(
        "pt_files", nargs="+", help="One or more .pt capture files to render."
    )
    args = parser.parse_args()

    for pt_file in args.pt_files:
        pt_path = Path(pt_file)
        out_dir = pt_path.with_name(pt_path.stem + "_vis")
        print(f"Loading {pt_path} ...")
        data = load_capture(pt_path)
        print(f"Rendering to {out_dir} ...")
        render_skip_images(data, out_dir)
        print(f"Done: {out_dir}")
