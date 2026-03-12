"""Unit tests for LiteAttentionRegistry._qk_map_to_replay_skip_lists.

All tests are CPU-only — no GPU required.
"""

import pytest
import torch
from lite_attention.calibrated_module import LiteAttentionRegistry


def decoded_compute_tiles(row: torch.Tensor, phase_true: bool) -> set[int]:
    """Decode a skip list row into the set of tile indices it computes.

    Args:
        row: int16 tensor of shape [ktiles+2]. row[0] is length (2*n_ranges),
             followed by reversed range pairs.
        phase_true: Whether the encoding used phase=True (end, start-1) or
                    phase=False (start, end+1).

    Returns:
        Set of tile indices that would be computed.
    """
    length = row[0].item()
    n_ranges = length // 2
    tiles: set[int] = set()
    for i in range(n_ranges):
        slot = 1 + i * 2
        val0 = row[slot].item()
        val1 = row[slot + 1].item()
        if phase_true:
            # val0 = end, val1 = start - 1
            start = val1 + 1
            end = val0
            tiles.update(range(start, end + 1))
        else:
            # val0 = start, val1 = end + 1
            start = val0
            end = val1 - 1
            tiles.update(range(start, end + 1))
    return tiles


def _run(qk_block_map: torch.Tensor, threshold: float, n_disabled: int = 0):
    """Shortcut to call _qk_map_to_replay_skip_lists."""
    return LiteAttentionRegistry._qk_map_to_replay_skip_lists(
        qk_block_map, threshold, n_disabled=n_disabled
    )


def _phase_for_timestep(t: int, n_disabled: int) -> bool:
    """Return the phase_true value for a given timestep index."""
    replay_idx = t - n_disabled + 1
    return replay_idx % 2 == 0


class TestAllTilesAboveThreshold:
    """All tiles are computed when all values >= threshold."""

    def test_basic(self):
        # 4 ktiles, all zeros, threshold=-1 → all above
        qk = torch.zeros(1, 1, 1, 1, 4)
        result = _run(qk, threshold=-1.0)

        # result[0] = compute_all, result[1] = skip list for t=0
        assert len(result) == 2
        for phase_true in [True, False]:
            # Both should give same decode; check with actual phase
            tiles = decoded_compute_tiles(result[1][0, 0, 0], _phase_for_timestep(0, 0))
            assert tiles == {0, 1, 2, 3}

    def test_compute_all_buffer(self):
        """The initial compute_all buffer covers all tiles."""
        qk = torch.zeros(1, 1, 1, 1, 4)
        result = _run(qk, threshold=-1.0)
        buf = result[0]
        # phase of compute_all is always phase_true (replay_idx=0 → even → True)
        # Actually compute_all is index 0 which is always used as-is.
        # It's encoded as: length=2, then (ktiles-1, -1) which is phase_true encoding
        assert buf[0, 0, 0, 0].item() == 2  # length
        assert buf[0, 0, 0, 1].item() == 3  # ktiles - 1 = end
        assert buf[0, 0, 0, 2].item() == -1  # start - 1


class TestNoTilesAboveThreshold:
    """All tiles skipped when all values < threshold."""

    def test_basic(self):
        qk = torch.full((1, 1, 1, 1, 4), -10.0)
        result = _run(qk, threshold=-5.0)
        assert len(result) == 2
        skip_list = result[1]
        # Length should be 0
        assert skip_list[0, 0, 0, 0].item() == 0
        tiles = decoded_compute_tiles(skip_list[0, 0, 0], _phase_for_timestep(0, 0))
        assert tiles == set()


class TestSingleTileAboveThreshold:
    """Only one tile above threshold in the middle."""

    def test_basic(self):
        qk = torch.full((1, 1, 1, 1, 4), -10.0)
        qk[0, 0, 0, 0, 2] = 0.0  # tile 2 above threshold
        result = _run(qk, threshold=-5.0)
        tiles = decoded_compute_tiles(result[1][0, 0, 0], _phase_for_timestep(0, 0))
        assert tiles == {2}


class TestTwoSeparateRanges:
    """Two non-contiguous ranges of tiles above threshold."""

    def test_basic(self):
        # 8 ktiles: tiles 0,1 and 5,6,7 above threshold
        qk = torch.full((1, 1, 1, 1, 8), -10.0)
        qk[0, 0, 0, 0, 0] = 0.0
        qk[0, 0, 0, 0, 1] = 0.0
        qk[0, 0, 0, 0, 5] = 0.0
        qk[0, 0, 0, 0, 6] = 0.0
        qk[0, 0, 0, 0, 7] = 0.0
        result = _run(qk, threshold=-5.0)
        skip_list = result[1]
        # Length should be 4 (2 ranges * 2)
        assert skip_list[0, 0, 0, 0].item() == 4
        tiles = decoded_compute_tiles(skip_list[0, 0, 0], _phase_for_timestep(0, 0))
        assert tiles == {0, 1, 5, 6, 7}


class TestThresholdExactlyAtBoundary:
    """Values exactly equal to threshold should be computed (>= comparison)."""

    def test_basic(self):
        qk = torch.full((1, 1, 1, 1, 4), -5.0)
        # All tiles are exactly at threshold → should be computed
        result = _run(qk, threshold=-5.0)
        tiles = decoded_compute_tiles(result[1][0, 0, 0], _phase_for_timestep(0, 0))
        assert tiles == {0, 1, 2, 3}

    def test_just_below(self):
        qk = torch.tensor([[[[[-5.0, -5.1, -4.9, -5.0]]]]])
        result = _run(qk, threshold=-5.0)
        tiles = decoded_compute_tiles(result[1][0, 0, 0], _phase_for_timestep(0, 0))
        # tile 1 is -5.1 < -5.0, so skipped; tile 2 is -4.9 >= -5.0, computed
        assert tiles == {0, 2, 3}


class TestPhaseAlternation:
    """Multiple timesteps alternate phase encoding; both decode the same."""

    def test_basic(self):
        # T=4, each timestep has tiles 1,2 above threshold
        T, B, H, q, k = 4, 1, 1, 1, 4
        qk = torch.full((T, B, H, q, k), -10.0)
        for t in range(T):
            qk[t, 0, 0, 0, 1] = 0.0
            qk[t, 0, 0, 0, 2] = 0.0

        result = _run(qk, threshold=-5.0)
        assert len(result) == T + 1  # compute_all + T skip lists

        for t in range(T):
            phase = _phase_for_timestep(t, 0)
            tiles = decoded_compute_tiles(result[t + 1][0, 0, 0], phase)
            assert tiles == {1, 2}, f"Failed at t={t}, phase_true={phase}"

    def test_encoding_differs_by_phase(self):
        """The raw encoded values differ between phases even though tiles are the same."""
        T = 2
        qk = torch.full((T, 1, 1, 1, 4), -10.0)
        for t in range(T):
            qk[t, 0, 0, 0, 1] = 0.0
            qk[t, 0, 0, 0, 2] = 0.0

        result = _run(qk, threshold=-5.0)
        row_t0 = result[1][0, 0, 0]
        row_t1 = result[2][0, 0, 0]

        # Both should decode to {1, 2}
        assert decoded_compute_tiles(row_t0, _phase_for_timestep(0, 0)) == {1, 2}
        assert decoded_compute_tiles(row_t1, _phase_for_timestep(1, 0)) == {1, 2}

        # But the raw encoded pair values should differ
        phase0 = _phase_for_timestep(0, 0)
        phase1 = _phase_for_timestep(1, 0)
        assert phase0 != phase1, "Adjacent timesteps should have different phases"
        assert not torch.equal(row_t0, row_t1), (
            "Raw encoding should differ between phases"
        )


class TestNDisabledSkipsLeadingTimesteps:
    """n_disabled skips leading timesteps."""

    def test_basic(self):
        T = 4
        qk = torch.zeros(T, 1, 1, 1, 4)
        result = _run(qk, threshold=-1.0, n_disabled=2)
        # compute_all + (T - n_disabled) skip lists = 1 + 2 = 3
        assert len(result) == 3

    def test_phases_start_from_disabled_offset(self):
        T = 4
        qk = torch.full((T, 1, 1, 1, 4), -10.0)
        for t in range(T):
            qk[t, 0, 0, 0, 0] = 0.0

        result = _run(qk, threshold=-5.0, n_disabled=2)
        # Skip lists are for t=2, t=3
        for i, t in enumerate([2, 3]):
            phase = _phase_for_timestep(t, 2)
            tiles = decoded_compute_tiles(result[i + 1][0, 0, 0], phase)
            assert tiles == {0}, f"Failed at t={t}"


class TestMultipleBatchHeadsQtiles:
    """Different patterns per (b, h, q) row are independent."""

    def test_basic(self):
        B, H, q, k = 2, 2, 2, 4
        qk = torch.full((1, B, H, q, k), -10.0)
        # (b=0, h=0, q=0): tile 0
        qk[0, 0, 0, 0, 0] = 0.0
        # (b=0, h=0, q=1): tiles 2,3
        qk[0, 0, 0, 1, 2] = 0.0
        qk[0, 0, 0, 1, 3] = 0.0
        # (b=1, h=1, q=0): all tiles
        qk[0, 1, 1, 0, :] = 0.0
        # (b=0, h=1, q=0): no tiles (all -10)
        # (b=1, h=0, q=0): no tiles
        # etc.

        result = _run(qk, threshold=-5.0)
        sl = result[1]
        phase = _phase_for_timestep(0, 0)

        assert decoded_compute_tiles(sl[0, 0, 0], phase) == {0}
        assert decoded_compute_tiles(sl[0, 0, 1], phase) == {2, 3}
        assert decoded_compute_tiles(sl[1, 1, 0], phase) == {0, 1, 2, 3}
        assert decoded_compute_tiles(sl[0, 1, 0], phase) == set()
        assert decoded_compute_tiles(sl[1, 0, 0], phase) == set()


class TestComputeAllInitialBuffer:
    """First element of result is always the compute_all buffer."""

    def test_shape_and_values(self):
        ktiles = 6
        B, H, q = 2, 3, 2
        qk = torch.zeros(1, B, H, q, ktiles)
        result = _run(qk, threshold=-1.0)
        buf = result[0]
        assert buf.shape == (B, H, q, ktiles + 2)
        assert buf.dtype == torch.int16
        # Every row: length=2, end=ktiles-1, start-1=-1
        assert (buf[:, :, :, 0] == 2).all()
        assert (buf[:, :, :, 1] == ktiles - 1).all()
        assert (buf[:, :, :, 2] == -1).all()

    def test_even_when_all_skipped(self):
        qk = torch.full((1, 1, 1, 1, 4), -10.0)
        result = _run(qk, threshold=-5.0)
        buf = result[0]
        assert buf[0, 0, 0, 0].item() == 2
        assert buf[0, 0, 0, 1].item() == 3
        assert buf[0, 0, 0, 2].item() == -1


class TestVaryingRangesPerRow:
    """Different qtile rows within the same timestep have different range counts."""

    def test_basic(self):
        k = 8
        qk = torch.full((1, 1, 1, 2, k), -10.0)
        # q=0: 1 contiguous range (tiles 2,3,4)
        qk[0, 0, 0, 0, 2] = 0.0
        qk[0, 0, 0, 0, 3] = 0.0
        qk[0, 0, 0, 0, 4] = 0.0
        # q=1: 3 separate ranges (tile 0), (tile 3), (tiles 6,7)
        qk[0, 0, 0, 1, 0] = 0.0
        qk[0, 0, 0, 1, 3] = 0.0
        qk[0, 0, 0, 1, 6] = 0.0
        qk[0, 0, 0, 1, 7] = 0.0

        result = _run(qk, threshold=-5.0)
        sl = result[1]
        phase = _phase_for_timestep(0, 0)

        tiles_q0 = decoded_compute_tiles(sl[0, 0, 0], phase)
        tiles_q1 = decoded_compute_tiles(sl[0, 0, 1], phase)

        assert tiles_q0 == {2, 3, 4}
        assert tiles_q1 == {0, 3, 6, 7}

        # q=0 has 1 range (length=2), q=1 has 3 ranges (length=6)
        assert sl[0, 0, 0, 0].item() == 2
        assert sl[0, 0, 1, 0].item() == 6
