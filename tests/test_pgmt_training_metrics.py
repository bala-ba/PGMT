import pytest
import torch

from pmt_tasks.pgmt_training_metrics import RewardWindow


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reward_window_preserves_pre_reset_cell_and_dt(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    w = RewardWindow(["upper_a", "terrain_b", "auxiliary_c"], ["upper", "terrain", "auxiliary"],
                     ["flat", "stairs"], 2, device=device)
    family = torch.tensor([0, 1], device=device)
    level = torch.tensor([1, 0], device=device)
    w.add(torch.tensor([[.1, -.2, -.3], [.5, -.6, -.7]], device=device), family, level)
    # The next step can belong to a new birth tile after a terminal reset.
    family[:] = 0; level[:] = 0
    w.add(torch.tensor([[.2, -.4, -.6], [1., -1.2, -1.4]], device=device), family, level)
    result = w.consume(.02)
    assert result["Window/transitions"] == 4
    assert result["RewardStep/upper_a"] == pytest.approx(.45)
    assert result["RewardRate/upper_a"] == pytest.approx(22.5)
    assert result["TerrainReward/stairs/L0/upper"] == pytest.approx(.5)
    assert result["TerrainReward/flat/L1/upper"] == pytest.approx(.1)
    assert result["TerrainReward/flat/L0/upper"] == pytest.approx(.6)
    assert "TerrainReward/stairs/L1/upper" not in result
    assert result["RewardStep/total"] == pytest.approx(sum(result[f"RewardGroupStep/{g}"] for g in w.groups))
    assert w.consume(.02) == {}
