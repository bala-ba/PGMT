"""Logging aggregation must preserve means without host reads or autograd."""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from motion_tracking_rl.algorithms.ppo_logging import PPOLogAccumulator

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


class NoScalarReads(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("minibatch metric accumulation read a tensor scalar")
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("device", DEVICES)
def test_metrics_snapshot_precision_changing_keys_and_single_transfer(device, monkeypatch):
    collector = PPOLogAccumulator(device)
    source = torch.tensor([1e8, 1., -1e8], device=device, requires_grad=True)
    host_reads = []
    original_cpu = torch.Tensor.cpu
    def cpu(tensor, *args, **kwargs):
        host_reads.append(tensor.shape)
        return original_cpu(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    with NoScalarReads():
        for i in range(3):
            collector.add("loss", {"cancel": source[i], "host": .25})
            collector.add("heads", {"head": source[i]}, dtype=torch.float32)
            # Optional values may appear late or change representation.
            if i:
                collector.add("aux", {"late": source[i] * 0 + 6})
            collector.add("mixed", {"switch": source[i] * 0 + 3 if i else 3.})
        for sums in collector._sums.values():
            assert not sums.requires_grad and sums.grad_fn is None
        # Sums own snapshots, never a detached view into mutable model buffers.
        with torch.no_grad():
            source.fill_(42.)
    assert not host_reads
    result = collector.means(3)
    assert len(host_reads) == 1
    assert result == {"loss": {"cancel": 1 / 3, "host": .25}, "heads": {"head": 0.},
                      "aux": {"late": 4.}, "mixed": {"switch": 3.}}
    assert all(isinstance(value, float) for group in result.values() for value in group.values())


@pytest.mark.parametrize("device", DEVICES)
def test_logging_does_not_silently_drop_nonfinite_metrics(device):
    collector = PPOLogAccumulator(device)
    collector.add("loss", {"nan": torch.tensor(float("nan"), device=device),
                           "inf": torch.tensor(float("inf"), device=device)})
    result = collector.means(1)["loss"]
    assert result["nan"] != result["nan"] and result["inf"] == float("inf")


def test_empty_and_host_only_metrics_keep_zero_missing_semantics():
    collector = PPOLogAccumulator("cpu")
    assert collector.means(2) == {}
    collector.add("aux", {"scale": .1})
    collector.add("aux", {})
    assert collector.means(2) == {"aux": {"scale": .05}}
    with pytest.raises(ValueError, match="positive"):
        collector.means(0)
    with pytest.raises(ValueError, match="scalar"):
        collector.add("loss", {"bad": torch.ones(2)})
