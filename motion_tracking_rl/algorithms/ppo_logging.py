"""Detached PPO metric sums with one device-to-host read per update."""
from collections import defaultdict

import torch


class PPOLogAccumulator:
    def __init__(self, device):
        self.device = device
        self._sums = {}
        self._host_sums = defaultdict(dict)

    def add(self, group, values, *, dtype=torch.float64):
        """Snapshot scalar metrics without retaining a graph or reading CUDA.

        Double sums match the former Python-float accumulation. Four-head
        value losses explicitly use float32 to preserve their original sums.
        Optional metric sets may change between minibatches; absent terms still
        contribute zero under the caller's global minibatch denominator.
        """
        names, tensors = [], []
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(f"PPO log metric {group}/{name} must be scalar")
                names.append(name)
                tensors.append(value.detach().reshape(()).to(device=self.device))
            else:
                sums = self._host_sums[group]
                sums[name] = sums.get(name, 0.0) + float(value)
        if tensors:
            packed = torch.stack(tensors).to(device=self.device, dtype=dtype)
            key = (group, tuple(names), dtype)
            if key in self._sums:
                self._sums[key].add_(packed)
            else:
                self._sums[key] = packed

    def means(self, denominator):
        """Return ordinary Python floats; transfer all GPU groups together."""
        if denominator <= 0:
            raise ValueError("PPO log denominator must be positive")
        result = {group: {name: value / denominator for name, value in sums.items()}
                  for group, sums in self._host_sums.items()}
        if self._sums:
            keys = []
            means = []
            for (group, names, _), sums in self._sums.items():
                keys.extend((group, name) for name in names)
                means.append((sums / denominator).to(torch.float64))
            # The only host read in the routine metric path, regardless of the
            # number of minibatches, groups or heads. No persistent GPU buffers.
            values = torch.cat(means).cpu().tolist()
            for (group, name), value in zip(keys, values):
                metrics = result.setdefault(group, {})
                metrics[name] = metrics.get(name, 0.0) + value
        return result
