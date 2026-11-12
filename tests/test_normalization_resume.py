"""A rollout must not make observation-normalizer checkpoints unloadable."""
import copy

import torch

from motion_tracking_rl.networks.layers.normalization import EmpiricalNormalization


def test_load_after_inference_rollout_and_continue_statistics():
    normalizer = EmpiricalNormalization(3)
    batches = [torch.tensor([[1., 2., 3.], [3., -2., 7.]]),
               torch.tensor([[-1., 4., 2.], [5., 6., -3.]])]
    with torch.inference_mode():
        normalizer.update(batches[0])
    saved = copy.deepcopy(normalizer.state_dict())
    with torch.inference_mode():
        normalizer.update(batches[1])
    # This used to raise: _std was replaced by an inference tensor during update.
    normalizer.load_state_dict(saved)
    for key, value in normalizer.state_dict().items():
        torch.testing.assert_close(value, saved[key], atol=0, rtol=0)
    cold_resume = EmpiricalNormalization(3)
    cold_resume.load_state_dict(saved)
    with torch.inference_mode():
        normalizer.update(batches[1])
        cold_resume.update(batches[1])
    joined = torch.cat(batches)
    torch.testing.assert_close(normalizer.mean, joined.mean(0))
    torch.testing.assert_close(normalizer.std, joined.std(0, unbiased=False))
    for key, value in normalizer.state_dict().items():
        torch.testing.assert_close(value, cold_resume.state_dict()[key], atol=0, rtol=0)
