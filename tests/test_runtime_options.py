"""Regression checks for launch-time data overrides and task lookup."""
import pytest
from omegaconf import OmegaConf

from pmt_tasks.registry_gym import CONFIGS_TASK_DIR, gym_id_for_stem, task_stem_for_id
from pmt_tasks.runtime_options import task_overrides


def test_motion_override_preserves_paths_with_spaces_and_punctuation(tmp_path):
    clip = tmp_path / "walk [dance], take=1.npz"
    clip.touch()
    cfg = OmegaConf.from_dotlist(task_overrides(["experiment_name=smoke"], str(clip)))
    assert cfg.motion.motion_files == str(clip)
    assert cfg.experiment_name == "smoke"


def test_missing_motion_and_unknown_flags_fail_before_simulation(tmp_path):
    with pytest.raises(FileNotFoundError):
        task_overrides([], str(tmp_path / "missing.npz"))
    with pytest.raises(ValueError, match="Unsupported arguments"):
        task_overrides(["--misspelled_flag=true"])


def test_all_registered_yaml_ids_round_trip():
    for path in CONFIGS_TASK_DIR.glob("*.yaml"):
        assert task_stem_for_id(gym_id_for_stem(path.stem)) == path.stem
    with pytest.raises(ValueError, match="Unknown PMT task"):
        task_stem_for_id("PMT-does-not-exist-v0")
