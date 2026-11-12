"""Validate terrain-filtered sampling, runtime contact targets and native lifecycle."""
import argparse
import csv
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default=str(ROOT / "logs/pgmt_sampling"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
report_file = output / "validation.json"
report_file.unlink(missing_ok=True)
launcher = AppLauncher(args, multi_gpu=False, fast_shutdown=True)
app = launcher.app


def main():
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from pmt_tasks.env_cfgs.pgmt.placement import PGMTPlacementEnvCfg
    from pmt_tasks.mdp.commands.multi_motion_command import AdaptiveSampler
    from pmt_tasks.pgmt_runtime import get_pgmt_state

    cfg = PGMTPlacementEnvCfg()
    cfg.pmt_motion_paths = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1")]
    cfg.__post_init__()
    # This validator deliberately edits frames and sampler outcomes; the
    # terrain curriculum is tested separately with genuine episode boundaries.
    cfg.curriculum.terrain_progress = None
    cfg.scene.num_envs = 7
    cfg.seed = 715
    cfg.sim.device = args.device
    cfg.commands.motion.storage_device = args.device
    cfg.commands.motion.placement_xy_range = (0., 0.)
    cfg.observations.proprio.enable_corruption = False
    cfg.observations.proprio_history.enable_corruption = False
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    reward_contact_reads = []

    def contact_probe(env):
        # Validation-only zero reward: record the labels at the real manager
        # evaluation point, before terminal resets and command advancement.
        state = get_pgmt_state(env)
        reward_contact_reads.append(state.reference_contacts())
        assert state.robot_foot_contact_forces().shape == (7, 2, 3)
        return torch.zeros(env.num_envs, device=env.device)

    cfg.rewards.auxiliary_contact_interface_probe = RewardTermCfg(func=contact_probe, weight=1.)
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        command = env.command_manager.get_term("motion")
        sampler = command.sampler
        assert isinstance(sampler, AdaptiveSampler)
        assert command.cfg.sampler_type == "adaptive" and not sampler.enable_frame_sampling
        assert command.cfg.fixed_start_frame == 0
        assert command.cfg.require_valid_support_start is False
        assert not hasattr(sampler, "failed_frame_count") and not hasattr(sampler, "failed_bin_count")
        ids = torch.arange(7, device=env.device)
        terrain = env.scene.terrain
        terrain.terrain_types[:] = ids
        files = [Path(m.source_file).stem for m in command.data_store.motions]
        state = get_pgmt_state(env)
        labels = command.contact_label_store
        motions = command.data_store.motions
        # Compare every packed frame to its source metadata, in native loader
        # order. Loading contacts must not depend on the reset support gate.
        assert all(tuple(m.metadata["foot_names"].tolist()) == state.foot_names for m in motions)
        all_motion_ids = torch.repeat_interleave(torch.arange(len(files), device=env.device), labels.motion_lengths)
        all_frame_ids = torch.arange(len(all_motion_ids), device=env.device) - labels.offsets[all_motion_ids]
        all_labels = labels.gather(all_motion_ids, all_frame_ids)
        for key in ("contacts", "contact_valid"):
            expected = torch.as_tensor(np.concatenate([m.metadata[key] for m in motions]), device=env.device)
            torch.testing.assert_close(all_labels[key], expected, atol=0, rtol=0)
        contact_frames = len(all_motion_ids)
        invalid_foot_frames = int((~all_labels["contact_valid"]).sum())
        assert all_labels["frame_valid"].all()
        assert [env.scene["robot"].body_names[i] for i in state.foot_body_indices.tolist()] == list(state.foot_names)
        assert [env.scene["contact_forces"].body_names[i] for i in state.foot_sensor_indices.tolist()] == list(state.foot_names)
        del all_labels, all_motion_ids, all_frame_ids

        def check_current_contacts():
            result = state.reference_contacts()
            for key in ("contacts", "contact_valid"):
                expected = torch.as_tensor(np.stack([
                    motions[mid].metadata[key][frame]
                    for mid, frame in zip(command.motion_ids.tolist(), command.frame_ids.tolist())
                ]), device=env.device)
                torch.testing.assert_close(result[key], expected, atol=0, rtol=0)
            future = state.reference_contacts(future=True)
            assert future["contacts"].shape == (7, 8, 2)
            for key in result:
                torch.testing.assert_close(result[key], future[key][:, 0], atol=0, rtol=0)
            expected_frames = command.frame_ids[:, None] + command.frame_ids.new_tensor([0, 1, 3, 7, 15, 31, 63, 127])
            expected_valid = expected_frames < command.data_store.motion_lengths[command.motion_ids, None]
            assert torch.equal(future["frame_valid"], expected_valid)
            assert not future["contact_valid"][~expected_valid].any()
            torch.testing.assert_close(state.robot_foot_contact_forces(),
                env.scene["contact_forces"].data.net_forces_w[:, state.foot_sensor_indices], atol=0, rtol=0)
            return result

        def checked_step():
            before = check_current_contacts()
            count = len(reward_contact_reads)
            result = env.step(torch.zeros(7, env.action_manager.total_action_dim, device=env.device))
            assert len(reward_contact_reads) == count + 1
            for key in before:
                torch.testing.assert_close(reward_contact_reads[-1][key], before[key], atol=0, rtol=0)
            check_current_contacts()
            return result

        with Path(command.cfg.motion_terrain_compatibility).open(newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        # Independent check against the actual current CSV, including both
        # stairs variants. This table currently contains only blank or stairs.
        assert {row["dismatched terrain"] for row in rows} <= {"", "stairs"}
        banned_names = {row["motion_name"] for row in rows if row["dismatched terrain"] == "stairs"}
        banned_ids = torch.tensor([name in banned_names for name in files], device=env.device)
        stairs = torch.tensor([family in ("stairs", "up_stairs") for family in command.terrain_families], device=env.device)
        total_draws = 0
        sampled_motions = torch.zeros(len(files), dtype=torch.bool, device=env.device)
        # Support labels must not remove real frames from the current reset pool.
        actual_frames = torch.arange(command.valid_starts.shape[1], device=env.device)[None]
        assert torch.equal(command.valid_starts, actual_frames < command.data_store.motion_lengths[:, None])
        for _ in range(10):
            reset_ids = ids.repeat(1000)
            result = command._draw_samples(reset_ids)
            assert not (stairs[reset_ids] & banned_ids[result.motion_ids]).any()
            assert command.compatible_motions[reset_ids, result.motion_ids].all()
            assert command.valid_starts[result.motion_ids, result.frame_ids].all()
            assert result.frame_ids.eq(0).all()
            sampled_motions[result.motion_ids] = True
            total_draws += len(reset_ids)
        eligibility = command.compatible_motions & command.valid_starts.any(-1)[None]
        counts = dict(zip(command.terrain_families, eligibility.sum(-1).tolist()))
        full_mask = command.compatible_motions.clone()
        crawl_ids = [files.index(f"walk3_subject3_clip{clip}") for clip in (18, 22, 23)]
        assert sampled_motions[crawl_ids].all()
        assert not command.unavailable_reset_motions

        # Isolate native lifecycle checks to two known, physically validated
        # start poses. This narrows the CSV pool and never permits a banned pair.
        walk = files.index("walk1_subject1_clip0")
        dance = files.index("dance1_subject1_clip0")
        smoke_pool = torch.zeros(len(files), dtype=torch.bool, device=env.device)
        smoke_pool[[walk, dance]] = True
        command.compatible_motions &= smoke_pool[None]
        command.cfg.placement_yaw_range = (0., 0.)
        audited_resets = 0
        original_draw = command._draw_samples

        def audited_draw(env_ids):
            nonlocal audited_resets
            sample = original_draw(env_ids)
            assert full_mask[terrain.terrain_types[env_ids], sample.motion_ids].all()
            assert command.valid_starts[sample.motion_ids, sample.frame_ids].all()
            assert sample.frame_ids.eq(0).all()
            audited_resets += len(env_ids)
            return sample

        command._draw_samples = audited_draw
        env.reset()
        before_contacts = check_current_contacts()
        assert sampler.current_failed_motion_count.sum() == 0
        assert sampler.current_success_motion_count.sum() == 0
        partial = torch.tensor([6, 2, 0], device=env.device)
        untouched = torch.tensor([1, 3, 4, 5], device=env.device)
        before = get_pgmt_state(env).reference()["body_pos"].clone()
        command._resample_command(partial)
        torch.testing.assert_close(get_pgmt_state(env).reference()["body_pos"][untouched], before[untouched], atol=0, rtol=0)
        after_contacts = check_current_contacts()
        for key in before_contacts:
            torch.testing.assert_close(after_contacts[key][untouched], before_contacts[key][untouched], atol=0, rtol=0)
            torch.testing.assert_close(state.reference_contacts(env_ids=partial)[key], after_contacts[key][partial], atol=0, rtol=0)
        assert sampler.current_success_motion_count.sum() + sampler.current_failed_motion_count.sum() == len(partial)

        # Attribute outcomes to global *start motion* ids, with no frame/bin
        # counters, then execute the native EMA update through command advancement.
        sampler._reset_counts()
        command.start_motion_ids[:] = torch.tensor([walk, dance, walk, dance, walk, dance, walk], device=env.device)
        terminated = torch.tensor([True, False, True, False, True, False, True], device=env.device)
        command._record_outcomes(ids, terminated)
        assert sampler.current_failed_motion_count[walk] == 4
        assert sampler.current_success_motion_count[dance] == 3
        assert sampler.current_failed_motion_count.sum() == 4
        assert sampler.current_success_motion_count.sum() == 3
        old_failure = sampler.get_failure_probabilities().clone()
        sampler._steps_since_update = sampler.update_interval - 1
        command._update_command()
        new_failure = sampler.get_failure_probabilities()
        assert new_failure[walk] > old_failure[walk] and new_failure[dance] < old_failure[dance]

        # Restore real start/outcome correspondence before stepping physics.
        for level, yaw in ((0, 0.), (5, 1.5707963267948966)):
            terrain.terrain_levels[:] = level
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            command.cfg.placement_yaw_range = (yaw, yaw)
            env.reset()
            assert full_mask[terrain.terrain_types, command.motion_ids].all()
            for _ in range(5):
                checked_step()
                assert full_mask[terrain.terrain_types, command.motion_ids].all()

        # Force a clip boundary through the actual step lifecycle. The reward
        # must see the old last-frame labels; post-step reads follow the reset
        # and subsequent command advancement, never the old contact snapshot.
        command.frame_ids[0] = command.data_store.motion_lengths[command.motion_ids[0]] - 1
        command._update_motion_buffers_for_envs(ids[:1])
        state.invalidate_observations()
        final_frame = int(command.frame_ids[0])
        terminal_contacts = state.reference_contacts(future=True, env_ids=ids[:1])
        assert not terminal_contacts["frame_valid"][0, 1:].any()
        assert not terminal_contacts["contact_valid"][0, 1:].any()
        _, _, _, timeouts, _ = checked_step()
        assert timeouts[0] and command.frame_ids[0] == 1
        assert reward_contact_reads[-1]["frame_ids"][0] == final_frame

        # These three crawling segments have no foot-support start. Prove that
        # their actual frame-0 poses can now reach the native flat-terrain reset,
        # while the usual joint limits / collision placement checks still run.
        crawl_resets = []
        for motion_id in crawl_ids:
            only_motion = torch.zeros(len(files), dtype=torch.bool, device=env.device)
            only_motion[motion_id] = True
            command.compatible_motions = full_mask & only_motion[None]
            command._resample_command(ids[:1])
            assert command.motion_ids[0] == motion_id and command.frame_ids[0] == 0
            check_current_contacts()
            reference = get_pgmt_state(env).reference()
            torch.testing.assert_close(env.scene["robot"].data.joint_pos[0], reference["joint_pos"][0], atol=2e-6, rtol=0)
            torch.testing.assert_close(env.scene["robot"].data.body_pos_w[0], reference["body_pos"][0], atol=3e-4, rtol=0)
            crawl_resets.append(files[motion_id])

        report = dict(passed=True, total_motions=len(files), sampler_type=command.cfg.sampler_type,
                      enable_frame_sampling=command.cfg.enable_frame_sampling,
                      fixed_start_frame=command.cfg.fixed_start_frame,
                      require_valid_support_start=command.cfg.require_valid_support_start,
                      compatibility_file=command.cfg.motion_terrain_compatibility,
                      compatibility_sha256=command.compatibility_sha256,
                      csv_stairs_exclusions=len(banned_names), eligible_motion_counts=counts,
                      unavailable_reset_motions=[Path(p).name for p in command.unavailable_reset_motions],
                      full_library_draws=total_draws, incompatible_draws=0,
                      all_sampled_start_frames_zero=True, crawling_frame0_flat_resets=crawl_resets,
                      native_reset_draws=audited_resets, native_physics_steps=len(reward_contact_reads),
                      smoke_motion_subset=[files[walk], files[dance]],
                      first_reset_skips_outcomes=True, partial_reset_isolated=True,
                      motion_outcome_attribution_and_ema=True,
                      contact_interface=dict(source_frames_checked=contact_frames,
                          invalid_foot_frames_preserved=invalid_foot_frames,
                          foot_names=state.foot_names, current_future_alignment=True,
                          exhausted_future_frames_masked=True, partial_reset_isolated=True,
                          reward_before_command_advance_and_terminal_reset=True,
                          terminal_old_contact_snapshot_preserved=True,
                          foot_sensor_order_checked=True, crawling_labels_preserved=True))
        report_file.write_text(json.dumps(report, indent=2) + "\n")
        print("PGMT SAMPLING VALIDATED: " + json.dumps(report), flush=True)
    finally:
        env.close()


try:
    main()
except Exception as error:
    report_file.write_text(json.dumps(dict(passed=False, error=repr(error)), indent=2) + "\n")
    traceback.print_exc()
    raise
finally:
    app.close()
