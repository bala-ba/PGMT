"""Read-only rollout metrics for PGMT TensorBoard and JSONL training logs."""
import torch


class RewardWindow:
    """Integrate all reward samples, retaining their PRE-step birth tile.

    Values passed to add are already weighted per-control-step contributions.
    No second weight or dt is applied here. Empty cells are not observations.
    """
    def __init__(self, terms, groups, families, levels, *, device):
        self.terms, self.groups, self.families, self.levels = terms, groups, families, levels
        self.cells = len(families) * levels
        self.sums = torch.zeros(self.cells, len(terms), device=device, dtype=torch.float64)
        self.samples = torch.zeros(self.cells, device=device, dtype=torch.long)
        self.steps = 0

    def add(self, contributions, birth_family, birth_level):
        cells = birth_family * self.levels + birth_level
        self.sums.index_add_(0, cells, contributions.to(torch.float64))
        self.samples += torch.bincount(cells, minlength=self.cells)
        self.steps += 1

    def consume(self, dt):
        sums, counts = self.sums.cpu(), self.samples.cpu()
        total = int(counts.sum())
        if not total:
            return {}
        mean = sums.sum(0) / total
        result = {f"RewardStep/{name}": float(mean[i]) for i, name in enumerate(self.terms)}
        result.update({f"RewardRate/{name}": float(mean[i])/dt for i, name in enumerate(self.terms)})
        result["RewardStep/total"] = float(mean.sum())
        result["Window/transitions"] = total
        for group in self.groups:
            indices = [i for i, name in enumerate(self.terms) if name.startswith(group + "_")]
            result[f"RewardGroupStep/{group}"] = float(mean[indices].sum())
            for cell in range(self.cells):
                if counts[cell]:
                    family, level = divmod(cell, self.levels)
                    result[f"TerrainReward/{self.families[family]}/L{level}/{group}"] = float(
                        sums[cell, indices].sum() / counts[cell])
        self.sums.zero_(); self.samples.zero_(); self.steps = 0
        return result


class PGMTTrainingMetrics:
    def __init__(self, env):
        self.env = env
        self.course = env._pgmt_terrain_curriculum
        families = {t["family_index"]: t["family"] for t in env.scene.terrain.manifest["tiles"]}
        self.families = [families[i] for i in range(len(families))]
        self.window = RewardWindow(env.reward_manager.active_terms, env.cfg.pgmt_reward_groups,
                                   self.families, self.course.num_levels, device=env.device)
        self.term_names = env.termination_manager.active_terms
        self.terminations = torch.zeros(len(self.term_names), dtype=torch.long, device=env.device)
        self.previous_counts = self.course.counts.clone()

    def before_step(self):
        self.birth_family = self.course.birth_types.clone()
        self.birth_level = self.course.birth_levels.clone()

    def after_step(self):
        self.window.add(self.env.reward_manager._step_reward * self.env.step_dt,
                        self.birth_family, self.birth_level)
        for i, name in enumerate(self.term_names):
            self.terminations[i] += self.env.termination_manager.get_term(name).sum()

    @torch.inference_mode()
    def log(self, writer, iteration):
        from pmt_tasks.pgmt_curriculum import OUTCOMES
        from pmt_tasks.pgmt_runtime import get_pgmt_state
        from pmt_tasks.pgmt_math import angle, inverse_rotate, yaw_quat

        values = self.window.consume(self.env.step_dt)
        transitions = values.get("Window/transitions", 1)
        for name, count in zip(self.term_names, self.terminations.cpu().tolist()):
            values[f"TerminationCount/{name}"] = count
            values[f"TerminationFraction/{name}"] = count / transitions
        self.terminations.zero_()
        counts = self.course.counts.cpu()
        delta = (self.course.counts - self.previous_counts).cpu()
        self.previous_counts.copy_(self.course.counts)
        levels, families = self.course.terrain.terrain_levels, self.course.terrain.terrain_types
        for family, name in enumerate(self.families):
            selected = families == family
            values[f"TerrainMix/{name}/actual_fraction"] = float(selected.float().mean())
            proportions = getattr(self.course.terrain, "family_proportions", None)
            if proportions is not None:
                values[f"TerrainMix/{name}/target_fraction"] = proportions[name]
            if selected.any():
                values[f"Curriculum/{name}/mean_level"] = float(levels[selected].float().mean())
            for level in range(self.course.num_levels):
                cell = counts[family, level]
                prefix = f"Curriculum/{name}/L{level}"
                values[f"{prefix}/env_count"] = int((selected & (levels == level)).sum())
                values[f"{prefix}/outcomes"] = int(cell.sum())
                if cell.sum():
                    values[f"{prefix}/success_fraction_of_outcomes"] = float(cell[0] / cell.sum())
                for i, outcome in enumerate(OUTCOMES):
                    values[f"{prefix}/{outcome}"] = int(cell[i])
                    values[f"OutcomeWindow/{name}/L{level}/{outcome}"] = int(delta[family, level, i])
        command = self.env.command_manager.get_term("motion")
        probability = command.sampler.get_sampling_probabilities()
        values.update({"Sampler/entropy": float(-(probability * probability.clamp_min(1e-20).log()).sum()),
                       "Sampler/effective_motion_count": float(1 / probability.square().sum()),
                       "Sampler/max_probability": float(probability.max()),
                       "Sampler/min_probability": float(probability.min())})
        writer.add_histogram("Sampler/motion_probabilities", probability.cpu(), iteration)
        # These are post-update state snapshots, not terminal reward inputs.
        state = get_pgmt_state(self.env)
        physical, ref = state.robot_state(), state.reference()
        actual_q = yaw_quat(physical["body_quat"][:, state.anchor])[:, None].expand_as(physical["body_quat"])
        target_q = yaw_quat(ref["body_quat"][:, state.anchor])[:, None].expand_as(ref["body_quat"])
        actual = inverse_rotate(actual_q, physical["body_pos"] - physical["body_pos"][:, state.anchor, None])
        target = inverse_rotate(target_q, ref["body_pos"] - ref["body_pos"][:, state.anchor, None])
        for group in ("upper", "lower"):
            body_ids, joint_ids = state.groups[group], state.groups[group + "_joints"]
            values[f"TrackingSnapshot/{group}_body_position_rmse_m"] = float(
                (actual[:, body_ids]-target[:, body_ids]).square().sum(-1).mean().sqrt())
            values[f"TrackingSnapshot/{group}_body_orientation_rad"] = float(
                angle(physical["body_quat"][:, body_ids], ref["body_quat"][:, body_ids]).mean())
            values[f"TrackingSnapshot/{group}_joint_rmse_rad"] = float(
                (physical["joint_pos"][:, joint_ids]-ref["joint_pos"][:, joint_ids]).square().mean().sqrt())
        for name, value in values.items():
            writer.add_scalar(name, value, iteration)
        writer.flush()
        return values
