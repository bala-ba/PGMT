"""PGMT P2/P3 checkpoint semantics around the native grouped PPO rollout loop."""
from copy import deepcopy
from pathlib import Path
import torch

from motion_tracking_rl.registry import register_runner
from motion_tracking_rl.runners.on_policy_runner import OnPolicyRunner
from motion_tracking_rl.utils.pgmt_checkpoint import canonical, migrate_stage1, migrate_three_head, restore_training, training_payload
from motion_tracking_rl.utils.pgmt_distributed import cpu_state, select_rank_state
from motion_tracking_rl.networks.layers.normalization import EmpiricalNormalization


@register_runner("PGMTStage2Runner")
class PGMTStage2Runner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        cfg = deepcopy(train_cfg)
        runtime_contract = cfg.pop("pgmt_training_contract")
        if not isinstance(runtime_contract, dict) or not runtime_contract:
            raise ValueError("Bind pgmt_training_contract to the actual environment before constructing the runner")
        self.training_contract = canonical(dict(runtime=runtime_contract,
            policy=cfg["policy"], algorithm=cfg["algorithm"], obs_groups=cfg["obs_groups"],
            num_steps_per_env=cfg["num_steps_per_env"]))
        self.completed_iterations = 0
        self.initialization = None
        super().__init__(env, cfg, log_dir, device)
        if self.is_distributed:
            for module in self.alg.policy.modules():
                if isinstance(module, EmpiricalNormalization):
                    module.sync_distributed = True
        if self.alg.policy.get_checkpoint_metadata()["policy_family"] not in ("pgmt_perception_only", "pgmt_terrain_tracking"):
            raise ValueError("PGMTStage2Runner requires the perception policy")
        environment_groups = getattr(getattr(getattr(env, "unwrapped", None), "cfg", None), "pgmt_reward_groups", None)
        if environment_groups is not None and tuple(environment_groups) != tuple(self.alg.policy.value_groups):
            raise ValueError("PGMT environment reward order and critic value_groups differ")
        original_update = self.alg.update
        def update():
            result = original_update()
            self.completed_iterations += 1
            if self.is_distributed:
                names = sorted(result)
                values = torch.tensor([result[name] for name in names], dtype=torch.float64, device=self.device)
                torch.distributed.all_reduce(values)
                result = dict(zip(names, (values / self.gpu_world_size).tolist()))
            return result
        self.alg.update = update

    def initialize_stage1(self, path):
        if self.completed_iterations or self.alg.optimizer.state or self.initialization is not None:
            raise ValueError("Stage-1 initialization requires a fresh runner")
        self.initialization = migrate_stage1(self.alg.policy, path)
        return deepcopy(self.initialization)

    def initialize_three_head(self, path):
        if self.completed_iterations or self.alg.optimizer.state or self.initialization is not None:
            raise ValueError("Three-head migration requires a fresh runner")
        self.initialization = migrate_three_head(self.alg.policy, path)
        return deepcopy(self.initialization)

    def initialize_policy(self, path):
        """Explicit weights-only start after a sampling/task-distribution change."""
        import hashlib
        from motion_tracking_rl.utils.pgmt_checkpoint import validate_tensors
        if self.completed_iterations or self.alg.optimizer.state or self.initialization is not None:
            raise ValueError("Policy initialization requires a fresh runner")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("policy_metadata") != self.alg.policy.get_checkpoint_metadata():
            raise ValueError("Weights-only initialization requires matching policy/observation metadata")
        validate_tensors(saved["model_state_dict"], self.alg.policy.state_dict())
        self.alg.policy.load_state_dict(saved["model_state_dict"], strict=True)
        self.initialization = dict(kind="policy_weights_only", source=str(Path(path).resolve()),
            sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            copied=sorted(saved["model_state_dict"]), initialized=[])
        return deepcopy(self.initialization)

    def save(self, path, infos=None):
        payload = training_payload(self, infos)
        if self.is_distributed:
            # Only small rank-local state is gathered. Policy/Adam are shared;
            # curriculum, sampler, RNG and device/seed contracts are rank-local.
            local = cpu_state({key: payload[key] for key in
                               ('training_contract', 'environment_state_dict', 'rng_state')})
            local['rank'] = self.gpu_global_rank
            states = [None] * self.gpu_world_size if self.gpu_global_rank == 0 else None
            torch.distributed.gather_object(local, states, dst=0)
            if self.gpu_global_rank != 0:
                return
            payload['distributed'] = dict(version=1, world_size=self.gpu_world_size, rank_states=states)
        # Execution switches do not alter the task/optimizer contract, but must
        # remain discoverable when comparing runs or resuming in another runtime.
        runtime = getattr(getattr(self.env, 'unwrapped', None), '_pgmt_performance', None)
        execution = deepcopy(runtime) if runtime is not None else {}
        execution['actor_critic_forward'] = (
            'joint_encode' if callable(getattr(self.alg.policy, 'act_and_evaluate', None))
            and not self.alg.policy.is_recurrent else 'separate_encode')
        execution['ppo_log_scalars'] = getattr(self.alg, 'log_scalar_mode', 'per_minibatch')
        payload['execution_options'] = execution
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def _save_training_checkpoint(self, path):
        # The native loop invokes this on all ranks, making save's gather safe.
        self.save(path)

    def load(self, path, load_optimizer=True, map_location=None, load_environment=None):
        if not load_optimizer or load_environment is False:
            raise ValueError("PGMT strict resume requires optimizer and environment; use explicit initialization for migration")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        saved = select_rank_state(saved, self.gpu_world_size, self.gpu_global_rank)
        return restore_training(self, saved)

    def learn(self, *args, **kwargs):
        if self.initialization is None:
            raise ValueError("Select explicit initialization or strict load before training")
        result = super().learn(*args, **kwargs)
        # Native loop labels its final iteration with the last zero-based index.
        # P2 checkpoints and subsequent calls always resume at the NEXT index.
        self.current_learning_iteration = self.completed_iterations
        return result
