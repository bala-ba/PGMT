"""P2.3 part 4 CPU/CUDA acceptance, including existing perception regressions.

Run with the pmt Python environment from PMT. Requires visible CUDA so a report
cannot silently pass with GPU cases skipped. Does not launch Isaac or load a
training checkpoint. Every invocation preserves its log, JUnit and JSON report.
"""
import contextlib
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
FILES = ["tests/test_pgmt_perception_ppo.py", "tests/test_pgmt_ifm.py",
         "tests/test_pgmt_terrain_glimpse.py", "tests/test_pgmt_perception.py",
         "tests/test_pgmt_rl.py", "tests/test_normalization_resume.py"]


def main():
    if Path.cwd() != ROOT:
        raise RuntimeError(f"Run from {ROOT}")
    if not torch.cuda.is_available():
        raise RuntimeError("P2.3 acceptance requires visible CUDA; no skipped GPU acceptance")
    sys.path.insert(0, str(ROOT))
    stamp = datetime.now().astimezone()
    directory = ROOT / "logs/pgmt_p23_acceptance" / ("ifm_step4_" + stamp.strftime("%Y%m%d_%H%M%S"))
    directory.mkdir(parents=True)
    args = ["-q", "-o", "junit_family=legacy", "--junitxml=" + str(directory / "pytest.xml"), *FILES]
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        code = int(pytest.main(args))
    (directory / "pytest.log").write_text(output.getvalue())
    cases = list(ET.parse(directory / "pytest.xml").getroot().iter("testcase"))
    ppo_cases = [case for case in cases if case.attrib.get("classname", "").endswith("test_pgmt_perception_ppo")]
    measurements = {case.attrib["name"]: {prop.attrib["name"]: float(prop.attrib["value"])
                    for prop in case.findall("./properties/property")} for case in ppo_cases}
    sources = ["motion_tracking_rl/algorithms/ppo.py", "motion_tracking_rl/storage/rollout_storage.py",
               "motion_tracking_rl/networks/pgmt_actor_critic.py",
               "motion_tracking_rl/networks/pgmt_terrain_glimpse.py",
               "motion_tracking_rl/networks/pgmt_terrain_input.py",
               "motion_tracking_rl/networks/layers/normalization.py",
               "motion_tracking_rl/pgmt_perception_contract.py", "pmt_tasks/pgmt_perception.py",
               "scripts/validate_pgmt_perception_ppo.py", *FILES]
    report = dict(
        status="passed" if code == 0 and not any(case.find("skipped") is not None for case in cases) else "failed",
        scope="P2.3 part 4: native PPO observation storage, recomputation and real gradients",
        timestamp=stamp.isoformat(), python=sys.version, torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(0), cuda_version=torch.version.cuda,
        pytest_arguments=args, exit_code=code, tests=len(cases),
        failures=sum(case.find("failure") is not None for case in cases),
        errors=sum(case.find("error") is not None for case in cases),
        skipped=sum(case.find("skipped") is not None for case in cases),
        ppo_cases=len(ppo_cases), ppo_cuda_cases=sum("cuda" in case.attrib["name"] for case in ppo_cases),
        optimizer_steps=sum(values.get("optimizer_steps", 0) for values in measurements.values()),
        measurements=measurements,
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        configuration=dict(num_envs=6, rollout_steps=4, epochs=2, minibatches=3, learning_rate=3e-4,
                           active_rollout_update_cycles=2, actor_critic_hidden_dims=[32, 16],
                           embedding_dim=128, terrain_glimpse="default", value_groups=["upper", "lower", "auxiliary"]),
        limitations=["Synthetic sensor hits through the production P2.1 preprocessing; no Isaac run",
                     "No model_15000.pt migration, strict checkpoint resume or formal task config changes",
                     "Full-policy real-map and performance acceptance remains P2.3 part 5",
                     "Zero terrain parameter change in fallback is tested with fresh Adam, without past momentum",
                     "Native act() holds the observation reference until process_env_step() copies it; caller must preserve it"])
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(output.getvalue())
    print(json.dumps({key: report[key] for key in ("status", "tests", "skipped", "ppo_cases", "ppo_cuda_cases", "optimizer_steps")}))
    print(f"Report: {directory / 'report.json'}")
    return code or int(report["status"] != "passed")


if __name__ == "__main__":
    sys.exit(main())
