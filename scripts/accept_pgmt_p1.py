"""Run P1 geometry/lifecycle acceptance checks and retain each native report."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default=str(ROOT / "logs/pgmt_p1_acceptance"))
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
checks = [
    ("start_poses", "validate_pgmt_start_poses.py", []),
    ("physical_placement", "inspect_pgmt_placement.py", ["--validate", "--start-frame", "0"]),
    ("map_queries", "validate_pgmt_map_queries.py", []),
    ("contact_sampling", "validate_pgmt_sampling.py", []),
    ("curriculum_lifecycle", "validate_pgmt_curriculum.py", []),
]
report = dict(passed=False, started_at=datetime.now().isoformat(), python=sys.executable,
              device=args.device, checks=[])
report_path = output / "acceptance.json"
for name, script, extra in checks:
    destination = output / name
    command = [sys.executable, str(ROOT / "scripts" / script), "--headless", "--device", args.device,
               "--output", str(destination), *extra]
    print(f"[P1 SUITE] {name} started", flush=True)
    with (output / f"{name}.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    result_file = destination / "validation.json"
    data = json.loads(result_file.read_text()) if result_file.exists() else {}
    item = dict(name=name, command=command, returncode=result.returncode,
                passed=result.returncode == 0 and data.get("passed") is True, report=str(result_file))
    report["checks"].append(item)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[P1 SUITE] {name} passed={item['passed']}", flush=True)
    if not item["passed"]:
        raise SystemExit(f"P1 acceptance failed: {name}; inspect {result_file}")
report.update(passed=True, finished_at=datetime.now().isoformat())
report_path.write_text(json.dumps(report, indent=2) + "\n")
print(f"[P1 SUITE] ALL PASSED: {report_path}", flush=True)
