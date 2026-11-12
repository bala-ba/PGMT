#!/usr/bin/env python3
"""Stop this user's live PMT viewers from any terminal; leave VNC/other jobs alone."""
import argparse
import os
from pathlib import Path
import signal
import time


TARGET = Path(__file__).with_name("mujoco_live_student.py").resolve()


def identity(pid):
    """Use start time as well as PID so escalation cannot kill a reused PID."""
    proc = Path("/proc") / str(pid)
    try:
        if proc.stat().st_uid != os.getuid():
            return None
        args = proc.joinpath("cmdline").read_bytes().split(b"\0")
        if not args or not Path(os.fsdecode(args[0])).name.startswith("python"):
            return None
        cwd = proc.joinpath("cwd").resolve()
        if not any(arg and (cwd / os.fsdecode(arg)).resolve() == TARGET for arg in args[1:]):
            return None
        # stat comm can contain spaces/parentheses; remaining tokens start at field 3.
        fields = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, ValueError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, help="Stop only this PID, after checking it is our live viewer")
    args = parser.parse_args()
    pids = [args.pid] if args.pid is not None else [int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit()]
    targets = {pid: token for pid in pids if (token := identity(pid)) is not None}
    if not targets:
        print("No matching PMT live simulation is running.")
        return
    for signum, timeout in ((signal.SIGINT, 3.), (signal.SIGTERM, 2.), (signal.SIGKILL, 1.)):
        for pid, token in list(targets.items()):
            if identity(pid) != token:
                del targets[pid]
                continue
            print(f"Stopping PMT live PID {pid}: {signum.name}", flush=True)
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                targets.pop(pid, None)
        deadline = time.monotonic() + timeout
        while targets and time.monotonic() < deadline:
            targets = {pid: token for pid, token in targets.items() if identity(pid) == token}
            if targets:
                time.sleep(.1)
        if not targets:
            print("PMT live simulation stopped.", flush=True)
            return
    raise SystemExit(f"Unable to stop PIDs: {list(targets)}")


if __name__ == "__main__":
    main()
