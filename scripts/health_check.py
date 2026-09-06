from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print(f"> {' '.join(command)}")
    completed = subprocess.run(command, cwd=ROOT, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    run([sys.executable, "scripts/smoke_test.py"])
    run([sys.executable, "scripts/run_random_agent.py", "--episodes", "2"])
    run([sys.executable, "scripts/train_q_agent.py", "--episodes", "60", "--eval-episodes", "15"])
    run([sys.executable, "scripts/smart_scan_demo.py"])
    print("health check passed")


if __name__ == "__main__":
    main()
