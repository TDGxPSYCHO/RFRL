# Smart Scan RF Spectrum RL

Starter simulation layer for **Smart Scan Technology**, an electronic-support style RF sensing system for detecting and characterizing spectrum activity in simulation.

Scope boundary: this project models detection, channel selection, signal presence, rough signal category, and encrypted-signal detect-only behavior. It does not implement jamming, decryption, content interception, or offensive EW effects.

This first milestone is intentionally small and runnable:

- A synthetic RF scene generator with frequency-hopping, chirp/LPI-like, narrowband, and encrypted-but-undecoded emitters.
- A Gymnasium-style environment for sensing one channel at a time.
- A Layer 3 chunk classifier that receives only the spectrum portion selected by Layer 1.
- A closed-loop pipeline where classifier feedback updates Layer 1 scan scoring.
- A random baseline CLI to validate the loop.
- A deterministic Smart Scan baseline that follows strongest observed spectrum energy.
- A lightweight Q-learning trainer that runs before heavier RL dependencies are installed.
- Tests that run without heavyweight SDR/RL dependencies.

The code is designed so real `RFRL Gym` and `TorchSig` adapters can be added later without rewriting the agent-facing environment.

## Quick Start

```powershell
cd C:\Users\loq\rf-spectrum-rl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
python scripts\run_random_agent.py --episodes 3
pytest
```

If PowerShell blocks venv activation, run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\python.exe scripts\run_random_agent.py --episodes 3
.\.venv\Scripts\python.exe -m pytest
```

Without installing `pytest`, you can still validate the core loop:

```powershell
$env:PYTHONPATH="C:\Users\loq\rf-spectrum-rl\src"
python scripts\smoke_test.py
python scripts\run_random_agent.py --episodes 3
python scripts\train_q_agent.py --episodes 500 --eval-episodes 50
python scripts\smart_scan_demo.py
python scripts\health_check.py
```

For an SIH demo, run:

```powershell
$env:PYTHONPATH="C:\Users\loq\rf-spectrum-rl\src"
python scripts\health_check.py
```

## Project Shape

```text
src/rf_spectrum_rl/
  env.py          Gymnasium-style RF sensing environment
  agents.py       Tabular Q-learning baseline and evaluator
  classifier.py   Layer 3 classifier interface and baseline classifier
  pipeline.py     Layer 1 plus Layer 3 closed-loop coordinator
  metrics.py      Detection, false-alarm, and scan-efficiency metrics
  scene.py        Synthetic RF scenario generation
  signals.py      Signal models: hopping, chirp, narrowband, encrypted marker
  spaces.py       Tiny fallback spaces if Gymnasium is not installed
scripts/
  run_random_agent.py
  train_q_agent.py
  smart_scan_demo.py
  health_check.py
tests/
```

## Near-Term Roadmap

## System Flow

```text
RFRL-Gym-like simulator / future TorchSig adapter
  -> I/Q-like spectrogram + hidden ground truth
  -> Layer 1 scan agent chooses one channel
  -> environment.step(channel)
  -> Layer 3 classifier sees only that observed chunk
  -> classifier result feeds back into Layer 1 scoring
  -> evaluator compares decisions against simulator ground truth
  -> report detection speed, missed detections, false alarms, and round-robin baseline
```

1. Add a TorchSig-backed dataset adapter for richer spectrogram generation.
2. Add a Stable-Baselines3 PPO/DQN baseline using the same environment.
3. Add an RFRL Gym wrapper or scenario bridge once dependency versions are pinned.
4. Add a wideband multi-signal scenario and detection metrics.
5. Add optional directional/ray-tracing data as an advanced extension.
