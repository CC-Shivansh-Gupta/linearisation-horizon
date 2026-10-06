"""Loads config/prereg.yaml, the only source of thresholds (PREREGISTRATION.md §6, §12)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_cfg(path=ROOT / "config" / "prereg.yaml"):
	with open(path, encoding="utf-8") as f:
		return yaml.safe_load(f)
