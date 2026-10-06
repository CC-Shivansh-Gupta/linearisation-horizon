"""The only file that touches TD-MPC2 (PREREGISTRATION.md §3.1). It supplies scripts.lh with a float64 map f(z, a), the
encoded real states, the encoder's pre-SimNorm output and the logged actions.

NOT RUN. Written after the freeze and before the reviewer's go-ahead (§9 step 3). It needs a CUDA GPU (TD-MPC2 hard-codes
cuda:0) and the TD-MPC2 code at the pinned commit e9f5932. Nothing here loads a checkpoint at import time.

The rollout, float64-copy and re-encoding conventions are those of protocols 1-2 (hankel-hierarchy
scripts/collect_anchors.py: C7 eval_mode=True, C8 float64 copy with z_t re-encoded, C11 dm-control 1.0.24 / mujoco 3.2.4).
They are re-implemented here, not imported, because this repository is separate from the completed studies.
"""
import hashlib
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from scripts import lh

# Pinned-architecture constants, from hankel-hierarchy/config/prereg.yaml models.architecture_expected. They are checked
# against the loaded model, never used to set a threshold.
LATENT_DIM, SIMNORM_GROUP, NUM_Q, NUM_BINS, VMIN, VMAX, EPISODE_LENGTH = 512, 8, 5, 101, -10, 10, 500
MODEL_SIZE = 5
PLANNER_MIN_STD = 0.05   # tdmpc2/config.yaml at e9f5932


def import_tdmpc2(path):
	"""path: the tdmpc2/tdmpc2 package directory of a clone at the pinned commit."""
	sys.path.insert(0, str(Path(path).resolve()))
	import hydra.utils
	hydra.utils.get_original_cwd = os.getcwd          # parse_cfg expects a hydra run
	from omegaconf import OmegaConf
	from common.parser import parse_cfg
	from common.seed import set_seed
	from common.world_model import WorldModel
	import envs.dmcontrol  # noqa: F401  (envs/__init__ swallows this import's errors; surface them here)
	from envs import make_env
	from tdmpc2 import TDMPC2
	return dict(OmegaConf=OmegaConf, parse_cfg=parse_cfg, set_seed=set_seed, make_env=make_env,
	            TDMPC2=TDMPC2, WorldModel=WorldModel, root=Path(path).resolve())


def make_cfg(T, task, checkpoint, seed=1):
	cfg = T["OmegaConf"].load(T["root"] / "config.yaml")
	cfg.task, cfg.obs, cfg.model_size = task, "state", MODEL_SIZE
	cfg.checkpoint, cfg.seed, cfg.compile = str(checkpoint), seed, False
	cfg.enable_wandb, cfg.save_video = False, False
	return T["parse_cfg"](cfg)


def check_architecture(cfg, model):
	assert cfg.latent_dim == LATENT_DIM and cfg.simnorm_dim == SIMNORM_GROUP
	assert cfg.num_q == NUM_Q and cfg.num_bins == NUM_BINS and [cfg.vmin, cfg.vmax] == [VMIN, VMAX]
	assert cfg.episode_length == EPISODE_LENGTH
	assert repr(model._dynamics[-1].act) == f"SimNorm(dim={SIMNORM_GROUP})"
	assert repr(model._encoder["state"][-1].act) == f"SimNorm(dim={SIMNORM_GROUP})"


def sha256(path):
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


def load_agent(T, task, ckpt):
	"""Builds the agent and loads one checkpoint in the pinned code. Any exception propagates: the caller records it as a
	load failure and excludes the file (§4). No conversion and no key renaming is attempted here."""
	cfg = make_cfg(T, task, ckpt)
	T["make_env"](cfg)
	agent = T["TDMPC2"](cfg)
	agent.load(str(ckpt))
	check_architecture(cfg, agent.model)
	return cfg, agent


def random_init_agent(T, task):
	"""S5 only (§10): the WorldModel constructor under torch seed 999. No checkpoint."""
	cfg = make_cfg(T, task, "none")
	T["make_env"](cfg)
	agent = T["TDMPC2"](cfg)
	T["set_seed"](999)
	agent.model = T["WorldModel"](cfg).to(agent.device).eval()
	check_architecture(cfg, agent.model)
	return cfg, agent


def rollout(T, cfg, agent, env_seed):
	"""One episode in the released evaluation mode (protocol 1, C7). Returns obs (501 x d), actions (500 x m), rewards."""
	cfg.seed = env_seed
	env = T["make_env"](cfg)
	T["set_seed"](env_seed)                           # torch seed = env seed
	obs, done, t = env.reset(), False, 0
	O, A, R = [obs.numpy().copy()], [], []
	while not done:
		a = agent.act(obs, t0=t == 0, eval_mode=True)
		obs, r, done, _ = env.step(a)
		O.append(obs.numpy().copy())
		A.append(a.numpy().copy())
		R.append(float(r))
		t += 1
	return np.stack(O).astype(np.float64), np.stack(A).astype(np.float64), np.array(R)


def to_float64(model):
	"""A float64 copy in evaluation mode, with gradients off (protocol 1, C8)."""
	m = deepcopy(model).double().eval()
	for name, p in m.named_parameters():
		assert p.dtype == torch.float64, name
		p.requires_grad_(False)
	return m


def dynamics(model64):
	"""f(z, a) of scripts.lh: WorldModel.next in float64. Accepts (512,) with (A,), and (..., 512) with (..., A)."""
	def f(z, a):
		return model64.next(z, a, None)
	return f


def encode(model64, obs, device):
	"""z = enc(o) in float64 for obs (T, d). Returns (T, 512)."""
	with torch.no_grad():
		return model64.encode(torch.as_tensor(obs, device=device), None)


def encode_pre_simnorm(model64, obs, device):
	"""y with SimNorm(y) = enc(o): the encoder's last NormedLinear stops before its activation (§3.4).
	Returns (T, 512)."""
	enc = model64._encoder["state"]
	with torch.no_grad():
		x = torch.as_tensor(obs, device=device)
		for layer in enc[:-1]:
			x = layer(x)
		last = enc[-1]
		return last.ln(torch.nn.functional.linear(x, last.weight, last.bias))


def action_jacobian_norms(model64, Z, Acts):
	"""||df/da||_2 (spectral norm) at (z_t, a_t) for each row. The map's output lies in the SimNorm tangent space, so this
	equals the spectral norm of U^T B of protocols 1-2 (§5)."""
	out = []
	jb = torch.func.jacrev(lambda a, z: model64.next(z, a, None))
	for z, a in zip(Z, Acts):
		out.append(torch.linalg.matrix_norm(jb(a, z), ord=2).item())
	return np.array(out)


def one_step_errors(model64, Z, Acts, Z_next):
	"""||f(z_t, a_t) - enc(o_{t+1})||_2 for every step t. Descriptive only (m_1, §3.4)."""
	with torch.no_grad():
		return torch.linalg.vector_norm(model64.next(Z, Acts, None) - Z_next, dim=-1).cpu().numpy()


def published_final_return(T, task, seed):
	"""TD-MPC2's published final return for (task, seed) from results/tdmpc2/{task}.csv at the pinned commit, or None."""
	import csv
	try:
		with open(T["root"].parent / "results" / "tdmpc2" / f"{task}.csv") as f:
			rows = [r for r in csv.DictReader(f) if int(r["seed"]) == seed]
		return float(max(rows, key=lambda r: float(r["step"]))["reward"])
	except Exception:
		return None


def check_encoder_consistency(model64, obs, device, tol=1.0e-12):
	"""SimNorm(encode_pre_simnorm(o)) must equal encode(o). Returns the worst absolute difference."""
	z = encode(model64, obs, device)
	d = (lh.simnorm(encode_pre_simnorm(model64, obs, device)) - z).abs().max().item()
	assert d <= tol, f"pre-SimNorm slice disagrees with the encoder: {d}"
	return d


def write_environment(out):
	import platform
	import subprocess
	lines = [f"python {platform.python_version()}", f"platform {platform.platform()}",
	         f"torch {torch.__version__}  cuda {torch.version.cuda}",
	         f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
	         f"numpy {np.__version__}"]
	for pkg in ("scipy", "mujoco", "dm_control", "tensordict", "gymnasium", "omegaconf", "hydra"):
		try:
			mod = __import__(pkg)
			lines.append(f"{pkg} {getattr(mod, '__version__', '?')}")
		except Exception as e:
			lines.append(f"{pkg} unavailable ({e})")
	Path(out).mkdir(parents=True, exist_ok=True)
	(Path(out) / "ENVIRONMENT.txt").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))   # bytes: no CRLF on Windows
