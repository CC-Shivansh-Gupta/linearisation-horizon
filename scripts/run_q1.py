"""Q1 driver (PREREGISTRATION.md §9-§11). NOT RUN. It needs a CUDA GPU and TD-MPC2 at the pinned commit.

    python -m scripts.run_q1 --tdmpc2 PATH/tdmpc2/tdmpc2 --ckpt-dir DIR --go-ahead "<who, when, which message>" [--tasks ...]

Refuses to start unless ALL of these hold, because the protocol-2 incident (P1) was a run that went ahead of the
reviewer's authorisation, and because two earlier runs recorded no code identity (closeout note, limitation 5):
  1. --go-ahead is given. It is a written record of the reviewer's explicit execution go-ahead (§9 step 3), copied into
     results/RUN.json. It cannot authenticate anything; the person running it is the one who has to have the go-ahead.
  2. The freeze commit is an ancestor of HEAD and the three frozen files equal the freeze commit byte for byte.
  3. No tracked file is modified, so HEAD identifies the code exactly. HEAD is printed and written into RUN.json and into
     every per-model file.

Order (§10): S5 first, on a random-init network with no checkpoint; then, for each of the 15 files, in the order of
config/prereg.yaml: SHA-256 check, load in the pinned code, 10 rollouts, the §5 statistics at every scale for both
linearisations. A file that fails its hash or its load is recorded in results/diagnostics/{task}_s{seed}_load_failure.json
and excluded: no conversion, no substitution (§4, R2). A file that is simply absent is an error, not an exclusion.

The run prints no statistic and computes no verdict. The verdict is a separate, deliberate command
(python -m scripts.lh_verdict), to be run only after the reviewer releases the outputs. At the end the run writes
results/MANIFEST.sha256 and prints its SHA-256, so the outputs can stay sealed and only that digest be shared.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scripts import lh
from scripts import tdmpc2_adapter as A
from scripts.prereg import ROOT, load_cfg

CFG = load_cfg()
FREEZE_SHA = "7c98789a9849e7dce0b2c63c7c90ae56755bd88f"
FROZEN_FILES = ["README.md", "PREREGISTRATION.md", "config/prereg.yaml"]
GRID = CFG["horizons"]["grid"]
EPS_LIST = CFG["perturbation"]["epsilon_reported"]
EPS_PRIMARY = CFG["perturbation"]["epsilon_primary"]
D = CFG["perturbation"]["directions_per_anchor"]
ANCHORS = list(range(CFG["rollouts"]["anchors"]["first"], CFG["rollouts"]["anchors"]["last"] + 1,
                     CFG["rollouts"]["anchors"]["step"]))
TOLERANCES = CFG["decision"]["tolerances_reported"]
TOL_PRIMARY = CFG["decision"]["tolerance_primary"]
H_MAX = max(GRID)


def git(*args):
	return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def provenance():
	"""HEAD, after checking that it identifies the code exactly and that the frozen files are untouched."""
	git("merge-base", "--is-ancestor", FREEZE_SHA, "HEAD")                      # raises if not an ancestor
	assert not git("status", "--porcelain", "--untracked-files=no"), "tracked files are modified: HEAD would not identify the code"
	assert not git("diff", "--name-only", FREEZE_SHA, "--", *FROZEN_FILES), "a frozen file differs from the freeze commit"
	return git("rev-parse", "HEAD")


def model_list():
	hashes = CFG["models"]["checkpoint_sha256"]
	return [(t, s, f"{t}-{s}.pt", hashes[f"{t}-{s}.pt"]) for t in CFG["models"]["tasks"] for s in CFG["models"]["seeds"]]


def _write(path, obj):
	Path(path).write_bytes((json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("utf-8"))   # bytes: no CRLF on Windows


# --- S5 (§10): plumbing on the real code path, random-init network, no checkpoint -----------------------------------

def s5(T, device, out, head):
	"""Prints and stores pass/fail per check, never a value."""
	task = CFG["models"]["tasks"][0]
	cfg, agent = A.random_init_agent(T, task)
	m64 = A.to_float64(agent.model)
	O, Acts, _ = A.rollout(T, cfg, agent, 0)
	O2, Acts2, _ = A.rollout(T, cfg, agent, 0)
	Z = A.encode(m64, O, device)[:-1]
	Y = A.encode_pre_simnorm(m64, O, device)[:-1]
	acts = torch.as_tensor(Acts, device=device)
	f = A.dynamics(m64)
	tau = ANCHORS[0]
	etas = torch.stack([lh.direction(task, 0, 0, tau, j) for j in range(2)]).to(device)
	real = Z[tau:tau + H_MAX]
	args = (f, Y[tau], acts[tau:tau + H_MAX], etas, [EPS_PRIMARY], GRID, real)
	r1, r2 = lh.anchor_errors(*args)[EPS_PRIMARY], lh.anchor_errors(*args)[EPS_PRIMARY]

	def same(a, b):
		return np.array_equal(a, b, equal_nan=True)

	P = lh.tangent_chain(f, Z, acts, etas, [1])[1]                       # JVP at (z_0, a_0), one step
	lead = torch.func.jacrev(lambda z: f(z, acts[0]))(Z[0])              # the explicit Jacobian at the same point
	checks = {
		"shapes": O.shape[0] == 501 and Acts.shape[0] == 500 and tuple(Z.shape) == (500, 512) and tuple(Y.shape) == (500, 512)
		          and r1["nom"]["e"].shape == (2, len(GRID)) and r1["real"]["e"].shape == (2, len(GRID)),
		"dtypes_float64": all(x.dtype == torch.float64 for x in (Z, Y, acts, etas, P)),
		"rollout_deterministic": np.array_equal(O, O2) and np.array_equal(Acts, Acts2),
		"anchor_errors_deterministic": all(same(r1[L][k], r2[L][k]) for L in ("nom", "real") for k in ("e", "angle", "pnorm", "unres")),
		"pre_simnorm_slice_reproduces_encoder": A.check_encoder_consistency(m64, O, device) <= 1e-12,
		"nominal_start_equals_encoded_state": (lh.simnorm(Y[tau]) - real[0]).abs().max().item() <= 1e-12,
		"L_nom_equals_L_real_at_k0": same(r1["nom"]["e"][:, 0], r1["real"]["e"][:, 0])
		                              or np.allclose(r1["nom"]["e"][:, 0], r1["real"]["e"][:, 0], rtol=1e-8, atol=0),
		"jvp_through_real_network_matches_explicit_jacobian":
			(torch.linalg.vector_norm(P - etas @ lead.T) / torch.linalg.vector_norm(etas @ lead.T)).item() <= 1e-10,
		"batched_next_equals_unbatched": (f(Z[:3], acts[:3]) - torch.stack([f(Z[i], acts[i]) for i in range(3)])).abs().max().item() <= 1e-12,
	}
	for k, ok in checks.items():
		print(f"S5 {'PASS' if ok else 'FAIL'}  {k}", flush=True)
	_write(Path(out) / "s5.json", {"head": head, "checks": checks, "values_printed": False})
	assert all(checks.values()), "S5 failed: the confirmatory run does not start"


# --- one model --------------------------------------------------------------------------------------------------------

def run_model(T, task, seed, name, want, ckpt_dir, out, device, head):
	work, diag, pairs = out / "work", out / "diagnostics", out / "pairs"
	target = work / f"{task}_s{seed}.json"
	if target.exists():
		print(f"skip {target.name} (exists)", flush=True)
		return
	ckpt = Path(ckpt_dir) / name
	if not ckpt.exists():
		raise FileNotFoundError(f"{ckpt} is absent: that is an error, not an exclusion")

	def exclude(reason):
		_write(diag / f"{task}_s{seed}_load_failure.json", {"checkpoint": name, "reason": reason, "excluded": True, "head": head})
		print(f"{task} s{seed}: EXCLUDED ({reason[:120]})", flush=True)

	digest = A.sha256(ckpt)
	if digest != want:
		return exclude(f"SHA-256 mismatch: got {digest}, pinned {want}")
	try:
		cfg, agent = A.load_agent(T, task, ckpt)
	except Exception as e:
		return exclude(f"{type(e).__name__}: {e}")

	t0 = time.time()
	trajs, returns = [], []
	for e in CFG["rollouts"]["env_seeds"]:
		O, Acts, R = A.rollout(T, cfg, agent, e)
		trajs.append((O, Acts))
		returns.append(float(R.sum()))
	m64 = A.to_float64(agent.model)
	f = A.dynamics(m64)
	acc = lh.Accumulator(EPS_LIST, GRID)
	one_step, b_ratio = [], []
	s_primary = lh.target_size(EPS_PRIMARY)
	for e, (O, Acts) in enumerate(trajs):
		Z = A.encode(m64, O, device)                      # (501, 512)
		Y = A.encode_pre_simnorm(m64, O, device)[:-1]
		acts = torch.as_tensor(Acts, device=device)
		assert Z.shape[0] == acts.shape[0] + 1 and ANCHORS[-1] + H_MAX <= acts.shape[0]
		one_step.append(A.one_step_errors(m64, Z[:-1], acts, Z[1:]))
		bnorm = A.action_jacobian_norms(m64, Z[:-1][ANCHORS], acts[ANCHORS])
		b_ratio.extend((s_primary / (A.PLANNER_MIN_STD * bnorm)).tolist())
		for tau in ANCHORS:
			etas = torch.stack([lh.direction(task, seed, e, tau, j) for j in range(D)]).to(device)
			acc.add(lh.anchor_errors(f, Y[tau], acts[tau:tau + H_MAX], etas, EPS_LIST, GRID, Z[tau:tau + H_MAX]))
	stack = acc.stack()
	m1 = float(np.median(np.concatenate(one_step)))
	stats = lh.summarise(stack, GRID, TOLERANCES, TOL_PRIMARY)
	np.savez_compressed(pairs / f"{task}_s{seed}.npz", **{f"{lh.key(eps)}_{L}_{k}": v
	                    for eps, by_l in stack.items() for L, d in by_l.items() for k, v in d.items()})
	seconds = time.time() - t0
	_write(diag / f"{task}_s{seed}.json", {
		"checkpoint_sha256": digest, "episode_returns": returns, "mean_return": float(np.mean(returns)),
		"published_final_return": A.published_final_return(T, task, seed), "m1": m1,
		"s_over_m1": s_primary / m1, "median_s_over_0p05_Bnorm": float(np.median(b_ratio)),
		"seconds": seconds, "head": head})
	_write(target, {"task": task, "seed": seed, "checkpoint_sha256": digest, "head": head, "stats": stats})   # last: its existence means complete
	print(f"{task} s{seed}: done, mean return {np.mean(returns):.1f}, {seconds:.0f}s", flush=True)
	del agent, m64
	torch.cuda.empty_cache()


def write_manifest(out):
	"""sha256sum -b format, LC_ALL=C order, then the digest of the manifest itself (the sealed-output receipt)."""
	files = sorted((p for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"),
	               key=lambda p: p.relative_to(out).as_posix().encode())
	lines = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()} *./{p.relative_to(out).as_posix()}\n" for p in files)
	(out / "MANIFEST.sha256").write_bytes(lines.encode("utf-8"))
	return hashlib.sha256(lines.encode("utf-8")).hexdigest(), len(files)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--tdmpc2", required=True, help="path to the tdmpc2/tdmpc2 package directory")
	ap.add_argument("--ckpt-dir", required=True)
	ap.add_argument("--out", default=str(ROOT / "results"))
	ap.add_argument("--go-ahead", required=True, help="the reviewer's explicit execution go-ahead: who, when, which message")
	ap.add_argument("--tasks", nargs="*")
	args = ap.parse_args()
	assert args.go_ahead.strip(), "an empty go-ahead record"
	assert torch.cuda.is_available(), "TD-MPC2 hard-codes cuda:0"
	head = provenance()
	print(f"CODE IDENTITY: repository HEAD {head}  (freeze {FREEZE_SHA}, frozen files byte-identical)", flush=True)
	out = Path(args.out)
	for d in ("work", "diagnostics", "pairs"):
		(out / d).mkdir(parents=True, exist_ok=True)
	run_json = out / "RUN.json"
	if run_json.exists():
		assert json.loads(run_json.read_text())["head"] == head, "results/ was started at a different HEAD; use a new --out"
	else:
		_write(run_json, {"head": head, "freeze": FREEZE_SHA, "go_ahead": args.go_ahead,
		                  "started_unix": time.time(), "tasks": args.tasks or CFG["models"]["tasks"]})
	A.write_environment(out)
	device = torch.device("cuda:0")
	T = A.import_tdmpc2(args.tdmpc2)
	s5(T, device, out, head)
	for task, seed, name, want in model_list():
		if args.tasks and task not in args.tasks:
			continue
		run_model(T, task, seed, name, want, args.ckpt_dir, out, device, head)
	digest, n = write_manifest(out)
	print(f"MANIFEST: {n} files, SHA-256 of results/MANIFEST.sha256 = {digest}\nOutputs stay sealed. No verdict was computed.", flush=True)


if __name__ == "__main__":
	main()
