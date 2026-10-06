"""Q1 §10 synthetic validation, cases S1-S4, plus the §2 and §6 decision logic.
Synthetic maps only; no checkpoint is loaded and no model is involved. S5 is NOT here: it runs only after authorisation.

Run with `python -m pytest tests` or `python tests/test_lh_synthetic.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from scripts import lh
from scripts import lh_verdict as V
from scripts.prereg import load_cfg

CFG = load_cfg()
GRID = CFG["horizons"]["grid"]
DT = torch.float64
N, A = 512, 6


def make_anchor(seed, scale=1.5, n_dir=8, h_max=128):
	g = torch.Generator().manual_seed(seed)
	y = scale * torch.randn(N, generator=g, dtype=DT)
	acts = 0.5 * torch.randn(h_max, A, generator=g, dtype=DT)
	etas = torch.stack([lh.direction("synthetic", seed, 0, 128, j) for j in range(n_dir)])
	return y, acts, etas


def orthogonal(seed):
	g = torch.Generator().manual_seed(seed)
	q, _ = torch.linalg.qr(torch.randn(N, N, generator=g, dtype=DT))
	return q


def linear_map(seed):
	Q = orthogonal(seed)
	Bm = 0.01 * torch.randn(N, A, generator=torch.Generator().manual_seed(seed + 1), dtype=DT)
	return lambda z, a: z @ Q.T + a @ Bm.T


def nonlinear_map(seed, gain):
	"""z' = SimNorm(gain * W2 tanh(W1 z + U a)): a smooth nonlinear map with a SimNorm output layer."""
	g = torch.Generator().manual_seed(seed)
	W1 = torch.randn(128, N, generator=g, dtype=DT) / N ** 0.5 * 8.0
	U = torch.randn(128, A, generator=g, dtype=DT) / A ** 0.5
	W2 = torch.randn(N, 128, generator=g, dtype=DT) / 128 ** 0.5 * gain
	return lambda z, a: lh.simnorm(torch.tanh(z @ W1.T + a @ U.T) @ W2.T)


def nominal_points(f, y, acts, h_max=128):
	return lh.rollout(f, lh.simnorm(y), acts, h_max)[:h_max]


def run_anchor(f, y, acts, etas, eps_list, real_points=None):
	if real_points is None:
		real_points = nominal_points(f, y, acts)
	return lh.anchor_errors(f, y, acts, etas, eps_list, GRID, real_points)


# --- S1 ------------------------------------------------------------------------------------------------------------

def test_s1_linear_map_is_exact():
	"""e_h <= 1e-12 at every h, for both linearisations, at the primary scale and at eps = 1.

	At eps = 0.01 the same code gives about 3e-12 (see DEVIATIONS.md C2): the absolute float64 error of two 128-step
	rollouts is about 4e-14, and e divides it by ||Delta||, which is 0.014 at that scale. That is a roundoff floor of the
	test, not a defect of the code, so the bound is asserted at eps >= 0.1 and a looser one at 0.01.
	"""
	f = linear_map(1)
	y, acts, etas = make_anchor(1)
	g = torch.Generator().manual_seed(7)
	real = lh.simnorm(torch.randn(128, N, generator=g, dtype=DT))   # arbitrary: a linear map has one Jacobian everywhere
	real[0] = lh.simnorm(y)
	res = run_anchor(f, y, acts, etas, [0.01, 0.1, 1.0], real_points=real)
	for eps, bound in ((0.1, 1e-12), (1.0, 1e-12), (0.01, 1e-11)):
		for L in ("nom", "real"):
			e = res[eps][L]["e"]
			assert np.isfinite(e).all() and e.max() <= bound, (eps, L, e.max())


# --- S2 ------------------------------------------------------------------------------------------------------------

def test_s2_error_scales_linearly_in_eps():
	f = nonlinear_map(2, gain=2.0)
	y, acts, etas = make_anchor(2)
	eps_list = [1e-4, 1e-3]
	res = run_anchor(f, y, acts, etas, eps_list)
	gi = GRID.index(4)
	m = {eps: np.median(res[eps]["nom"]["e"][:, gi]) for eps in eps_list}
	slope = np.log10(m[1e-3] / m[1e-4])
	assert 0.9 <= slope <= 1.1, (m, slope)
	# with the real points equal to the nominal ones, L-real is L-nom
	for eps in eps_list:
		assert np.allclose(res[eps]["real"]["e"], res[eps]["nom"]["e"], rtol=1e-9, atol=0)


def test_s2_h_star_is_non_increasing_in_eps():
	f = nonlinear_map(2, gain=2.0)
	eps_list = [1e-4, 1e-3, 1e-2, 1e-1]
	stacks = {eps: lh.Accumulator([eps], GRID) for eps in eps_list}
	for seed in (2, 3):
		y, acts, etas = make_anchor(seed)
		res = run_anchor(f, y, acts, etas, eps_list)
		for eps in eps_list:
			stacks[eps].add({eps: res[eps]})
	hs = []
	for eps in eps_list:
		by_h = lh.summarise(stacks[eps].stack(), GRID, [0.1], 0.1)[lh.key(eps)]["nom"]
		hs.append(by_h["h_star"]["0.1"])
	assert all(a >= b for a, b in zip(hs, hs[1:])), hs
	assert hs[0] > hs[-1], hs   # the test must not pass vacuously


def test_s2_real_points_that_differ_from_the_nominal_ones_floor_the_error():
	f = nonlinear_map(2, gain=2.0)
	y, acts, etas = make_anchor(2)
	other = nominal_points(f, y + 0.5 * torch.randn(N, generator=torch.Generator().manual_seed(9), dtype=DT), acts)
	other[0] = lh.simnorm(y)
	small = run_anchor(f, y, acts, etas, [1e-4], real_points=other)[1e-4]
	gi = GRID.index(8)
	assert np.median(small["real"]["e"][:, gi]) > 100 * np.median(small["nom"]["e"][:, gi])


# --- S3 ------------------------------------------------------------------------------------------------------------

def test_s3_perturbation_is_on_the_manifold_with_the_requested_size():
	y, _, etas = make_anchor(3)
	z = lh.simnorm(y)
	for eps in (0.01, 0.1, 1.0):
		s = lh.target_size(eps)
		for eta in etas:
			d = lh.make_delta(y, eta, s)
			assert d is not None
			zp = z + d
			assert (zp > 0).all()
			assert (zp.reshape(64, 8).sum(-1) - 1).abs().max() < 1e-12
			assert abs(torch.linalg.vector_norm(d).item() - s) <= 1e-9 * s


def test_s3_directions_are_deterministic_and_centred():
	a = lh.direction("walker-stand", 1, 0, 128, 0)
	b = lh.direction("walker-stand", 1, 0, 128, 0)
	c = lh.direction("walker-stand", 1, 0, 128, 1)
	assert torch.equal(a, b) and not torch.equal(a, c)
	assert a.reshape(64, 8).sum(-1).abs().max() < 1e-12
	assert lh.direction_seed("walker-stand", 1, 0, 128, 0) != lh.direction_seed("walker-stand", 2, 0, 128, 0)


def test_s3_frozen_test_vector_is_reproduced():
	tv = CFG["perturbation"]["seed_test_vector"]
	import hashlib
	assert hashlib.sha256(tv["string"].encode()).hexdigest() == tv["sha256"]
	assert lh.direction_seed("walker-stand", 1, 0, 128, 0) == tv["seed"]
	assert [float(x) for x in lh.draw_eta_raw(tv["seed"])[:3]] == tv["first_draws"]


def test_s3_scale_comes_from_the_config():
	p = CFG["perturbation"]
	assert lh.target_size(p["epsilon_primary"]) == 0.14142135623730953
	assert abs(lh.VERTEX - p["vertex_distance"]) < 1e-15
	assert abs(lh.DIAMETER - CFG["reported_descriptives"]["latent_diameter"]) < 1e-4
	assert p["size_rule"] == "epsilon_times_sqrt2" and p["m1_use"] == "reported_descriptive_only"


def test_s3_unreachable_target_is_unresolved_at_every_horizon():
	y, acts, etas = make_anchor(3)
	assert lh.make_delta(y, etas[0], 1000.0) is None
	f = linear_map(1)
	res = lh.anchor_errors(f, y, acts, etas[:2], [1000.0 / lh.VERTEX], GRID, nominal_points(f, y, acts))
	for L in ("nom", "real"):
		r = res[1000.0 / lh.VERTEX][L]
		assert np.isinf(r["e"]).all() and r["unres"].all() and np.isnan(r["pnorm"]).all()


def test_s3_a_response_below_the_floor_is_unresolved_at_that_horizon_only():
	y, acts, etas = make_anchor(3)
	zc = lh.simnorm(torch.zeros(N, dtype=DT))
	f = lambda z, a: zc.expand(*z.shape[:-1], N) + 0 * a.sum(-1, keepdim=True)   # constant map: the response is exactly 0
	res = run_anchor(f, y, acts, etas, [0.1])[0.1]["nom"]
	assert np.isinf(res["e"]).all() and res["unres"].all()


# --- S4 ------------------------------------------------------------------------------------------------------------

def test_s4_jvp_matches_explicit_float64_jacobian_products():
	g = torch.Generator().manual_seed(11)
	W1 = torch.randn(64, N, generator=g, dtype=DT) / N ** 0.5 * 3.0
	U = torch.randn(64, A, generator=g, dtype=DT) / A ** 0.5
	W2 = torch.randn(N, 64, generator=g, dtype=DT) / 8.0
	f = lambda z, a: torch.tanh(z @ W1.T + a @ U.T) @ W2.T + 0.3 * z
	y, acts, etas = make_anchor(4, n_dir=4)
	points = lh.rollout(f, lh.simnorm(y), acts, 16)
	grid = [1, 2, 4, 8, 16]
	out = lh.tangent_chain(f, points, acts, etas, grid)
	P = torch.eye(N, dtype=DT)
	for k in range(16):
		J = torch.func.jacrev(lambda z, a=acts[k]: f(z, a))(points[k])
		P = J @ P
		if k + 1 in grid:
			ref = etas @ P.T
			rel = torch.linalg.vector_norm(out[k + 1] - ref) / torch.linalg.vector_norm(ref)
			assert rel.item() <= 1e-10, (k + 1, rel.item())


def test_batched_jvp_equals_one_direction_at_a_time():
	f = nonlinear_map(5, gain=2.0)
	y, acts, etas = make_anchor(5, n_dir=4)
	pts = nominal_points(f, y, acts, 16)
	batch = lh.tangent_chain(f, pts, acts, etas, [16])[16]
	for b in range(4):
		one = lh.tangent_chain(f, pts, acts, etas[b:b + 1], [16])[16][0]
		assert torch.allclose(batch[b], one, rtol=1e-12, atol=1e-14)


def test_results_do_not_depend_on_which_scales_share_the_batch():
	f = nonlinear_map(2, gain=2.0)
	y, acts, etas = make_anchor(2)
	pts = nominal_points(f, y, acts)
	together = lh.anchor_errors(f, y, acts, etas, [1e-3, 1e-2, 1e-1], GRID, pts)
	for eps in (1e-3, 1e-2, 1e-1):
		alone = lh.anchor_errors(f, y, acts, etas, [eps], GRID, pts)[eps]
		for L in ("nom", "real"):
			assert np.array_equal(together[eps][L]["unres"], alone[L]["unres"])
			for k in ("e", "pnorm"):
				a, b = together[eps][L][k], alone[L][k]
				assert np.array_equal(np.isfinite(a), np.isfinite(b)) and np.allclose(a[np.isfinite(a)], b[np.isfinite(b)], rtol=1e-9)


# --- §2 validity horizon and §6 decision rules -----------------------------------------------------------------------

def test_validity_horizon_needs_a_pass_at_every_grid_horizon():
	assert lh.validity_horizon([True, True, True, False, True, True, True, True], GRID) == 4   # fail at 8, pass at 16
	assert lh.validity_horizon([False] + [True] * 7, GRID) == 0
	assert lh.validity_horizon([True] * 8, GRID) == 128
	assert lh.validity_horizon([True] * 7 + [False], GRID) == 64


def _hyp(meets):
	d = CFG["decision"]
	return V.decide_hypothesis(meets, CFG["models"]["tasks"], d["task_min_seeds"], d["hypothesis_min_tasks"])


TASKS = CFG["models"]["tasks"]


def test_four_of_five_tasks_support_and_three_do_not():
	ok = [True, True, False]
	assert _hyp({t: ok for t in TASKS[:4]} | {TASKS[4]: [False] * 3})["verdict"] == "SUPPORTED"
	assert _hyp({t: ok for t in TASKS[:3]} | {t: [False] * 3 for t in TASKS[3:]})["verdict"] == "REJECTED"


def test_one_lost_seed_is_judged_two_of_two():
	assert V.task_supports([True, True], 2) and not V.task_supports([True, False], 2)
	assert not V.task_supports([True], 2) and not V.task_supports([], 2)
	r = _hyp({t: [True, True] for t in TASKS[:4]} | {TASKS[4]: [True, False]})
	assert r["verdict"] == "SUPPORTED" and r["tasks_supporting"] == 4


def test_attrition_that_makes_support_impossible_is_inconclusive_and_the_literal_s4_reading_is_shown():
	meets = {t: [True, True, True] for t in TASKS[:3]} | {TASKS[3]: [True], TASKS[4]: []}
	r = _hyp(meets)
	assert r["verdict"] == "INCONCLUSIVE" and r["verdict_literal_s4"] == "REJECTED"
	assert r["tasks_with_enough_seeds"] == 3
	# attrition that leaves SUPPORTED reachable is an ordinary rejection
	r = _hyp({t: [True, True, True] for t in TASKS[:3]} | {TASKS[3]: [False] * 3, TASKS[4]: [True]})
	assert r["verdict"] == "REJECTED" and r["verdict_literal_s4"] == "REJECTED"


def _model(h_star_by_l):
	"""A stored per-model result whose medians pass exactly up to the given h* at the primary scale."""
	stats = {lh.key(0.1): {}}
	for L, hs in h_star_by_l.items():
		stats[lh.key(0.1)][L] = {"by_h": {str(h): {"median_e": 0.05 if h <= hs else 0.5} for h in GRID}}
	return {"stats": stats}


def test_decide_recomputes_h_star_and_keeps_the_secondary_verdict_nested():
	models = {t: [_model({"nom": 64, "real": 16}) for _ in range(3)] for t in TASKS}
	out = V.decide(models, CFG)
	assert out["primary"]["Q1.1"]["verdict"] == "SUPPORTED"
	assert out["primary"]["Q1.2"]["verdict"] == "REJECTED"
	assert out["secondary"]["Q1.2"]["verdict"] == "SUPPORTED"   # reported beside the primary, never instead of it
	models = {t: [_model({"nom": 32, "real": 32}) for _ in range(3)] for t in TASKS}
	out = V.decide(models, CFG)
	assert out["primary"]["Q1.1"]["verdict"] == "REJECTED" and out["secondary"]["Q1.1"]["verdict"] == "SUPPORTED"


def test_a_median_of_infinity_is_a_fail_and_a_median_at_the_tolerance_passes():
	assert lh.h_star_from_medians({str(h): {"median_e": float("inf")} for h in GRID}, GRID, 0.1) == 0
	assert lh.h_star_from_medians({str(h): {"median_e": 0.1} for h in GRID}, GRID, 0.1) == 128


def test_summarise_counts_unresolved_pairs_as_failures():
	y, acts, etas = make_anchor(6)
	f = linear_map(6)
	acc = lh.Accumulator([0.1], GRID)
	acc.add(run_anchor(f, y, acts, etas, [0.1]))
	st = lh.summarise(acc.stack(), GRID, CFG["decision"]["tolerances_reported"], 0.1)
	nom = st["0.1"]["nom"]
	assert nom["h_star"]["0.1"] == 128 and nom["by_h"]["64"]["unresolved_frac"] == 0.0
	# one pair in the stack forced unresolved flips a majority-unresolved median to a fail
	stack = acc.stack()
	for L in ("nom", "real"):
		stack[0.1][L]["e"][:, :] = np.inf
		stack[0.1][L]["unres"][:, :] = True
	assert lh.summarise(stack, GRID, [0.1], 0.1)["0.1"]["nom"]["h_star"]["0.1"] == 0


if __name__ == "__main__":
	tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
	failed = 0
	for name, fn in tests:
		try:
			fn()
			print(f"PASS  {name}", flush=True)
		except Exception as e:
			failed += 1
			print(f"FAIL  {name}: {type(e).__name__}: {e}", flush=True)
	print(f"\n{len(tests) - failed}/{len(tests)} passed")
	sys.exit(1 if failed else 0)
