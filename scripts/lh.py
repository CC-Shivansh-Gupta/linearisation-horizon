"""Q1 core (PREREGISTRATION.md §2, §3, §5). Pure functions of a map f(z, a) -> z'; no model and no checkpoint is
imported here.

Written after the freeze and validated only on the synthetic cases S1-S4 (§9 step 2, §10). The real-model adapter that
supplies f, the logged actions and the encoder's pre-SimNorm output is a separate file and is not run until the reviewer
authorises it.

Conventions
  f(z, a) maps a latent z of shape (..., 512) and an action a of the same leading shape to the next latent. It must
  also accept a single unbatched z of shape (512,) with a of shape (A,), because torch.func.jvp linearises one point.
  Everything is float64.
"""
import hashlib
import math

import numpy as np
import torch
from torch.func import jvp, vmap

GROUP = 8                      # SimNorm group size (tdmpc2 simnorm_dim)
VERTEX = math.sqrt(2.0)        # distance between two vertices of a simplex of any dimension
DIAMETER = math.sqrt(128.0)    # 64 simplices of 8: sqrt(64) * sqrt(2)
UNRESOLVED_REL = 1.0e-10       # config response.unresolved_if_response_below_rel


# --- §3.4 the perturbation ---------------------------------------------------------------------------------------

def simnorm(y, group=GROUP):
	"""softmax within each consecutive group of `group` coordinates, as tdmpc2's SimNorm."""
	shp = y.shape
	return torch.softmax(y.reshape(*shp[:-1], -1, group), dim=-1).reshape(shp)


def direction_seed(task, seed, episode, tau, j):
	"""First 8 bytes, big-endian, of SHA-256('{task}|{seed}|{episode}|{tau}|{j}')."""
	text = f"{task}|{seed}|{episode}|{tau}|{j}"
	return int.from_bytes(hashlib.sha256(text.encode("ascii")).digest()[:8], "big")


def draw_eta_raw(seed, dim=512):
	"""One call, nothing else drawn from the generator."""
	return np.random.Generator(np.random.PCG64(seed)).standard_normal(dim)


def centre_groups(eta, group=GROUP):
	"""Subtract the mean within each group. Softmax ignores a per-group constant."""
	e = np.asarray(eta, dtype=np.float64).reshape(-1, group)
	return (e - e.mean(axis=1, keepdims=True)).reshape(-1)


def direction(task, seed, episode, tau, j):
	"""The centred direction eta as a float64 tensor, from the frozen seed rule."""
	raw = draw_eta_raw(direction_seed(task, seed, episode, tau, j))
	return torch.from_numpy(centre_groups(raw))


def target_size(eps):
	"""s = eps * sqrt(2). No checkpoint quantity (review O2)."""
	return eps * VERTEX


def make_delta(y, eta, s, t_max=100.0, rel_tol=1.0e-9, max_iter=200):
	"""delta = SimNorm(y + t* eta) - SimNorm(y), with ||delta||_2 = s to rel_tol, by bisection on t in [0, t_max].

	Returns None if the target is not reached by t_max (the pair is unresolved at every horizon) or if bisection does
	not converge. z + delta is exactly a SimNorm output, so it lies on the product of simplices.
	"""
	z = simnorm(y)

	def dist(t):
		return torch.linalg.vector_norm(simnorm(y + t * eta) - z).item()

	if not dist(t_max) >= s:
		return None
	lo, hi = 0.0, t_max
	for _ in range(max_iter):
		mid = 0.5 * (lo + hi)
		d = dist(mid)
		if abs(d - s) <= rel_tol * s:
			return simnorm(y + mid * eta) - z
		if d < s:
			lo = mid
		else:
			hi = mid
	return None


# --- §3.2, §3.3 rollouts and first-order predictions -------------------------------------------------------------

def _apply(f, z, a):
	return f(z, a.expand(*z.shape[:-1], a.shape[-1]))


def rollout(f, z0, acts, h_max):
	"""Open-loop nonlinear rollout under the logged actions. z0: (..., 512). Returns (..., h_max + 1, 512), entry k = F_k(z0)."""
	zs = [z0]
	z = z0
	with torch.no_grad():
		for k in range(h_max):
			z = _apply(f, z, acts[k])
			zs.append(z)
	return torch.stack(zs, dim=-2)


def tangent_chain(f, points, acts, V, grid):
	"""v_0 = V, v_{k+1} = (df/dz)(points[k], acts[k]) v_k, by forward-mode JVP (§3.3).

	points: (>= max(grid), 512) linearisation points (the nominal rollout for L-nom, the encoded real states for L-real).
	V: (B, 512), B tangent vectors propagated together. Returns {h: (B, 512)} for h in grid.
	"""
	wanted = set(grid)
	out = {}
	for k in range(max(grid)):
		a = acts[k]

		def g(z, a=a):
			return f(z, a)

		V = vmap(lambda v, z=points[k], g=g: jvp(g, (z,), (v,))[1])(V)
		if k + 1 in wanted:
			out[k + 1] = V
	return out


# --- §2 errors at one anchor ---------------------------------------------------------------------------------------

def _per_h(delta_h, pred_h, delta_norm):
	"""e, angle (deg), ||P delta||, and the response-too-small flag, for (B, 512) arrays at one horizon."""
	dn = torch.linalg.vector_norm(delta_h, dim=-1)
	pn = torch.linalg.vector_norm(pred_h, dim=-1)
	small = (dn < UNRESOLVED_REL * delta_norm).cpu().numpy()
	e = (torch.linalg.vector_norm(delta_h - pred_h, dim=-1) / dn).cpu().numpy()
	cos = ((delta_h * pred_h).sum(-1) / (dn * pn)).cpu().numpy()
	angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
	return e, angle, pn.cpu().numpy(), small


def anchor_errors(f, y_tau, acts, etas, eps_list, grid, real_points):
	"""Errors of both linearisations at one anchor, for every eps and horizon.

	y_tau: (512,) the encoder's pre-SimNorm output at o_tau, so that z_tau = SimNorm(y_tau).
	acts: (>= max(grid), A) logged actions a_tau, a_{tau+1}, ...
	etas: (B, 512) centred directions.
	real_points: (>= max(grid), 512) the encoded real states z_{tau+k}, k = 0, 1, ...; real_points[0] must equal z_tau.
	Returns {eps: {"nom"|"real": {"e", "angle", "pnorm": (B, G) float arrays, "unres": (B, G) bool}}}.
	An unresolved pair has e = +inf and angle = pnorm = nan, at every h if the target size is not reached, and at that
	h only if the response is below 1e-10 of the perturbation.

	The perturbations are built on the CPU (bisection is scalar work), then every resolved (eps, direction) pair is
	propagated in one batch. A pair's result does not depend on which other pairs share the batch.
	"""
	h_max, B, G = max(grid), etas.shape[0], len(grid)
	eps_list = list(eps_list)
	z0 = simnorm(y_tau)
	nom = rollout(f, z0, acts, h_max)
	y_cpu, etas_cpu = y_tau.cpu(), etas.cpu()
	deltas = {eps: [make_delta(y_cpu, etas_cpu[b], target_size(eps)) for b in range(B)] for eps in eps_list}
	res = {eps: {L: {"e": np.full((B, G), np.inf), "angle": np.full((B, G), np.nan),
	                 "pnorm": np.full((B, G), np.nan), "unres": np.ones((B, G), dtype=bool)} for L in ("nom", "real")}
	       for eps in eps_list}
	pairs = [(ei, b) for ei, eps in enumerate(eps_list) for b in range(B) if deltas[eps][b] is not None]
	if not pairs:
		return res
	ei_arr, b_arr = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
	D = torch.stack([deltas[eps_list[ei]][b] for ei, b in pairs]).to(device=z0.device, dtype=z0.dtype)
	dnorm = torch.linalg.vector_norm(D, dim=-1)
	pert = rollout(f, z0 + D, acts, h_max)
	preds = {"nom": tangent_chain(f, nom, acts, D, grid), "real": tangent_chain(f, real_points, acts, D, grid)}
	for gi, h in enumerate(grid):
		delta_h = pert[:, h] - nom[h]
		for L in ("nom", "real"):
			e, angle, pn, small = _per_h(delta_h, preds[L][h], dnorm)
			e = np.where(np.isfinite(e), e, np.inf)
			e[small] = np.inf
			angle[small] = np.nan
			pn = np.where(small, np.nan, pn)
			for ei, eps in enumerate(eps_list):
				m = ei_arr == ei
				r = res[eps][L]
				r["e"][b_arr[m], gi] = e[m]
				r["angle"][b_arr[m], gi] = angle[m]
				r["pnorm"][b_arr[m], gi] = pn[m]
				r["unres"][b_arr[m], gi] = small[m]
	return res


# --- §5 per-model statistics, §2 validity horizon -----------------------------------------------------------------

class Accumulator:
	"""Stacks anchor_errors results over the anchors and episodes of one model."""

	def __init__(self, eps_list, grid):
		self.eps_list, self.grid = list(eps_list), list(grid)
		self._parts = {eps: {L: [] for L in ("nom", "real")} for eps in self.eps_list}

	def add(self, anchor_result):
		for eps in self.eps_list:
			for L in ("nom", "real"):
				self._parts[eps][L].append(anchor_result[eps][L])

	def stack(self):
		return {eps: {L: {k: np.concatenate([p[k] for p in parts], axis=0) for k in ("e", "angle", "pnorm", "unres")}
		              for L, parts in by_l.items()} for eps, by_l in self._parts.items()}


def validity_horizon(passes, grid):
	"""Largest grid horizon h with a pass at every grid horizon <= h; 0 if the first grid horizon fails (§2)."""
	n = 0
	for ok in passes:
		if not ok:
			break
		n += 1
	return grid[n - 1] if n else 0


def key(x):
	"""JSON key for a scale or tolerance."""
	return f"{x:g}"


def summarise(stack, grid, tolerances, tol_primary):
	"""The §5 statistics for one model from Accumulator.stack().

	Medians and percentiles use every pair, with an unresolved pair counted as +inf. A median that is +inf is a fail.
	"""
	stats = {}
	for eps, by_l in stack.items():
		stats[key(eps)] = {}
		for L, d in by_l.items():
			rows = {}
			for gi, h in enumerate(grid):
				e = d["e"][:, gi]
				ang = d["angle"][:, gi]
				rows[str(h)] = {
					"median_e": float(np.median(e)),
					"p90_e": float(np.percentile(e, 90, method="higher")),
					"median_angle_deg": float(np.nanmedian(ang)) if np.isfinite(ang).any() else None,
					"overshoot_frac": float(np.mean(d["pnorm"][:, gi] > DIAMETER)),
					"unresolved_frac": float(np.mean(d["unres"][:, gi])),
					"n_pairs": int(e.size),
				}
			h_star = {key(t): validity_horizon([rows[str(h)]["median_e"] <= t for h in grid], grid) for t in tolerances}
			stats[key(eps)][L] = {"by_h": rows, "h_star": h_star}
	return stats


def h_star_from_medians(by_h, grid, tol):
	"""Recompute h* from stored per-horizon medians against a tolerance; this is what the verdict script uses."""
	return validity_horizon([by_h[str(h)]["median_e"] <= tol for h in grid], grid)
