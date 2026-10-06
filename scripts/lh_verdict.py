"""Q1 verdict (PREREGISTRATION.md §6). Reads only config/prereg.yaml and the per-model result files.

    python -m scripts.lh_verdict [results_dir]

Per-model files: results/work/{task}_s{seed}.json (the §5 statistics, keyed by scale then linearisation) and, for an
excluded checkpoint, results/diagnostics/{task}_s{seed}_load_failure.json. h* is recomputed here from the stored
per-horizon medians against the tolerance in the config, so no stored h* decides anything.

Verdict labels
  SUPPORTED     at least `hypothesis_min_tasks` tasks support the hypothesis.
  REJECTED      attrition leaves SUPPORTED reachable, and fewer tasks support it.
  INCONCLUSIVE  attrition leaves fewer than `hypothesis_min_tasks` tasks with enough loadable seeds, so SUPPORTED cannot be
                reached (§13, R2: the result is inconclusive, never repaired by substitution). §4 words the same case as
                "counts as not supporting", which read literally is REJECTED. Both are written out: `verdict` follows R2
                and `verdict_literal_s4` follows §4, so a reader can see that they differ and only in this case.
"""
import json
import sys
from pathlib import Path

from scripts.lh import h_star_from_medians, key
from scripts.prereg import ROOT, load_cfg

HYPOTHESES = {"Q1.1": "nom", "Q1.2": "real"}


def task_supports(meets, min_seeds):
	"""meets: one bool per LOADED seed. Needs at least min_seeds loaded, and at least min_seeds of them meeting."""
	return len(meets) >= min_seeds and sum(meets) >= min_seeds


def decide_hypothesis(meets_by_task, tasks, min_seeds, min_tasks):
	"""meets_by_task: {task: [bool per loaded seed]}. Returns the verdict record for one hypothesis at one window."""
	supports = {t: task_supports(meets_by_task.get(t, []), min_seeds) for t in tasks}
	able = [t for t in tasks if len(meets_by_task.get(t, [])) >= min_seeds]
	n_support = sum(supports.values())
	if n_support >= min_tasks:
		verdict = literal = "SUPPORTED"
	else:
		literal = "REJECTED"
		verdict = "REJECTED" if len(able) >= min_tasks else "INCONCLUSIVE"
	return {"verdict": verdict, "verdict_literal_s4": literal, "tasks_supporting": n_support,
	        "tasks_with_enough_seeds": len(able), "task_supports": supports,
	        "seeds_meeting": {t: [int(m) for m in meets_by_task.get(t, [])] for t in tasks}}


def decide(models, cfg):
	"""models: {task: [per-model result dict, ...]} for the LOADED checkpoints only (§4).

	Each result dict carries stats[key(eps)][L]["by_h"]. The verdict-bearing scale is epsilon_primary and the
	verdict-bearing tolerance is tolerance_primary (§2, §6).
	"""
	dec, hz = cfg["decision"], cfg["horizons"]
	grid, eps_key = hz["grid"], key(cfg["perturbation"]["epsilon_primary"])
	tol = dec["tolerance_primary"]
	tasks = cfg["models"]["tasks"]
	out = {"tolerance": tol, "epsilon": cfg["perturbation"]["epsilon_primary"], "primary": {}, "secondary": {}, "models": {}}
	for name, L in HYPOTHESES.items():
		per_model = {t: [h_star_from_medians(m["stats"][eps_key][L]["by_h"], grid, tol) for m in models.get(t, [])]
		             for t in tasks}
		out["models"][name] = per_model
		for label, window in (("primary", hz["primary_window"]), ("secondary", hz["secondary_window"])):
			meets = {t: [h >= window for h in hs] for t, hs in per_model.items()}
			out[label][name] = {"window": window,
			                    **decide_hypothesis(meets, tasks, dec["task_min_seeds"], dec["hypothesis_min_tasks"])}
	return out


def load_set(base, cfg):
	"""{task: [result]} for loaded seeds, and the list of excluded checkpoints. A file that is missing with no recorded
	load failure is an error: nothing is dropped silently."""
	models, excluded = {}, []
	for task in cfg["models"]["tasks"]:
		models[task] = []
		for s in cfg["models"]["seeds"]:
			if (base / "diagnostics" / f"{task}_s{s}_load_failure.json").exists():
				excluded.append(f"{task}-{s}")
				continue
			p = base / "work" / f"{task}_s{s}.json"
			if not p.exists():
				raise FileNotFoundError(f"{p} missing with no recorded load failure")
			models[task].append(json.loads(p.read_text()))
	return models, excluded


def main(argv):
	base = Path(argv[1]) if len(argv) > 1 else ROOT / "results"
	cfg = load_cfg()
	models, excluded = load_set(base, cfg)
	out = decide(models, cfg)
	out["excluded"] = excluded
	text = json.dumps(out, indent=2, sort_keys=True) + "\n"
	(base / "verdict.json").write_bytes(text.encode("utf-8"))   # bytes, so Windows does not write CRLF
	for label in ("primary", "secondary"):
		for name, v in out[label].items():
			print(f"{label:9s} {name} (h* >= {v['window']}): {v['verdict']}  {v['tasks_supporting']}/"
			      f"{len(cfg['models']['tasks'])} tasks")
	if excluded:
		print("excluded:", ", ".join(excluded))


if __name__ == "__main__":
	main(sys.argv)
