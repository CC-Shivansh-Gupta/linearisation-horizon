"""The driver's gates and receipts (scripts/run_q1.py), on files in a temporary directory.
No GPU, no TD-MPC2, no checkpoint and no S5: only the S5 *gate* is tested, against hand-written s5.json files.

Run with `python -m pytest tests` or `python tests/test_run_gate.py`.
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import run_q1 as R

HEAD = "a" * 40


def _s5(d, head=HEAD, ok=True):
	d.mkdir(parents=True, exist_ok=True)
	checks = {"shapes": True, "L_nom_equals_L_real_at_k0": ok}
	R._write(d / "s5.json", {"head": head, "checks": checks, "passed": all(checks.values()), "values_printed": False})
	return d / "s5.json"


def _raises(fn, *args):
	try:
		fn(*args)
	except (AssertionError, FileNotFoundError):
		return True
	return False


def test_confirmatory_gate_accepts_a_passing_s5_at_the_same_head():
	with tempfile.TemporaryDirectory() as t:
		R.s5_passed_at(_s5(Path(t) / "s5"), HEAD)


def test_confirmatory_gate_refuses_without_s5():
	with tempfile.TemporaryDirectory() as t:
		assert _raises(R.s5_passed_at, Path(t) / "s5" / "s5.json", HEAD)


def test_confirmatory_gate_refuses_a_failed_s5():
	with tempfile.TemporaryDirectory() as t:
		assert _raises(R.s5_passed_at, _s5(Path(t) / "s5", ok=False), HEAD)


def test_confirmatory_gate_refuses_s5_from_another_commit():
	with tempfile.TemporaryDirectory() as t:
		assert _raises(R.s5_passed_at, _s5(Path(t) / "s5", head="b" * 40), HEAD)


def test_manifest_is_sha256sum_b_format_and_its_digest_sits_outside_the_sealed_directory():
	R.HEAD = HEAD
	with tempfile.TemporaryDirectory() as t:
		out = Path(t) / "confirmatory"
		(out / "work").mkdir(parents=True)
		run, work = b"{}\n", b'{"a": 1}\n'
		(out / "work" / "x_s1.json").write_bytes(work)
		(out / "RUN.json").write_bytes(run)
		digest, n = R.write_manifest(out)
		lines = (out / "MANIFEST.sha256").read_bytes().decode().splitlines()
		assert n == 2 and lines == [hashlib.sha256(run).hexdigest() + " *./RUN.json",
		                            hashlib.sha256(work).hexdigest() + " *./work/x_s1.json"]
		assert digest == hashlib.sha256((out / "MANIFEST.sha256").read_bytes()).hexdigest()
		receipt = (Path(t) / "confirmatory.manifest-digest.txt").read_bytes()
		assert receipt == f"{digest}  confirmatory/MANIFEST.sha256  head {HEAD}\n".encode()
		assert not (out / "confirmatory.manifest-digest.txt").exists()


def test_manifest_changes_if_any_sealed_byte_changes():
	R.HEAD = HEAD
	with tempfile.TemporaryDirectory() as t:
		out = Path(t) / "s5"
		_s5(out)
		d1, _ = R.write_manifest(out)
		rec = json.loads((out / "s5.json").read_text())
		rec["values_printed"] = True
		R._write(out / "s5.json", rec)
		d2, _ = R.write_manifest(out)
		assert d1 != d2


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
