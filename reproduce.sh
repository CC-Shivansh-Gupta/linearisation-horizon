#!/usr/bin/env bash
# One entry point. Needs a CUDA GPU for `run` (TD-MPC2 hard-codes cuda:0) and internet for `setup` and `download`.
#
#   bash reproduce.sh test       # the synthetic validation suite (S1-S4 and the decision logic); no GPU, no checkpoint
#   bash reproduce.sh setup      # install the pinned stack, fetch TD-MPC2 at the pinned commit
#   bash reproduce.sh download   # fetch the 15 checkpoints and SHA-256 check each; loads none of them
#   GO_AHEAD="<who, when, which message>" bash reproduce.sh run [tasks...]
#                                # S5 first, then the confirmatory run. Refuses without GO_AHEAD (PREREGISTRATION.md §9).
#   bash reproduce.sh verdict    # the verdict, from results/work. Run only after the reviewer releases the outputs.
set -euo pipefail
cd "$(dirname "$0")"
EXT=ext
CKPT=ckpt
cfg() { python -c "from scripts.prereg import load_cfg; c=load_cfg(); print($1)"; }

setup() {
	pip install -q -r requirements-tdmpc2.txt
	local commit; commit=$(cfg "c['models']['code_commit']")
	if [ ! -d "$EXT/tdmpc2/.git" ]; then git clone -q https://github.com/nicklashansen/tdmpc2 "$EXT/tdmpc2"; fi
	git -C "$EXT/tdmpc2" checkout -q "$commit"
	test "$(git -C "$EXT/tdmpc2" rev-parse HEAD | cut -c1-${#commit})" = "$commit"
	echo "tdmpc2 at $commit"
}

download() {   # hash check only: no torch.load
	local rev; rev=$(cfg "c['models']['checkpoint_revision']")
	mkdir -p "$CKPT"
	python - "$rev" "$CKPT" "$@" <<'PY'
import sys, urllib.request, hashlib
from pathlib import Path
from scripts.prereg import load_cfg
rev, out, tasks = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
c = load_cfg()
bad = 0
for name, want in c["models"]["checkpoint_sha256"].items():
	if tasks and name.rsplit("-", 1)[0] not in tasks:
		continue
	p = out / name
	if not p.exists():
		urllib.request.urlretrieve(f"https://huggingface.co/nicklashansen/tdmpc2/resolve/{rev}/dmcontrol/{name}", p)
	got = hashlib.sha256(p.read_bytes()).hexdigest()
	print(("OK   " if got == want else "BAD  ") + name)
	bad += got != want
# A bad file is NOT fatal here: the run records it as an exclusion, never a substitution (PREREGISTRATION.md §4, R2).
print(f"{bad} file(s) failed the hash check")
PY
}

case "${1:-}" in
	test)     for t in tests/test_*.py; do python "$t"; done ;;
	setup)    setup ;;
	download) shift; download "$@" ;;
	run)      shift
	          : "${GO_AHEAD:?set GO_AHEAD to the reviewer explicit execution go-ahead (who, when, which message)}"
	          python -m scripts.run_q1 --tdmpc2 "$EXT/tdmpc2/tdmpc2" --ckpt-dir "$CKPT" --go-ahead "$GO_AHEAD" ${@:+--tasks "$@"} ;;
	verdict)  python -m scripts.lh_verdict ;;
	*) sed -n 2,11p "$0"; exit 1 ;;
esac
