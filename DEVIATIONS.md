# Deviations and clarifications (PREREGISTRATION.md §12)

The frozen files (`README.md`, `PREREGISTRATION.md`, `config/prereg.yaml`) are byte-identical to commit `7c98789`.
This file is the only place a departure or a reading is recorded. Each entry states what had been seen when it was written.

**What had been seen, for every entry dated 6 Oct 2026 (C1-C7, N1):** nothing from any trained checkpoint. No `torch.load` of any of
the 15 files, no forward pass of any trained model, no discovery-set run and no S5. Only synthetic maps (S1-S4) were run.

---

## C1. Code written after the freeze (6 Oct 2026) — not a deviation

`scripts/lh.py` (core), `scripts/lh_verdict.py` (verdict), `scripts/prereg.py` (config loader) and
`tests/test_lh_synthetic.py` (S1-S4 plus the decision logic) were added on top of `7c98789`, per §9 step 2. The real-model
adapter (the file that supplies `f`, the logged actions and the encoder's pre-SimNorm output from a checkpoint) is **not
written yet** and nothing here imports `tdmpc2`, `dm_control` or any checkpoint.

## C2. S1 is asserted at ε ≥ 0.1; at ε = 0.01 the float64 roundoff floor is about 3e-12 (6 Oct 2026) — clarification

§10 states S1 as "e_h ≤ 10⁻¹² at every h" without naming a scale. On the test map (an orthogonal 512×512 map plus a small
action term, 128 steps) the measured worst case over 8 directions is:

| ε | s = ε√2 | max e_h, L-nom and L-real |
|---|---|---|
| 1 | 1.414 | 2.5e-14 |
| 0.1 | 0.1414 (primary) | 2.5e-13 |
| 0.01 | 0.01414 | 2.5e-12 |

The absolute float64 error of the difference of two 128-step rollouts is about 3.5e-14, and e divides it by ‖Δ_h‖, which
is about s for a norm-preserving map. So the bound holds with a factor 4 to spare at the primary scale, and the code is
exact to float64 resolution. At the reported scale ε = 0.01 the same floor is 2.5e-12, which is above 1e-12. The test
asserts 1e-12 at ε = 0.1 and 1, and 1e-11 at ε = 0.01. The tolerances that decide anything (0.05 to 0.5) are 10 orders above
this floor.

## C3. S2 is evaluated on L-nom, and on L-real only when its points coincide with the nominal ones (6 Oct 2026) — clarification

§10 says "for small ε, median e_h scales linearly in ε". That holds for L-nom, whose Jacobians are taken along the same
rollout the perturbation follows. L-real linearises at different points (the encoded real states), so its error does
**not** vanish as ε → 0: it floors at the point mismatch. The test therefore checks three things:
- L-nom: the log-log slope over ε ∈ {1e-4, 1e-3} at h = 4 is in [0.9, 1.1] (measured 1.00);
- L-real with the real points set equal to the nominal ones equals L-nom to rtol 1e-9;
- L-real with deliberately different points is more than 100 times worse than L-nom at ε = 1e-4 (the floor).

The S2 map is `z' = SimNorm(2 · W₂ tanh(W₁ z + U a))`. An earlier gain of 40 was far too expansive (errors 1e35 at h = 64) and
its slope was 0.89 because second-order terms already dominated at ε = 1e-3; gain 2 gives e ∝ ε at every h up to 16 and
h\* = 64, 32, 16, 2 for ε = 1e-4, 1e-3, 1e-2, 1e-1. This is a choice of test map, not of any protocol parameter.

## C4. Attrition: §4 and §13 R2 word one case differently; the verdict script writes both (6 Oct 2026) — clarification

§4 says a task that loses two or more seeds "cannot support any hypothesis, and counts as not supporting", which read
literally gives REJECTED. §13 R2, the reviewer's later and more specific decision, says that if attrition makes the
prespecified support rule impossible "the result is **inconclusive**". These differ only when fewer than 4 of the 5 tasks
keep at least 2 loadable seeds, because only then can SUPPORTED not be reached.

`scripts/lh_verdict.py` follows R2 in the field `verdict` (INCONCLUSIVE) and writes the §4 reading in `verdict_literal_s4`
(REJECTED), so the two are visible side by side. In every other case they are equal. **This is flagged for the reviewer;
it changes no number and no threshold.**

## C5. Median over pairs with an unresolved pair counted as +∞ (6 Oct 2026) — clarification

The median is `numpy.median` over all 2,000 pairs, with an unresolved pair counted as +∞ (§3.4). With an even count, NumPy
averages the two middle values, so exactly 1,000 resolved pairs out of 2,000 can give +∞, a fail. That is the conservative
side, and the 90th percentile uses `method="higher"` so it never interpolates through an infinity.

## N1. Observation, no change: a contractive model fails by being unresolved, not by linearisation error (6 Oct 2026)

§3.4 marks a pair unresolved at horizon h if ‖Δ_h‖ < 10⁻¹⁰ · ‖δ‖, and an unresolved pair counts as +∞. On a synthetic
contractive map (gain 1) the response falls below that floor by h = 64 and the median becomes +∞, so h\* stops at 32 even
though the linearisation is accurate wherever it is defined. Protocol 1 found R-full strongly contractive (γ between 0.55 and
0.82), so a real network can be contractive over 64 steps. The frozen rule is applied as written. `unresolved_frac` is
reported at every h, so a reader can tell a fail through unresolved pairs from a fail through error. Nothing is changed.

## C6. Real-model adapter and driver added, NOT RUN (6 Oct 2026) — not a deviation

Files: `scripts/tdmpc2_adapter.py`, `scripts/run_q1.py`, `reproduce.sh`, `requirements-tdmpc2.txt`, and `results/ext/ckpt` in
`.gitignore`. **Nothing here has been executed.** Only `py_compile` and `bash -n` were run on them, so no import of
`tdmpc2`, `dm_control` or `torch.load` has happened, and the first real execution of this path is S5. What had been seen:
nothing from any checkpoint, and no TD-MPC2 code was run.

Choices a reader should know, each of which makes the run stricter or more recorded, never looser:
- **A run needs a recorded go-ahead.** `run_q1.py` refuses to start without `--go-ahead`, which is copied into
  `results/RUN.json`. It also refuses unless the freeze commit is an ancestor of HEAD, the three frozen files equal the freeze
  commit, and no tracked file is modified. HEAD is printed and written into `RUN.json` and every per-model file (§11).
- **S5 runs first and aborts the run on any failure.** It stores pass/fail per check and no value. Beyond the four things
  §10 lists (shapes, dtypes, determinism, L-nom = L-real at k = 0), it also checks that `SimNorm(y)` reproduces the encoder, that
  the JVP through the real network equals the explicit Jacobian product at one step, and that batched `next` equals unbatched.
  These are plumbing checks on a random-initialisation network; they compare two computations of the same quantity.
- **The run computes no verdict and prints no statistic.** It writes `results/MANIFEST.sha256` and prints that file's
  SHA-256, so the outputs can stay sealed and only a digest be sent. `results/` is git-ignored.
- **Load handling.** `agent.load(path)` in the pinned code, with every exception recorded as a load failure and the file
  excluded (§4, R2). A failed SHA-256 check is also an exclusion. An absent file raises, so a download problem cannot masquerade
  as an exclusion. An architecture assertion failing inside the load block is recorded as an exclusion with its message.
- **`lh.anchor_errors` now batches all scales together.** A test (`test_results_do_not_depend_on_which_scales_share_the_batch`)
  checks that a pair's result does not depend on which other pairs share the batch. This is a change of speed, not of
  definition.
- **‖B̃_τ‖₂ (§5)** is computed as the spectral norm of ∂f/∂a at (z_τ, a_τ) in the full 512 coordinates. The output lies in the
  SimNorm tangent space, so this equals the norm of Uᵀ B used by protocols 1-2.
- **Cost estimate, unverified:** about 190k vmapped JVP calls of batch 24 per model, in float64 on a T4. Protocol 1 took about
  23 minutes per seed for its Jacobians. No Q1 timing exists.

## C7. The run is split into two commands, S5 and confirmatory, each with its own go-ahead (6 Oct 2026) — not a deviation

The reviewer approved the code-writing step (6 Oct 2026) and asked, for the execution gate, for one exact command for S5
and one for the confirmatory run. `run_q1.py` therefore has two stages, `s5` and `confirmatory`, and `reproduce.sh` has
`s5` and `confirmatory` in place of `run`. What had been seen: as in C6, nothing from any checkpoint, and no TD-MPC2 code
was run. Neither stage has been executed.
- **Each stage needs its own `--go-ahead` record** and repeats the provenance checks of C6.
- **`confirmatory` refuses unless `results/s5/s5.json` exists, was written at the same HEAD, and every check passed.** A fix
  after a failed S5 is a new commit, so S5 must pass again at that commit before any checkpoint is loaded. A failed S5 is
  now recorded in `s5.json` (`"passed": false`) and the stage exits non-zero, where C6 raised before writing.
- **Output layout:** `results/s5/` and `results/confirmatory/`. Each ends with `MANIFEST.sha256` in `sha256sum -b` format
  (`<hex> *./<path>`, one line per output file, byte order of the path) and the SHA-256 of that manifest written to
  `results/{stage}.manifest-digest.txt`, outside the directory it seals, together with HEAD. `scripts/lh_verdict.py` now
  reads `results/confirmatory/` by default.
- **`tests/test_run_gate.py`** (6 tests) checks the S5 gate against hand-written `s5.json` files and the manifest format on
  a temporary directory. No GPU, TD-MPC2 or checkpoint is involved, and S5 itself is not run.
- `ENVIRONMENT.txt` is now written as bytes, so it has LF line endings on every platform.
- **The confirmatory console shows no episode return.** C6 printed each model's mean return when it finished. The protocol-2
  incident (P1) recorded episode returns as one of the things seen before authorisation, so the console now prints only
  the model name and wall time. The returns stay in the sealed `diagnostics/{task}_s{seed}.json`.
