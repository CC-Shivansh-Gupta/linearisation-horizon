# Preregistration — The Linearisation Validity Horizon of a Learned World Model

**Protocol version:** Q1-1 (a new question; not a revision of either Hankel protocol)
**Author:** Shivansh Gupta (IIT Bombay)
**Track:** The Bu1LD research track, reviewed by Ryan Gomez
**Status:** FROZEN at the first commit that contains this file. No checkpoint in §4 had been loaded in any form, and no
forward pass of any trained checkpoint had been run for this question, at that commit. The reviewer's decisions on every
choice that was open are recorded in §13. Every numeric parameter is in `config/prereg.yaml`.

---

## 0. Relation to the completed studies

Two preregistered studies asked whether the Hankel spectrum of TD-MPC2's linearised latent dynamics defines an
abstraction for hierarchical planning. Both were rejected:
- protocol 1: frozen `0173614f`, result `f0e8c94c`;
- protocol 2: frozen `c2474be5`, result `00a36164`;
- the closeout note is `hankel-hierarchy/NOTE.md` at `480306d4`.

Both studies built their Gramians from products of up to 64 Jacobians, and neither measured whether those products
describe the network (closeout note, limitation 1). This protocol measures that. It is **separate** from the completed
studies:
- It tests the validity of a first-order lens, not the presence of any structure. **No outcome here can produce, restore
  or strengthen a Hankel-structure claim.**
- **No outcome here changes any verdict of protocol 1 or 2.** Those verdicts are statements about the frozen
  construction, and they stand whatever this protocol finds.
- It uses **only checkpoints that neither study loaded** (§4). So it bears on the model class, not on the 35 models
  those studies analysed. Any transfer of its answer to those models is an inference, and is labelled as one.

## 1. Question and hypotheses

**Question.** Over what horizon does the first-order linearisation of TD-MPC2's latent dynamics predict the model's
own nonlinear response to a latent perturbation?

Two linearisations are tested. They differ only in where the Jacobians are evaluated (§3.3):
- **L-nom, the tangent-linear model.** Jacobians along the model's own nominal rollout. This is the exact first-order
  prediction of the nonlinear response, so its error is pure nonlinearity at the perturbation scale.
- **L-real, the lens as used.** Jacobians at the encoded real states, which is how protocols 1 and 2 built Φ. Its error
  is nonlinearity plus the mismatch between the real-state linearisation points and the states the model actually
  passes through.

Each hypothesis has its own verdict, and neither rescues the other:

- **Q1.1 (tangent validity over the Gramian window).** L-nom predicts the nonlinear latent response to within the
  frozen tolerance at every grid horizon up to **h = 64**.
- **Q1.2 (lens validity over the Gramian window).** L-real does the same up to **h = 64**.

For each of Q1.1 and Q1.2 there is also a **secondary verdict at h = 16**, the H2.3 window. It is nested: validity to
64 implies validity to 16. So it can never turn a rejection at 64 into support, and it is reported beside the
primary verdict, never instead of it.

## 2. What "predicts the nonlinear response" means

For one model, one anchor τ, one perturbation δ and one horizon h (definitions in §3):

- the **reference response** is Δ_h = F_h(z_τ + δ) − F_h(z_τ), the difference between two nonlinear rollouts of the
  model's own dynamics under the same logged actions;
- the **first-order prediction** is P_h δ, where P_h is the product of h Jacobians of L-nom or of L-real;
- the **error** is e_h = ‖Δ_h − P_h δ‖₂ / ‖Δ_h‖₂.

**L predicts the nonlinear response at horizon h** for a model if the **median of e_h over all anchors and
perturbations of that model is ≤ 0.10**, at the primary perturbation scale (ε = 0.1).

**The validity horizon** h\*_L of a model is the largest grid horizon h such that L predicts the response at **every**
grid horizon ≤ h. If L fails at h = 1, then h\* = 0. Validity must hold at every grid horizon up to h\*, so a
failure at 8 followed by a pass at 16 gives h\* = 4.

**Why 0.10.** A relative error e ≤ 0.1 bounds the angle between Δ_h and P_hδ by arcsin(0.1) = 5.7°. That is well below
the 15–30° angle thresholds and margins on which protocols 1 and 2 decided. At this tolerance, linearisation error alone
could not have moved one of their subspace statistics across a threshold.

## 3. Construction

### 3.1 Inherited from protocols 1–2, unchanged

- **Model and code:** TD-MPC2, single-task, state-based, `model_size=5`. Code at `nicklashansen/tdmpc2@e9f5932`,
  checkpoints at Hugging Face revision `8fb2a82e`.
- **Rollouts:** 10 episodes, with env seed = torch seed = 0…9 and `agent.act(obs, t0, eval_mode=True)` (protocol 1, C7).
  DMControl episodes are 500 agent steps.
- **Numerics:** a float64 copy of the weights, in evaluation mode. z_t = enc(o_t) is re-encoded in float64 from the
  observation that the float32 agent saw (protocol 1, C8).
- **Environment:** dm-control 1.0.24 and mujoco 3.2.4 (protocol 1, C11), recorded in `ENVIRONMENT.txt`.
- **Anchors:** τ ∈ {128, 138, …, 368}, 25 per episode and 250 per model. The largest grid horizon is 128, and
  τ + 128 ≤ 496 < 500.

### 3.2 Nonlinear rollouts and the reference response

- F_h(z) is h applications of the dynamics, z ↦ f(z, a) with `WorldModel.next`, using the **logged** actions
  a_τ, …, a_{τ+h−1} of that episode. It is **open-loop**: the planner is not re-run on imagined states.
- The **nominal rollout** is ẑ_{τ+k} = F_k(z_τ).
- The **perturbed rollout** is F_k(z_τ + δ).
- Both rollouts use the same actions, the same float64 weights and evaluation mode.

### 3.3 The two first-order predictions

Both are computed by forward-mode Jacobian–vector products (`torch.func.jvp`) in float64. This is mathematically
identical to multiplying the explicit Jacobians, and synthetic case S4 checks that.

- **L-nom:** v₀ = δ, then v_{k+1} = ∂f/∂z(ẑ_{τ+k}, a_{τ+k}) v_k. Jacobians are taken along the nominal rollout.
- **L-real:** v₀ = δ, then v_{k+1} = ∂f/∂z(z_{τ+k}, a_{τ+k}) v_k, with z_{τ+k} = enc(o_{τ+k}). This is exactly
  Φ(τ+h, τ)δ of protocols 1–2.

Every δ lies in the SimNorm tangent space, since each group of 8 sums to zero (§3.4). The dynamics output lies on the
product of simplices, so its Jacobian maps into the same space. So working in the full 512 coordinates is equivalent
to the 448-dimensional tangent coordinates U of protocols 1–2, and no projection is needed.

### 3.4 The perturbation

**Direction, built on the state manifold.**
- Let y_τ be the encoder's pre-SimNorm output at o_τ, the LayerNorm output of its last layer, so that z_τ = SimNorm(y_τ).
- Draw η ~ N(0, I₅₁₂) and subtract its mean within each group of 8. Softmax ignores per-group constants. The draw is
  one call, `rng.standard_normal(512)` in float64, reshaped to (64, 8) in the model's latent order, and nothing else is
  drawn from `rng`.
- For t ≥ 0, let z′(t) = SimNorm(y_τ + t·η).
- The perturbation is **δ = z′(t\*) − z_τ**, where t\* is chosen so that ‖δ‖₂ equals the target size s. It is found by
  bisection to 10⁻⁹ relative accuracy.
- So z_τ + δ is exactly a SimNorm output: every entry is positive and every group sums to 1. δ is the exact
  displacement, not a linear approximation of one. So the error of §2 contains no error from linearising SimNorm at
  the input.

**Scale, fixed by SimNorm geometry alone.**
- Each of the 64 groups of 8 lies on a simplex whose vertices are √2 apart. The target size is **s = ε · √2**, in the
  Euclidean norm of the full 512 coordinates.
- **Primary scale: ε = 0.1**, so s = 0.1·√2 = 0.14142135623730953.
- **Reported scales: ε ∈ {0.01, 0.1, 1}.** Only ε = 0.1 is verdict-bearing.
- s uses **no checkpoint quantity and no rollout statistic**. It is identical for every model, task, anchor and
  direction. The protected perturbation magnitude is independent of each checkpoint's own rollout statistics (§13, O2).
- m₁, the model's median one-step consistency error ‖f(z_t, a_t) − enc(o_{t+1})‖₂ over its 10 episodes, is **reported
  as a descriptive only**, together with s/m₁. It plays no part in any perturbation or verdict.

*Why this scale.* It is a fixed fraction of the simplex vertex distance, the natural unit of the latent geometry. The value
0.1 matches the order of the tolerance in §2, and was set from that geometry before any trained-model data existed (§9). A
rejection at ε = 0.1 says that the lens fails for a displacement of one tenth of a vertex distance. Whether that is
smaller or larger than a given model's own one-step error is reported through s/m₁, and is not assumed.

**Count and seeding.**
- There are **D = 8** directions per anchor, so 2,000 anchor–direction pairs per model.
- **The direction-generation procedure and its seeds are frozen here, before any protected execution.** η is drawn from
  `numpy.random.Generator(PCG64(seed))`. The seed is the first 8 bytes, big-endian, of
  SHA-256(`"{task}|{seed}|{episode}|{tau}|{j}"`), where j is the direction index. The same η is used at every ε and
  for both linearisations.
  - Fields are decimal integers with no padding, `task` is the task name as in §4, `seed` ∈ {1, 2, 3}, `episode` ∈ 0…9,
    `tau` ∈ {128, …, 368} and j ∈ 0…7.
  - **Test vector** (computed from the string alone; no checkpoint involved, NumPy 2.4.4): the string
    `walker-stand|1|0|128|0` gives SHA-256 `79fb6b1a9104edd5c123a8b9f5460943b05fe5b7d4000b4e259b27e1ae9c0568`, seed
    8789736859589995989, and the first three draws 1.9149714590152809, 0.9242923568957517, −1.0555831182448194. Synthetic
    case S3 checks this vector.
  - The bisection runs on t ∈ [0, 100]; a pair whose target is not reached by t = 100 is unresolved.

**Unresolved pairs.** An unresolved pair counts as e = +∞, a failure. The count of unresolved pairs is reported.
- If no t ≤ 100 reaches the target size, the pair is unresolved at **every** h.
- If ‖Δ_h‖₂ < 10⁻¹⁰ · ‖δ‖₂, the pair is unresolved at **that** h only.

### 3.5 Horizon grid

**h ∈ {1, 2, 4, 8, 16, 32, 64, 128}.** It includes:
- the H2.3 window (16);
- the protocol-1/2 Gramian window (64);
- the protocol-1 robustness horizon (128), reported only (§8).

## 4. Checkpoints and the untouched boundary

**Fresh evidence boundary.** Every checkpoint used here satisfies all three conditions:
1. it was never loaded by protocol 1 or 2, in any form;
2. it has had no forward pass for any purpose;
3. its state-dict keys have never been inspected.

These are the five tasks of the closeout note's §9:
- `walker-stand`, `cheetah-jump`, `reacher-easy`, `finger-turn-easy` and `hopper-hop-backwards`;
- seeds 1, 2 and 3 each, giving **15 checkpoints**;
- each file's SHA-256 is pinned in `config/prereg.yaml`. The digests are copied from the hashed-but-unused alternate
  pool of `hankel-hierarchy/prereg-2/config/prereg.yaml`.

`cheetah-jump` and `hopper-hop-backwards` are TD-MPC2's custom tasks, defined in `tdmpc2/envs/tasks/` at the pinned
commit.

**Excluded, and the reason for each:**
- the 35 checkpoints loaded by protocols 1–2 (18 and 17, counting the excluded `humanoid-walk-3`) and all their controls;
- `humanoid-stand` 1–3, whose state-dict keys were inspected in the protocol-2 layout screen;
- every other checkpoint, because the list above is the whole pool;
- the protocol-1 discovery set, which is not used and is not run.

**Selection.** There is none. All 15 files are used, and the list is final at the freeze.

**No layout screen.** A key-only screen reads the file, and it is what excluded `humanoid-stand` above. So these files
are **not** screened before the freeze.
- A file that fails its SHA-256 check or fails to load in the pinned code after the freeze is **excluded, never
  converted and never substituted**.
- A task that loses one seed is judged **2-of-2**.
- A task that loses two or more seeds **cannot support** any hypothesis, and counts as not supporting.
- Every exclusion is recorded with its load error.

**Behaviour.** Each model's mean episode return is reported against TD-MPC2's published final return. There is no
return gate: validity of a model's linearisation does not depend on how well it performs.

## 5. Statistics reported per model

For each model, each L ∈ {nom, real}, each ε in §3.4 and each h in §3.5:
- the median e_h, the verdict statistic at ε = 0.1;
- the 90th percentile of e_h;
- the median angle between Δ_h and P_hδ;
- the fraction of pairs where ‖P_hδ‖₂ exceeds the diameter of the latent manifold, √128 ≈ 11.31. At such points the
  linear prediction is necessarily wrong, because the true response is bounded;
- the fraction of unresolved pairs;
- m₁ and s/m₁ (descriptive only, §3.4);
- the median ‖δ‖₂ / (0.05 · ‖B̃_τ‖₂), which relates the perturbation to the planner's action-noise floor
  (`min_std` = 0.05 at the pinned commit);
- h\*_nom and h\*_real.

## 6. Decision rules

**Model criterion.** A model meets Q1.1 (Q1.2) if h\*_nom ≥ 64 (h\*_real ≥ 64) at ε = 0.1 and tolerance 0.10.

**Task.** A task supports a hypothesis if at least 2 of its 3 seeds meet the model criterion (2 of 2 after one
exclusion; §4).

**Hypothesis.** A hypothesis is **SUPPORTED** if at least **4 of the 5** tasks support it. Otherwise it is
**REJECTED**. Protocols 1–2 required 4 of 6. Requiring 4 of 5 makes the positive claim harder to reach, not easier.

**Secondary verdicts** apply the same counting with h\* ≥ 16. They are labelled secondary, and never replace a
primary verdict (§1).

**Invariants:**
- Every statistic is computed and reported whatever the verdict.
- No tolerance, scale, horizon, direction count, task, seed or rule is changed after the freeze.
- A different scale or tolerance would be a new question with its own preregistration.
- The verdict comes mechanically from the verdict script applied to `config/prereg.yaml`.

### 6.1 Failure conditions, stated in advance

- **Q1.1 is REJECTED** if fewer than 4 of 5 tasks have enough seeds whose tangent-linear prediction stays within 10%
  of the nonlinear response at every grid horizon up to 64. Rejection means: at a perturbation of one tenth of the
  simplex vertex distance, first-order propagation along the model's own trajectory does not describe the network over the Gramian window.
- **Q1.2 is REJECTED** under the same counting for L-real. Rejection means: the lens that protocols 1–2 used does not
  describe the network's response over their window, at this fixed scale.

## 7. What each outcome licenses

| Q1.1 | Q1.2 | Reading, for these five tasks and this model class |
|---|---|---|
| supported | supported | The 64-step first-order lens describes the network at this scale. The protocol-1/2 negatives are more likely to be statements about the networks, not only about the lens (an inference; their models are not re-tested). |
| supported | rejected | Propagation is locally linear, but real-state linearisation points are the wrong ones: the lens fails through state mismatch, not nonlinearity. |
| rejected | rejected | The first-order lens fails over 64 steps at this scale. The protocol-1/2 negatives should be read as statements about the lens. Their verdicts are unchanged. |
| rejected | supported | Not expected. It would mean real-state Jacobians track the nonlinear response better than the model's own tangent model. It is reported as found, and flagged for investigation. |

The secondary verdicts at h = 16 are read the same way for the H2.3 window. **No row licenses any claim about Hankel
structure or planning.**

## 8. Exploratory and sealed

**Exploratory (reported and labelled; never verdict-bearing):**
- scales ε = 0.01 and ε = 1;
- tolerances 0.05, 0.25 and 0.5;
- h = 128;
- the 90th-percentile and angle statistics;
- the diameter-overshoot fraction;
- relating h\* to growth ‖Φ(τ+64, τ)‖₂ or to m₁ (descriptive only; closeout note Q2 would need its own preregistration);
- relating h\* to m₁ or to s/m₁ (descriptive only);
- the leading right singular vector of P_64 as an extra, non-random perturbation direction;
- an output-channel check, predicting Q̄(F_h(z_τ + δ)) − Q̄(F_h(z_τ)) with C_{τ+h} P_h δ;
- an action-channel check: perturb a_τ by δa of norm 0.05, the planner's `min_std`, and predict
  F_h(z_τ; a_τ + δa, …) − F_h(z_τ; a_τ, …) with Φ(τ+h, τ+1) B_τ δa. This is the controllability-factor column of the
  old protocols.

**Not part of this protocol, and not run under it:**
- anything on the 35 protocol-1/2 checkpoints or their discovery set;
- R-full or R-dyn controls. Validity is a property of each model's own map, so no architecture comparison is needed;
- closed-loop rollouts that re-plan on imagined states;
- any Hankel, Gramian-spectrum or planning analysis. Protocol-1 experiments D–G stay sealed.

## 9. Before and after the freeze

**Done before this protocol was written, and permitted:**
- reading the pinned TD-MPC2 code;
- copying the 15 SHA-256 digests from the protocol-2 config;
- writing this document.

**Not done, and not permitted before the freeze:**
- any `torch.load` of the 15 files, including a key-only load;
- any forward pass of any trained checkpoint, old or new;
- any run on the discovery set;
- using any protocol-1/2 output to set a Q1 parameter. None was used: ε, the tolerance, the grid and D were set from
  the definitions above, without consulting any retained statistic. That includes the retained one-step errors. The scale
  s = ε·√2 uses no checkpoint quantity at all (§13, O2).

**After the freeze, in this order:**
1. The freeze SHA is sent to the reviewer. **A SHA receipt is not authorisation** (protocol-2 P1).
2. The analysis code is written and validated only on the synthetic cases S1–S4 (§10).
3. **Nothing result-bearing runs until the reviewer gives an explicit go-ahead.** The go-ahead covers S5 and the
   confirmatory run.
4. The result commit sits on top of the freeze, with the frozen files byte-identical.

## 10. Synthetic validation

No trained checkpoint is involved.

- **S1, a linear map.** With f(z, a) = Az + Ba, e_h ≤ 10⁻¹² at every h for both linearisations.
- **S2, a smooth nonlinear map with a SimNorm output layer.**
  - For small ε, median e_h scales linearly in ε: the log–log slope over ε ∈ {10⁻⁴, 10⁻³} lies in [0.9, 1.1].
  - h\* is non-increasing in ε.
- **S3, the on-manifold construction.**
  - z_τ + δ has group sums equal to 1 within 10⁻¹², and every entry positive.
  - ‖δ‖₂ is within 10⁻⁹ relative of the target.
  - Determinism: the same seed gives the same δ, and the §3.4 test vector is reproduced.
- **S4, JVP against explicit products.** The forward-mode JVP product agrees with explicit float64 Jacobian products
  to within 10⁻¹⁰ relative, on random small networks.
- **S5, a plumbing smoke test on the real code path.** It uses a random-initialisation network (torch seed 999), and
  no checkpoint is loaded. It checks shapes, dtypes, determinism, and that L-nom = L-real at k = 0. It is run **only
  after authorisation**, as the first step of the authorised run.

## 11. Artifacts

The result commit contains:
- the code and the synthetic tests;
- `results/` with per-model statistics, per-pair errors, verdicts and a receipt per model (checkpoint SHA-256, the 10
  episode returns, m₁, wall time);
- `ENVIRONMENT.txt`;
- the figures;
- `RESULTS.md`, with a confirmatory section and a separate exploratory section.

**The run prints the repository HEAD SHA into its outputs.** Neither earlier run recorded its code identity this way
(closeout note, limitation 5).

## 12. Deviations

Any departure after the freeze is logged in `DEVIATIONS.md`, with its date, its reason and what had been seen when it
was written. A deviation never changes a verdict, and the verdict under this protocol as written is always reported.

---

## 13. Review decisions

Recorded on 4 Oct 2026 from the reviewer's written decisions, before the freeze and before any trained-checkpoint output
existed. The reviewer stated that freezing the protocol is **not** authorisation to execute it. **No Q1 evaluation,
forward pass, layout screen or discovery run is authorised by the freeze.**

**Fixed rules**
- **R1. No screening: accepted.** No key-only load, layout check or forward pass of any of the 15 files before the freeze
  (§4, §9).
- **R2. Exclusion without replacement: accepted.** A file that fails its SHA-256 check or fails to load after the freeze
  is excluded, never converted and never substituted. If attrition makes the prespecified support rule impossible, the
  result is **inconclusive**, not repaired by substitution (§4). A task that loses one seed is judged 2-of-2; a task that
  loses two cannot support a hypothesis.
- **R3. Threshold provenance: accepted.** The rejection threshold is set from theory and stated tolerance only. No
  untouched-checkpoint outcome may enter threshold selection (§9).

**Choices**

| # | Choice | Decision |
|---|---|---|
| O1 | Error tolerance | **Accepted: median e_h ≤ 0.10.** The 5.7° interpretation (arcsin 0.1) is documented rationale, not an empirical guarantee (§2). |
| O2 | Perturbation scale | **Changed by the reviewer: SimNorm-geometry-only fixed scale, not the checkpoint-derived m₁ scale.** The protected perturbation magnitude is independent of per-checkpoint rollout statistics. Implemented as s = ε·√2 with ε = 0.1 (§3.4). The reviewer specified the principle; the numerical value ε = 0.1 is the author's, set from the geometry without trained-model data, and is part of what is frozen. |
| O3 | Counting rule | **Accepted: 4 of 5 tasks, each with at least 2 of 3 seeds, exactly as prespecified** (§6). |
| O4 | Validity window | **Accepted: h\* ≥ 64 primary, ≥ 16 secondary** (§1, §6). |
| O5 | Directions per anchor | **Accepted: D = 8.** The direction-generation procedure and random seed rule are frozen before any protected execution (§3.4). |

The October 6 self-contained write-up handoff is unchanged and independent of Q1 execution.
