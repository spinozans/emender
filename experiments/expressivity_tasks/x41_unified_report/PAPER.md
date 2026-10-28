# Emender Memory: architectural emendation, a measured capacity boundary, and the cost of ordered state

Draft X41 · 28 October 2026 · architecture-first receipt-backed synthesis

## Abstract

Emender Memory is the operator's name for a budgeted slot memory whose management follows architectural rules. An age-1 correction is an *emendation*: a revision of recently stored content. The architecture combines content-addressed reads with APPEND, CORRECT, POP, DECAY and MERGE; learned representations and gates remain part of the model. “Zero learned management” describes the routing program, not the absence of learned influences on its inputs. [C04](CLAIMS.md#c04) [C07](CLAIMS.md#c07)

A CMA-ES-hardened fixed-state pairing at 1.3B scale supplies the motivation: nonlinearity-in-time bought no demonstrated loss advantage under the matched Pile recipe. Emender Memory instead changes the write program and makes allocation and exhaustion explicit. The old pairing and its methodology are inherited as the scale-up baseline. [C31](CLAIMS.md#c31) [C34](CLAIMS.md#c34)

On the registered balanced a^k b^k pools, a 736-slot model scores 0.9805 and 0.9844 at 2x and 4x depth, then 0.5000 at 9x; a separately trained 2944-slot arm reaches 0.9473 at 9x. The controls approach chance at 4x/9x, but retain partial discrimination at 2x. Pressure perfectly flags incorrect **accept-class words in the B0 sample**, while the larger arm reveals both early warning and pressure-free errors. This is a measured boundary for particular trained systems, not a universal failure detector or an unbounded-generalization theorem. [C08](CLAIMS.md#c08) [C09](CLAIMS.md#c09) [C10](CLAIMS.md#c10) [C11](CLAIMS.md#c11) [C12](CLAIMS.md#c12) [C13](CLAIMS.md#c13)

A fused implementation and subsequent LM engineering substantially reduce cost, without closing all performance gates. TinyStories pilots establish a live memory lane, an honest zero-shot counting null, and architectural interventions that change occupancy. The semantic-key plus match-wire arm has mean finite-window occupancy slope 0.108 and a 1.15% quality cost against its registered control. This supports a conceptual-cache hypothesis; it does not establish that occupancy measures retained information. [C15](CLAIMS.md#c15) [C16](CLAIMS.md#c16) [C20](CLAIMS.md#c20) [C24](CLAIMS.md#c24) [C26](CLAIMS.md#c26)

## 1. The mechanism: emendation as a write program

We use *emendare*, “to remove faults,” as the operator-provided naming rationale. Here emendation has a concrete referent: the write program can revise the live slot born one token earlier. “Tape machine,” used in historical receipts, risks confusion with Neural Turing Machines and Delétang et al.'s memory tape. Emender Memory is a bounded, tunable slot store, not a claim to a Turing tape, an unbounded stack, or exclusively LIFO access. Content-addressed READ and POP, age-based CORRECT, and budget-triggered MERGE distinguish its operation from a stack pointer alone. [C04](CLAIMS.md#c04) [C07](CLAIMS.md#c07)

The intended contribution is a linked set of observations: explicit management rules, a measured capacity-dependent failure boundary, an instrument exposing exhaustion, an implementation cost account, and pilot design science about using memory. The direction of the structured-memory result is prior art. Delétang et al. already studied structured-memory generalization, and Merrill, Petty and Sabharwal give restrictions on the SSM classes they analyze. We neither re-prove those results nor infer from them that every bounded recurrence, nonlinear state model, or delta-rule variant cannot track state. [C01](CLAIMS.md#c01) [C05](CLAIMS.md#c05) [C06](CLAIMS.md#c06)

The store contains keys, values, strengths and birth information. READ precedes POP, post-POP death, WRITE, rounded decay, age advance and decay death. CORRECT selects the lowest-index live age-1 row; correction does not renew its birth. F12/F12g/F12h are semantic engineering amendments, not interchangeable versions of a content-argmax write. Even ZERO_RESIDUAL can request an append. Repeated content therefore does not automatically mean no allocation. [C07](CLAIMS.md#c07)

This ordering makes the instrument interpretable but also costly. An append creates the next token's potential correction target. The target's current payload affects residuals and gate computation. A detached routing decision has zero derivative through its discrete choice; it does **not** become independent of earlier state. Pressure records cap/request behavior, and raw cumulative pressure must be distinguished from a trailing-window saturation event. [C07](CLAIMS.md#c07) [C13](CLAIMS.md#c13) [C17](CLAIMS.md#c17)

At the slot budget, MERGE folds an append request into an existing live payload instead of allocating a new slot. This is graceful degradation of a finite store, not guaranteed semantic preservation. The pressure register makes requests against exhausted capacity observable. The budget is a tunable architectural resource; content-driven occupancy and explicit management are the proposed design point. [C07](CLAIMS.md#c07) [C13](CLAIMS.md#c13)

The X32 survey suggests the following map. Its “empty quadrant” is a bounded literature-search observation, not a priority theorem over all published systems. The useful axis is **content-driven occupancy at a tunable budget**, rather than unbounded growth. Fixed architectural management also needs a definition: learned features, write-enable inputs and content affect the execution of fixed rules. [C05](CLAIMS.md#c05) [C07](CLAIMS.md#c07)

| State organization | Learned management/controller | Architectural update/management rules |
|---|---|---|
| Fixed-size compressed state | Learned routing and test-time-memory variants require case-specific classification | Fixed-state recurrence families; strong updates do not alone imply a growing store |
| Re-addressable slots, stack, or variable occupancy | NTM/DNC, stack controllers, NRAM; DeltaStack has learned soft actions | Emender's proposed design point: architectural slot routing, tunable occupancy, merge and pressure |

This map summarizes X32, including its qualifications about hybrid systems; it does not classify the entirety of a model as “learned” or “unlearned.” [C05](CLAIMS.md#c05)

DeltaStack is the closest comparison. The X32 full-PDF addendum confirms learned soft action weights and a continuous soft pointer. The code uses a bounded power-of-two stack and clamps the pointer. Crucially, the paper **does** have a stack-size ablation and an S≥D sufficiency condition; claiming “no capacity analysis” would be false. The narrower reported gap is failure-boundary characterization when capacity is inadequate, with saturation telemetry and a degradation policy. Its parallel soft-pointer recurrence is also a real efficiency advantage, not an equivalent scheduling of hard age-1 correction. Bibliographic entries and the detailed comparison are in [RELATED.md](RELATED.md). [C06](CLAIMS.md#c06) [C17](CLAIMS.md#c17)

## 2. Motivation measured at full scale: the 1.3B null

The obvious fix—put nonlinearity into fixed recurrent state and tune the system seriously—was run at scale and bought no demonstrated language-model loss advantage. The retired manuscript's E97 is a fixed matrix-state nonlinear-in-time recurrence; it is **not** the budgeted slot memory introduced here. Its CMA-ES-hardened E97-MLP/GDN2-MLP pairing is now motivation and a baseline row. The [retirement reconciliation](../../../paper/RECONCILIATION.md) records which old assertions stand and which expectations retire. Within the measured fixed-state families, compute explains the shared loss trend better than the choice of state nonlinearity. “FLOPs determine loss” is an empirical summary of these receipts, not a universal scaling law or a significance test. The missing capability motivating Emender Memory is architectural revision of stored content and allocation, rather than state nonlinearity alone. [C31](CLAIMS.md#c31) [C32](CLAIMS.md#c32) [C33](CLAIMS.md#c33)

The long pairing uses the same Pile stream, p50k_base tokenizer, 2048-token chunks, bf16 schedule-free AdamW, per-rank batch 4 and eight local DiLoCo islands, with averaging every 250 steps, outer learning rate 1 and beta 0. Each family keeps its separately searched geometry and learning rate. At the last common logged cutoff, **150,793,420,800 tokens**, the machine receipt aligns **92,037 records**, with zero single-arm records. Loss deltas below are GDN2 minus E97, in nats/token. [C31](CLAIMS.md#c31) [C34](CLAIMS.md#c34)

| Matched-token statistic | E97-MLP | GDN2-MLP | Delta |
|---|---:|---:|---:|
| Raw cutoff | 2.477000 | 2.432400 | −.044600 |
| Trailing 80 aligned observations | 2.437045 | 2.426705 | −.010340 |
| Last 100 optimizer-step interval | 2.428350 | 2.442300 | +.013950 |
| Last 1000 optimizer-step interval | 2.435820 | 2.427318 | −.008503 |

These are training-log observations, not held-out measurements: the last two intervals contain only 4 and 40 logged observations, respectively. The sign reversal across windows supports the inherited loss-parity null, not a quantified equivalence margin or independently estimated noise distribution. Pairwise held-out tokenizer-invariant bpb and downstream validation remain open. [C31](CLAIMS.md#c31)

A separate short-budget search-scale receipt puts **five** CMA-best families in a **.0887-bpb** non-averaged held-out band, **2.0126–2.1013**, on the same **131,072-token** Pile-tail slice. The historical “0.088” label is approximate; subtraction of the recorded endpoints yields .0887. E97-MLP is nominally ahead of GDN2-MLP in both search mean and non-averaged held-out bpb, so the null must not be rewritten as a GDN2 victory. Architecture-dependent schedule-free averaging changes the ranking; single-seed short runs do not estimate significance. This is a different experiment from the long pairing. [C32](CLAIMS.md#c32)

The predecessor FLOPs-per-bit package supplies a third, distinct view: CMA-tuned E1, E88, FLA-GDN and Mamba2 at roughly 480M parameters share a broad compute-economy trend. Its FLOPs are estimated as 6 × parameters × processed tokens. Convergence concerns the rate of loss reduction over the measured trajectory, not identical endpoints, a common asymptote, or universal family equivalence; tight thresholds separate the curves and the E1 arm does not reach the tightest target. It has one seed per family and no M2RNN arm. The original JSON search receipts are snapshotted alongside the inherited trajectory and threshold CSV pointers. [C33](CLAIMS.md#c33)

The X20–X27 autopsy supplies no clean impossibility proof: theory suggested a separator, X20 found **NO LICENSED CONSTRUCTION**, X21/X23 retain censored horizons, and X22 distinguishes growing precision from fixed budgets. X24's softplus disposition remains a qualified terminal outcome; X25's recipe-controlled tanh learns ID without robust OOD. Native-fidelity reconciliation in X26c was **MIXED / ID-only**; X27 sign-reversed the “too big” prediction, with small tanh failing where the larger arm learned ID and small softplus censored for nonfinite behavior. Capacity was required for the observed learning transitions, not sufficient for the proposed separation. The full-scale language-model null reinforces the motivation to revise the memory architecture; it does not turn that mixed expressivity record into a theorem. Existing Lean work retains its own assumptions and is not rebuilt here. [C01](CLAIMS.md#c01) [C02](CLAIMS.md#c02) [C03](CLAIMS.md#c03) [C29](CLAIMS.md#c29) [C30](CLAIMS.md#c30)

## 3. The certified boundary

The following numbers are re-read from the selected-checkpoint `final_eval.json` files. They are balanced accuracies on the registered pools, not a new rerun. “Matched controls” refers to the X28E protocol, not equal runtime, equal state bytes, or a newly certified state-accounting theorem. [C08](CLAIMS.md#c08) [C09](CLAIMS.md#c09) [C10](CLAIMS.md#c10) [C11](CLAIMS.md#c11)

| Arm | ID | 2x | 4x | 9x |
|---|---:|---:|---:|---:|
| Emender B0, 736 slots | .994141 | .980469 | .984375 | .500000 |
| Emender B0x4, 2944 slots | .990234 | .980469 | .970703 | .947266 |
| GDN2 | .990234 | .564453 | .498047 | .498047 |
| E97 | .998047 | .541016 | .500000 | .500000 |

Receipts: [C08](CLAIMS.md#c08)–[C11](CLAIMS.md#c11). In particular, “both controls collapse to .50 at 2x” is an inaccurate condensation: their 2x scores are .5645 and .5410.

Fine sweeps place B0's first raw cap event and accept-collapse at k=736; k=720 holds. The learned schedule alternates append and age-1 correction through both a and b phases, with no POP in the traced words. This makes exhaustion a physical allocation boundary for this learned strategy. The larger-budget arm is separately trained, so its 9x result does not isolate a test-time budget intervention on identical weights. Moreover, its non-capacity hole means that more slots do not guarantee pointwise dominance at every depth. [C12](CLAIMS.md#c12) [C13](CLAIMS.md#c13)

### Pressure: specific budget exhaustion with lead time

B0 has 6400 traced words, including 3200 accept-class words. Of the accepts, 1408 are incorrect and pressured; 1792 are correct and unpressured. Thus PPV and sensitivity are both 1.000 **for this accept-class contingency**. Calling this perfection over all error types or all 6400 words would silently change the denominator. The larger arm's all-accept PPV is .333 and sensitivity .176: pressure is not a general error oracle. [C12](CLAIMS.md#c12)

At B0x4, raw pressure begins at k=2944 and accept-collapse at k=3040: a 96-depth lead. Windowed saturation begins later and leads collapse by 64 depth. First saturation positions are 1524 and 5940, following 2B+52 in these runs. The signal occurs before EOS but **inside the b phase**, contrary to the pre-registered before-b expectation. Cumulative pressure never reaches the registered .9 saturation threshold. The directed hypothesis remains **MIXED**. [C13](CLAIMS.md#c13)

The honest nulls matter. B0x4 rejects balanced words in the pressure-free hole spanning at least k=560–816, then recovers at later sampled depths. It also saturates at some depths while accuracy remains .832, above the registered collapse criterion. The prediction that the larger arm holds everywhere B0 collapses is refuted. Pressure exposes budget exhaustion and can precede one failure mode; it is specific to resource stress and insensitive to some non-capacity failures. [C13](CLAIMS.md#c13)

## 4. The cost account: ordered state and sequentiality

The X30 v1 maintenance result is approximately 423x on CPU and 677x on GPU at S=11776, comparing an inert full-store maintenance workload. These numbers exclude reads, content writes, snapshots and synchronization; they are not whole-cell speedups. X31 v2 reports 109-test numerical parity at the registered tolerance and a 93.109x fresh-forward speedup over its matched X30 reference at S11776. The cold H16/H1 scaling gate still fails. The parent adjudication calls this 7/8 gates plus a separate steady-state pass on the eighth, with initialization removed; it does not erase the cold-start failure. [C14](CLAIMS.md#c14) [C15](CLAIMS.md#c15)

The LM-scale program moves from X34's roughly 23.4x historical smoke overhead to X36 part 6's 1.541x B64/T512 whole-LM wall ratio. These are different workloads and stages, not a single controlled speedup factor. Fresh capacity doubling passes its registered limits. Continued capacity, long context, inclusive launches and continued host synchronization remain open failures; the narrow custom-kernel count is not the inclusive launch count. Part 7 also rejects a proposed dense-gradient-scatter diagnosis: that scatter executes zero times in the measured capacity workload. [C16](CLAIMS.md#c16) [C20](CLAIMS.md#c20)

Near-budget arithmetic is a useful engineering account, not a performance lower-bound theorem. Part 6's near-budget step ratios are about 2.050 and 2.531 on successive doublings, but useful live occupancy grows as well. Logical access counts are not measured DRAM traffic. The part-7 arithmetic identifies remaining work and required savings without establishing an irreducible hardware floor. The later parent adjudication labels near-budget scaling a FLOOR and a registration-regime mismatch. We retain that disposition as an attributed interpretation: doubling live occupancy explains approximately doubled logical work, but alone does not prove a wall-time lower bound or explain every measured ratio above two. Part 7 also changes fixed-live outcomes (including a live1024 step-doubling failure), so its detailed machine ledger takes precedence over condensed claims that all medium-live controls pass. Its short/long whole-LM ratios are 1.551x/2.621x; the 1.541x headline explicitly names part 6. The paper therefore retains the near-budget performance floor as a qualified arithmetic argument, not a proved hardware limit. [C16](CLAIMS.md#c16)

X35/v3 refutes the unconditional entry-only two-pass routing proposal. Two tokens evaluated against empty entry state would both append; sequentially the second can correct the first token's slot. Even append-only traces can retain payload and causal-read dependencies. The trained ladder has a .5 event-token fraction, so sparse exceptions do not remove this case. Interpretation: the ordered content dependence participates in the useful mechanism and resists this parallelization. This is not a theorem that all expressive memory must be serial; DeltaStack buys parallelism through a different soft-pointer recurrence. [C17](CLAIMS.md#c17)

Our interpretation is that emendation, bounded allocation and explicit exhaustion jointly create a useful research instrument: failure is measurable in state coordinates rather than accuracy alone. The larger arm's hole prevents a capacity-only theory of all errors. The serial counterexamples prevent an unconditional parallel schedule for the present recurrence. The semantic-wire intervention shows that learning content is not enough to change allocation when the necessary read-to-gate connection is absent. Each conclusion is local to the receipted mechanisms. [C13](CLAIMS.md#c13) [C17](CLAIMS.md#c17) [C25](CLAIMS.md#c25)

The proposed trade-off is therefore architectural: a hard ordered correction program offers explicit operations and interpretable exhaustion, while soft-pointer designs can support more parallel execution. Neither the measured performance failures nor the local dependency obstruction proves a universal expressivity/efficiency theorem. Boundary characterization, rather than the existence of structured memory or a tunable store-size knob, is the defensible distinction from the surveyed prior art. [C06](CLAIMS.md#c06) [C16](CLAIMS.md#c16) [C17](CLAIMS.md#c17)

## 5. LM pilots: lane activity and architectural design science

X34 total parameters are 21,913,872 versus 21,914,119, a .001127% discrepancy. Trainable counts differ, and actual wall cost is not compute matched. Both B8 arms reach 100,003,840 tokens. The final training minibatch loss favors Emender, 1.854418 versus 1.867671, but held-out NLL is 1.799193 versus baseline 1.798722 and PPL is 6.044765 versus 6.041918. The train-loss edge must not be retold as a held-out language-model win. [C18](CLAIMS.md#c18) [C19](CLAIMS.md#c19)

The lane is active on real text: occupancy grows from 256/head at T512 to 2048/head at T4096, with slope 1.0 and the same append/correct alternation diagnosis. That resembles a token transcript rather than evidence of selective conceptual storage. Both arms score .5 on the registered zero-shot depth probes. Few paired corruptions and tokenizer-dependent depth further limit inference; real-text lane activity is not transfer of the certified counting capability. [C20](CLAIMS.md#c20)

A consequential confound was caught: historical X34 uses B8, whereas the intervention family uses B64 and many fewer optimizer steps. The new B64 control supplies matched exposure at 100,007,936 tokens, PPL 6.856418. X37 b/c and X38's selected full arm pass the intended quality band against that control. The existing comparability JSON lacks terminal d; re-reading d's completed receipt gives 6.682684, ratio .974661 to control. This corrects a historical “all worse” interpretation without making the B64 family beat the stronger absolute B8 result. [C21](CLAIMS.md#c21) [C22](CLAIMS.md#c22)

### Differentiation is architectural in the measured intervention

Seeding interventions b and c end with identical head peaks in 16/16 stories; the hierarchy arm d has 0/16 identical peaks and the best PPL among these interventions. This is an occupancy result, not proof of identical reads or an exact weight-permutation fixed point. The source audit finds unequal head rows already at initialization, and learned W_in distances diverge even when occupancy agrees. Latent functional diversity may therefore escape the peak statistic. [C22](CLAIMS.md#c22)

X38 moves mean read cosine from .000750 to −.313177, without monotonic decrease; physical-slot JS ends lower, .028949→.019245. It provides forced anticorrelation, not complementary slots. Its quality cost is comparator dependent: 11.4% worse than historical B8 a, about .61% worse than B64 c, and better than the protocol-matched B64 control. The hierarchy is also costly: about 7.876x c's median CUDA-event step time. Architectural policy is a demonstrated lever for occupancy diversity here, not a proof that learned diversity is impossible or free. [C23](CLAIMS.md#c23) [C21](CLAIMS.md#c21) [C22](CLAIMS.md#c22)

### Usage and the conceptual-cache hypothesis

All rows below use X39's registered B64 exposure and fixed-window quality comparator. Slopes fit log peak occupancy against log T at 512/1024/2048/4096; the mean is an arithmetic summary across four heads, not a confidence interval. [C24](CLAIMS.md#c24)

| Lever | Mean slope | PPL / B64 control | What the receipt supports |
|---|---:|---:|---|
| Decay horizon 128 | ≈0 | 1.009752 | Built-in expiration regulator, not semantic selection |
| Gate cost .01 | .946827 | 1.007983 | Lower allocation constant; fails strong-compression screen |
| Poverty S128 | ≈0 | 1.005479 | Capacity cap |
| Same poverty weights, S2944 | 1.000000 | 1.002633 | Thrift does not transfer to rich deployment |
| Shared semantic keys + match wire + cost | .107614 | 1.011518 | Strong occupancy screen passes with registered quality |

All five measurements: [C24](CLAIMS.md#c24). Gate cost's near-one slope does **not** prove that a loss knob can never change an asymptotic exponent. Likewise the semantic slope and +1.15% PPL support an occupancy hypothesis, not the stronger claim that occupancy has been shown to scale with information rather than tokens.

The mechanism ledger explains why some appealing interventions fail. Append amplitude changes stored values after requests have been selected; its differentiable path does not supply an allocation-count gradient. Semantic keys improve retrieval but the ISA does not automatically deduplicate repeated concepts. In the CPU alias construction, semantic keys plus a read-to-gate wire suppress repeat allocation, while either ablation keeps allocating. The optimized circuit repeats this distinction at the corrected READ scale. Necessity and sufficiency apply to those constructions, not every possible model or language distribution. [C25](CLAIMS.md#c25)

The trained semantic arm extends beyond the witness and has targeted production-shape replay parity. Still, no trained-language causal ablation identifies each component's contribution. The quality comparator is the registered control, not a newly trained widened parameter-matched base. Source hashes identify snapshots but do not conclusively identify cached bytecode in the inherited provenance record. Small depth probes do not certify retained information. These limitations accompany the compression result. [C26](CLAIMS.md#c26) [C24](CLAIMS.md#c24)

## 6. Open: scale, replication and the pure model

The planned Pile program and a **1.3B Emender Memory run explicitly INHERIT the retired paper's exact methodology**: per-family CMA-ES selection, the shared Pile/tokenizer/chunk and optimizer recipe, local averaging policy, matched aggregate-token readout, multiple declared loss windows, and immutable checkpoint/configuration/hash releases. This is a proposed comparison, not a submitted or completed run. The old E97-MLP/GDN2-MLP pairing becomes the baseline row, not discarded history. If Emender Memory moves off the convergent-loss curve under this matched methodology, that is the headline to test. Neither the slot-boundary result nor the TinyStories pilot licenses it in advance; recovery of missing original search histories and held-out evaluation of the long pair remain open. [C31](CLAIMS.md#c31) [C32](CLAIMS.md#c32) [C33](CLAIMS.md#c33) [C34](CLAIMS.md#c34)

The LM leg remains **pilot-grade**: one training seed per arm, TinyStories scale, small fixed probes, incomplete trainable-capacity and runtime equivalence, inherited provenance qualifications, and no demonstrated zero-shot structural transfer. Budget-capped flat occupancy can destroy information. Scale-up and general semantic retention remain untested by these receipts. A curriculum probe is proposed to distinguish learnability from absent zero-shot transfer; no positive outcome is assumed. [C18](CLAIMS.md#c18) [C20](CLAIMS.md#c20) [C22](CLAIMS.md#c22) [C26](CLAIMS.md#c26)

The certified append-only and unbounded arms have checkpoint receipts but no terminal final evaluation used here. They remain pending/censored evidence, not negative experimental results. No fresh training, GPU work, Lean verification, Frontier submission or resilient-training validation is part of X41. [C28](CLAIMS.md#c28) [C03](CLAIMS.md#c03)

### X40 slot — EmenderLM pure model: IN FLIGHT

Reserved for the separately registered attention-free EmenderLM program. The registration is implementation/preflight evidence, not a completed model result. Insert outcomes only after endpoint receipts, policy-specific parity and matched probe evidence exist; retain failures if they occur. No X40 number enters this draft. [C27](CLAIMS.md#c27)

### Proposed next evidence

We propose independent seeds and task variants for the capacity boundary; budget interventions on identical checkpoints; state-byte accounting and stronger controls; causal trained-language ablations for the match wire; retention probes alongside occupancy; and performance measurements at continued near-budget state. These are research proposals motivated by the holes and failed gates, not completed experiments. [C05](CLAIMS.md#c05) [C13](CLAIMS.md#c13) [C16](CLAIMS.md#c16) [C26](CLAIMS.md#c26)
