# Emender Memory: architectural emendation, a measured capacity boundary, and the cost of ordered state

Draft X41 · 28 October 2026 · receipt-backed synthesis

## Abstract

Emender Memory is the operator's name for a budgeted slot memory whose management follows architectural rules. An age-1 correction is an *emendation*: a revision of recently stored content. The architecture combines content-addressed reads with APPEND, CORRECT, POP, DECAY and MERGE; learned representations and gates remain part of the model. “Zero learned management” describes the routing program, not the absence of learned influences on its inputs. [C04](CLAIMS.md#c04) [C07](CLAIMS.md#c07)

On the registered balanced a^k b^k pools, a 736-slot model scores 0.9805 and 0.9844 at 2x and 4x depth, then 0.5000 at 9x; a separately trained 2944-slot arm reaches 0.9473 at 9x. The controls approach chance at 4x/9x, but retain partial discrimination at 2x. Pressure perfectly flags incorrect **accept-class words in the B0 sample**, while the larger arm reveals both early warning and pressure-free errors. This is a measured boundary for particular trained systems, not a universal failure detector or an unbounded-generalization theorem. [C08](CLAIMS.md#c08) [C09](CLAIMS.md#c09) [C10](CLAIMS.md#c10) [C11](CLAIMS.md#c11) [C12](CLAIMS.md#c12) [C13](CLAIMS.md#c13)

A fused implementation and subsequent LM engineering substantially reduce cost, without closing all performance gates. TinyStories pilots establish a live memory lane, an honest zero-shot counting null, and architectural interventions that change occupancy. The semantic-key plus match-wire arm has mean finite-window occupancy slope 0.108 and a 1.15% quality cost against its registered control. This supports a conceptual-cache hypothesis; it does not establish that occupancy measures retained information. [C15](CLAIMS.md#c15) [C16](CLAIMS.md#c16) [C20](CLAIMS.md#c20) [C24](CLAIMS.md#c24) [C26](CLAIMS.md#c26)

## 1. Introduction: memory that can emend

We use *emendare*, “to remove faults,” as the operator-provided naming rationale. Here emendation has a concrete referent: the write program can revise the live slot born one token earlier. “Tape machine,” used in historical receipts, risks confusion with Neural Turing Machines and Delétang et al.'s memory tape. Emender Memory is a bounded, tunable slot store, not a claim to a Turing tape, an unbounded stack, or exclusively LIFO access. Content-addressed READ and POP, age-based CORRECT, and budget-triggered MERGE distinguish its operation from a stack pointer alone. [C04](CLAIMS.md#c04) [C07](CLAIMS.md#c07)

The intended contribution is a linked set of observations: explicit management rules, a measured capacity-dependent failure boundary, an instrument exposing exhaustion, an implementation cost account, and pilot design science about using memory. The direction of the structured-memory result is prior art. Delétang et al. already studied structured-memory generalization, and Merrill, Petty and Sabharwal give restrictions on the SSM classes they analyze. We neither re-prove those results nor infer from them that every bounded recurrence, nonlinear state model, or delta-rule variant cannot track state. [C01](CLAIMS.md#c01) [C05](CLAIMS.md#c05) [C06](CLAIMS.md#c06)

## 2. Background: the quadrant and the failed shortcuts

The X32 survey suggests the following map. Its “empty quadrant” is a bounded literature-search observation, not a priority theorem over all published systems. The useful axis is **content-driven occupancy at a tunable budget**, rather than unbounded growth. Fixed architectural management also needs a definition: learned features, write-enable inputs and content affect the execution of fixed rules. [C05](CLAIMS.md#c05) [C07](CLAIMS.md#c07)

| State organization | Learned management/controller | Architectural update/management rules |
|---|---|---|
| Fixed-size compressed state | Learned routing and test-time-memory variants require case-specific classification | Fixed-state recurrence families; strong updates do not alone imply a growing store |
| Re-addressable slots, stack, or variable occupancy | NTM/DNC, stack controllers, NRAM; DeltaStack has learned soft actions | Emender's proposed design point: architectural slot routing, tunable occupancy, merge and pressure |

This map summarizes X32, including its qualifications about hybrid systems; it does not classify the entirety of a model as “learned” or “unlearned.” [C05](CLAIMS.md#c05)

The earlier X20–X27 arc is not a clean impossibility proof. X20 terminated with **NO LICENSED CONSTRUCTION** for a wrapped E97/GDN expressivity separator. X21 horizon measurements and X23 separator runs retain censoring where crossings were unobserved. X26c's native raw-default reconciliation yielded **MIXED / ID-only**, not a binary success/failure slogan. X27's size grid reversed the “too big” prediction: small tanh failed to learn where the larger comparator learned ID; the small softplus arm was censored for nonfinite behavior. Capacity was required for the observed learning transition, not proved sufficient for generalization. Existing Lean material under `formal/lean/ElmanProofs` is available as formal work with its own assumptions; this report does not rebuild it. [C01](CLAIMS.md#c01) [C02](CLAIMS.md#c02) [C03](CLAIMS.md#c03) [C29](CLAIMS.md#c29)

X22 distinguishes a precision budget that grows with input length from fixed-budget reasoning; the regimes cannot be silently exchanged. X24’s terminal softplus disposition is retained in its machine receipt, rather than promoted to a clean activation victory. X25’s recipe-controlled tanh graph arm reaches 1.0 ID but only .5602/.4979 in its two OOD bins. These are evidence for separating learning, capacity and extrapolation, not an architecture-wide ranking. [C30](CLAIMS.md#c30) [C02](CLAIMS.md#c02)

DeltaStack is the closest comparison. The X32 full-PDF addendum confirms learned soft action weights and a continuous soft pointer. The code uses a bounded power-of-two stack and clamps the pointer. Crucially, the paper **does** have a stack-size ablation and an S≥D sufficiency condition; claiming “no capacity analysis” would be false. The narrower reported gap is failure-boundary characterization when capacity is inadequate, with saturation telemetry and a degradation policy. Its parallel soft-pointer recurrence is also a real efficiency advantage, not an equivalent scheduling of hard age-1 correction. Bibliographic entries and the detailed comparison are in [RELATED.md](RELATED.md). [C06](CLAIMS.md#c06) [C17](CLAIMS.md#c17)

## 3. The instrument: ordered operations and receipts

The store contains keys, values, strengths and birth information. READ precedes POP, post-POP death, WRITE, rounded decay, age advance and decay death. CORRECT selects the lowest-index live age-1 row; correction does not renew its birth. F12/F12g/F12h are semantic engineering amendments, not interchangeable versions of a content-argmax write. Even ZERO_RESIDUAL can request an append. Repeated content therefore does not automatically mean no allocation. [C07](CLAIMS.md#c07)

This ordering makes the instrument interpretable but also costly. An append creates the next token's potential correction target. The target's current payload affects residuals and gate computation. A detached routing decision has zero derivative through its discrete choice; it does **not** become independent of earlier state. Pressure records cap/request behavior, and raw cumulative pressure must be distinguished from a trailing-window saturation event. [C07](CLAIMS.md#c07) [C13](CLAIMS.md#c13) [C17](CLAIMS.md#c17)

## 4. The certified boundary

The following numbers are re-read from the selected-checkpoint `final_eval.json` files. They are balanced accuracies on the registered pools, not a new rerun. “Matched controls” refers to the X28E protocol, not equal runtime, equal state bytes, or a newly certified state-accounting theorem. [C08](CLAIMS.md#c08) [C09](CLAIMS.md#c09) [C10](CLAIMS.md#c10) [C11](CLAIMS.md#c11)

| Arm | ID | 2x | 4x | 9x |
|---|---:|---:|---:|---:|
| Emender B0, 736 slots | .994141 | .980469 | .984375 | .500000 |
| Emender B0x4, 2944 slots | .990234 | .980469 | .970703 | .947266 |
| GDN2 | .990234 | .564453 | .498047 | .498047 |
| E97 | .998047 | .541016 | .500000 | .500000 |

Receipts: [C08](CLAIMS.md#c08)–[C11](CLAIMS.md#c11). In particular, “both controls collapse to .50 at 2x” is an inaccurate condensation: their 2x scores are .5645 and .5410.

Fine sweeps place B0's first raw cap event and accept-collapse at k=736; k=720 holds. The learned schedule alternates append and age-1 correction through both a and b phases, with no POP in the traced words. This makes exhaustion a physical allocation boundary for this learned strategy. The larger-budget arm is separately trained, so its 9x result does not isolate a test-time budget intervention on identical weights. Moreover, its non-capacity hole means that more slots do not guarantee pointwise dominance at every depth. [C12](CLAIMS.md#c12) [C13](CLAIMS.md#c13)

## 5. Pressure: specific exhaustion telemetry, not universal correctness

B0 has 6400 traced words, including 3200 accept-class words. Of the accepts, 1408 are incorrect and pressured; 1792 are correct and unpressured. Thus PPV and sensitivity are both 1.000 **for this accept-class contingency**. Calling this perfection over all error types or all 6400 words would silently change the denominator. The larger arm's all-accept PPV is .333 and sensitivity .176: pressure is not a general error oracle. [C12](CLAIMS.md#c12)

At B0x4, raw pressure begins at k=2944 and accept-collapse at k=3040: a 96-depth lead. Windowed saturation begins later and leads collapse by 64 depth. First saturation positions are 1524 and 5940, following 2B+52 in these runs. The signal occurs before EOS but **inside the b phase**, contrary to the pre-registered before-b expectation. Cumulative pressure never reaches the registered .9 saturation threshold. The directed hypothesis remains **MIXED**. [C13](CLAIMS.md#c13)

The honest nulls matter. B0x4 rejects balanced words in the pressure-free hole spanning at least k=560–816, then recovers at later sampled depths. It also saturates at some depths while accuracy remains .832, above the registered collapse criterion. The prediction that the larger arm holds everywhere B0 collapses is refuted. Pressure exposes budget exhaustion and can precede one failure mode; it is specific to resource stress and insensitive to some non-capacity failures. [C13](CLAIMS.md#c13)

## 6. Efficiency: maintenance, fused execution, and the remaining floor question

The X30 v1 maintenance result is approximately 423x on CPU and 677x on GPU at S=11776, comparing an inert full-store maintenance workload. These numbers exclude reads, content writes, snapshots and synchronization; they are not whole-cell speedups. X31 v2 reports 109-test numerical parity at the registered tolerance and a 93.109x fresh-forward speedup over its matched X30 reference at S11776. The cold H16/H1 scaling gate still fails. The parent adjudication calls this 7/8 gates plus a separate steady-state pass on the eighth, with initialization removed; it does not erase the cold-start failure. [C14](CLAIMS.md#c14) [C15](CLAIMS.md#c15)

The LM-scale program moves from X34's roughly 23.4x historical smoke overhead to X36 part 6's 1.541x B64/T512 whole-LM wall ratio. These are different workloads and stages, not a single controlled speedup factor. Fresh capacity doubling passes its registered limits. Continued capacity, long context, inclusive launches and continued host synchronization remain open failures; the narrow custom-kernel count is not the inclusive launch count. Part 7 also rejects a proposed dense-gradient-scatter diagnosis: that scatter executes zero times in the measured capacity workload. [C16](CLAIMS.md#c16) [C20](CLAIMS.md#c20)

Near-budget arithmetic is a useful engineering account, not a performance lower-bound theorem. Part 6's near-budget step ratios are about 2.050 and 2.531 on successive doublings, but useful live occupancy grows as well. Logical access counts are not measured DRAM traffic. The part-7 arithmetic identifies remaining work and required savings without establishing an irreducible hardware floor. The later parent adjudication labels near-budget scaling a FLOOR and a registration-regime mismatch. We retain that disposition as an attributed interpretation: doubling live occupancy explains approximately doubled logical work, but alone does not prove a wall-time lower bound or explain every measured ratio above two. Part 7 also changes fixed-live outcomes (including a live1024 step-doubling failure), so its detailed machine ledger takes precedence over condensed claims that all medium-live controls pass. Its short/long whole-LM ratios are 1.551x/2.621x; the 1.541x headline explicitly names part 6. The paper therefore retains the near-budget performance floor as a qualified arithmetic argument, not a proved hardware limit. [C16](CLAIMS.md#c16)

X35/v3 refutes the unconditional entry-only two-pass routing proposal. Two tokens evaluated against empty entry state would both append; sequentially the second can correct the first token's slot. Even append-only traces can retain payload and causal-read dependencies. The trained ladder has a .5 event-token fraction, so sparse exceptions do not remove this case. Interpretation: the ordered content dependence participates in the useful mechanism and resists this parallelization. This is not a theorem that all expressive memory must be serial; DeltaStack buys parallelism through a different soft-pointer recurrence. [C17](CLAIMS.md#c17)

## 7. Language modeling: a live lane, a pilot, and a null

X34 total parameters are 21,913,872 versus 21,914,119, a .001127% discrepancy. Trainable counts differ, and actual wall cost is not compute matched. Both B8 arms reach 100,003,840 tokens. The final training minibatch loss favors Emender, 1.854418 versus 1.867671, but held-out NLL is 1.799193 versus baseline 1.798722 and PPL is 6.044765 versus 6.041918. The train-loss edge must not be retold as a held-out language-model win. [C18](CLAIMS.md#c18) [C19](CLAIMS.md#c19)

The lane is active on real text: occupancy grows from 256/head at T512 to 2048/head at T4096, with slope 1.0 and the same append/correct alternation diagnosis. That resembles a token transcript rather than evidence of selective conceptual storage. Both arms score .5 on the registered zero-shot depth probes. Few paired corruptions and tokenizer-dependent depth further limit inference; real-text lane activity is not transfer of the certified counting capability. [C20](CLAIMS.md#c20)

A consequential confound was caught: historical X34 uses B8, whereas the intervention family uses B64 and many fewer optimizer steps. The new B64 control supplies matched exposure at 100,007,936 tokens, PPL 6.856418. X37 b/c and X38's selected full arm pass the intended quality band against that control. The existing comparability JSON lacks terminal d; re-reading d's completed receipt gives 6.682684, ratio .974661 to control. This corrects a historical “all worse” interpretation without making the B64 family beat the stronger absolute B8 result. [C21](CLAIMS.md#c21) [C22](CLAIMS.md#c22)

## 8. Differentiation: geometry, objectives, and policy

Seeding interventions b and c end with identical head peaks in 16/16 stories; the hierarchy arm d has 0/16 identical peaks and the best PPL among these interventions. This is an occupancy result, not proof of identical reads or an exact weight-permutation fixed point. The source audit finds unequal head rows already at initialization, and learned W_in distances diverge even when occupancy agrees. Latent functional diversity may therefore escape the peak statistic. [C22](CLAIMS.md#c22)

X38 moves mean read cosine from .000750 to −.313177, without monotonic decrease; physical-slot JS ends lower, .028949→.019245. It provides forced anticorrelation, not complementary slots. Its quality cost is comparator dependent: 11.4% worse than historical B8 a, about .61% worse than B64 c, and better than the protocol-matched B64 control. The hierarchy is also costly: about 7.876x c's median CUDA-event step time. Architectural policy is a demonstrated lever for occupancy diversity here, not a proof that learned diversity is impossible or free. [C23](CLAIMS.md#c23) [C21](CLAIMS.md#c21) [C22](CLAIMS.md#c22)

## 9. Usage: four levers and a conceptual-cache hypothesis

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

## 10. Discussion: the architectural trade

Our interpretation is that emendation, bounded allocation and explicit exhaustion jointly create a useful research instrument: failure is measurable in state coordinates rather than accuracy alone. The larger arm's hole prevents a capacity-only theory of all errors. The serial counterexamples prevent an unconditional parallel schedule for the present recurrence. The semantic-wire intervention shows that learning content is not enough to change allocation when the necessary read-to-gate connection is absent. Each conclusion is local to the receipted mechanisms. [C13](CLAIMS.md#c13) [C17](CLAIMS.md#c17) [C25](CLAIMS.md#c25)

The proposed trade-off is therefore architectural: a hard ordered correction program offers explicit operations and interpretable exhaustion, while soft-pointer designs can support more parallel execution. Neither the measured performance failures nor the local dependency obstruction proves a universal expressivity/efficiency theorem. Boundary characterization, rather than the existence of structured memory or a tunable store-size knob, is the defensible distinction from the surveyed prior art. [C06](CLAIMS.md#c06) [C16](CLAIMS.md#c16) [C17](CLAIMS.md#c17)

## 11. Limitations and open program

The LM leg remains **pilot-grade**: one training seed per arm, TinyStories scale, small fixed probes, incomplete trainable-capacity and runtime equivalence, inherited provenance qualifications, and no demonstrated zero-shot structural transfer. Budget-capped flat occupancy can destroy information. Scale-up and general semantic retention remain untested by these receipts. A curriculum probe is proposed to distinguish learnability from absent zero-shot transfer; no positive outcome is assumed. [C18](CLAIMS.md#c18) [C20](CLAIMS.md#c20) [C22](CLAIMS.md#c22) [C26](CLAIMS.md#c26)

The certified append-only and unbounded arms have checkpoint receipts but no terminal final evaluation used here. They remain pending/censored evidence, not negative experimental results. No fresh training, GPU work, Lean verification, Frontier submission or resilient-training validation is part of X41. [C28](CLAIMS.md#c28) [C03](CLAIMS.md#c03)

### X40 slot — EmenderLM pure model: IN FLIGHT

Reserved for the separately registered attention-free EmenderLM program. The registration is implementation/preflight evidence, not a completed model result. Insert outcomes only after endpoint receipts, policy-specific parity and matched probe evidence exist; retain failures if they occur. No X40 number enters this draft. [C27](CLAIMS.md#c27)

### Proposed next evidence

We propose independent seeds and task variants for the capacity boundary; budget interventions on identical checkpoints; state-byte accounting and stronger controls; causal trained-language ablations for the match wire; retention probes alongside occupancy; and performance measurements at continued near-budget state. These are research proposals motivated by the holes and failed gates, not completed experiments. [C05](CLAIMS.md#c05) [C13](CLAIMS.md#c13) [C16](CLAIMS.md#c16) [C26](CLAIMS.md#c26)
