# What learned when nothing was frozen?

**A next-token loss trained the whole path on an invented lookup task.** Two 4,771-parameter token models learned from scratch to read a different miniature manual on each episode. Both answered 4,096/4,096 independent evaluation queries correctly. When the facts changed underneath exactly the same prompts and search weights, both followed the new facts on 4,096/4,096 queries. **A blank-start passage state was already sufficient.** This establishes trainability of the system in this setting, not an advantage for seeded transformations or performance on natural language.

[Recipe and source fixed before fitting](https://github.com/spinozans/emender/tree/0db599684ea9925dd93c03800623465ba6b5615e/experiments/expressivity_tasks/retrieved_span_effects) · [Joint-training design](../../../docs/RETRIEVED_SPAN_EFFECTS.md) · [Aggregate including first-step gradients/updates](receipts/joint_fixture.json).

One dense gated-delta head processed the real prompt; another processed the four separate passages. Soft search scored all four titles, with no target or supporting-row supervision. STATE ran the passage head from zero. EFFECT ran it from a copy of the live head state without detaching it. Embeddings, both recurrences, search keys/queries, reader query, merge/gate and token decoder all trained. Every one of the 13 parameterized modules had a finite nonzero first-step gradient and parameter update in both arms. The fits had the same initial weights, episodes and optimizer schedule: 320 updates × 64 one-token targets per arm. No interim evaluation or second seed. Independent evaluation draws are not a new language corpus or necessarily unique manuals.

| Evaluation at unchanged fitted weights | STATE correct / 4,096 | STATE nats/token | EFFECT correct / 4,096 | EFFECT nats/token |
|---|---:|---:|---:|---:|
| Original manual and original answer | 4,096 | .007333 | 4,096 | .007289 |
| Disable passage hint | 534 | 2.771166 | 544 | 2.826832 |
| Changed manual, scored on **old** answer | 0 | 8.406715 | 0 | 8.418632 |
| Same changed manual, scored on **new** answer | 4,096 | .007307 | 4,096 | .007246 |
| Rotate only the seeded contributions across titles; keep the original blank-start writes | — | — | 4,096 | .007420 |

The disable condition is a same-weight intervention, not a trained no-passage baseline. The prompt alone has a 1-in-8 chance rate. The whole-value change rotated the two field values together across titles; by construction every correct answer changed, and the code checked that every search weight stayed identical. This is evidence the model used the **content**, not merely the existence of a search result. Rotating just `A_i S` cost EFFECT .000131 nats/token and changed no top-1 answer. The fixture tests integration and credit flow, **not** whether a trained natural LM benefits from EFFECT more than from STATE. Neither failure nor success of a frozen retrofit was needed to run it.

Eight CPU unit cases (including the prior algebra cases) passed before fitting. Both fits and the one evaluation ran on CPU with a single Torch thread and no natural data, external weights, GPU or remote submission. The compact JSON is the worker's aggregate output, SHA256 `cc7bb11d9c5886e93d2d2ff7e3cce6586dcfa6aa59bfede242637173e2353e6c`. Fit-loop times were 2.08 and 1.92 seconds respectively; do not extrapolate that single toy timing to corpus retrieval or full LM training.
