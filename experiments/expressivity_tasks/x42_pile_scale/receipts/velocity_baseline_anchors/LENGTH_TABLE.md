# Three-family LM length generalization

Proxy scale: 150M trainable parameters, 45-minute active training, Pile slice; NOT TinyStories and NOT the terminal 1.3B comparison. First apples-to-apples GDN2-vs-emender LM length receipt only when all arms complete.

| family | status | T | NLL | perplexity | PPL / own T512 |
|---|---|---:|---:|---:|---:|
| transformer | queued | pending | pending | pending | pending |
| gdn2 | queued | pending | pending | pending | pending |
| pure | queued | pending | pending | pending | pending |

Incomplete/failed rows are not measurements. No three-family result is claimed until all arms complete.
X40 window discipline: two reset contiguous batch-1 windows starting i*T; coverage differs by T.
No resilient DiLoCo behavior changed; requirement IDs not applicable.
