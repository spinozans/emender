# Three-family LM length generalization

Proxy scale: 150M trainable parameters, 45-minute active training, Pile slice; NOT TinyStories and NOT the terminal 1.3B comparison. First apples-to-apples GDN2-vs-emender LM length receipt only when all arms complete.

| family | status | T | NLL | perplexity | PPL / own T512 |
|---|---|---:|---:|---:|---:|
| transformer | completed | 512 | 5.597676038742065 | 269.7986766257226 | 1.0 |
| transformer | completed | 1024 | 4.534669876098633 | 93.19274507312625 | 0.34541587171091875 |
| transformer | completed | 2048 | 4.751263618469238 | 115.73043128129233 | 0.4289510709566564 |
| transformer | completed | 4096 | 4.8190507888793945 | 123.84747756580498 | 0.459036638410247 |
| gdn2 | completed | 512 | 5.115240097045898 | 166.54076212845786 | 1.0 |
| gdn2 | completed | 1024 | 4.037253022193909 | 56.67045630937872 | 0.34027979447858603 |
| gdn2 | completed | 2048 | 4.261444807052612 | 70.91236416013992 | 0.42579584273454385 |
| gdn2 | completed | 4096 | 4.2801525592803955 | 72.25146179758515 | 0.4338365026926888 |
| pure | queued | pending | pending | pending | pending |

Incomplete/failed rows are not measurements. No three-family result is claimed until all arms complete.
X40 window discipline: two reset contiguous batch-1 windows starting i*T; coverage differs by T.
No resilient DiLoCo behavior changed; requirement IDs not applicable.
