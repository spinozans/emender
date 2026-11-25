# Two notes recover some of the old vector's predictive value, not all of it

**A byte and two small notes were enough to reconstruct something useful about a particular earlier occurrence. They did not replace the full vector without a loss.** Consider encountering `color: ` again. The memory finds an earlier `b`. Instead of storing the model's entire 32-number representation at that old `b`, D remembers two bytes: how surprising the `b` was then to the model and to a text counter. It uses the letter and those notes to reconstruct a vector when the reader asks for one.

On 128 new documents, that reconstruction beat using the same average vector for every `b` (**D:P +.000739 nats/byte**), and beat decoding from the notes of a different earlier `b` (**D:S +.000643**). All three readers had the *right* notes to decide whether to trust the copy; only the input to the value decoder differed. The full-vector reader J still beat D by **.001210** [.000952,.001457]. Descriptively, D recovered **38%** of the full vector's loss advantage over the per-byte average (`.000739/.001948`); that is not a fraction of the vector's information or variance.

D's advantage over the confirmed U reader was **+.000248 [−.000140,+.000664]**. Positive at all seeds but inconclusive across documents: **do not advance D to confirmation or claim equivalence**. The result answers a reconstruction question about a frozen reader, not what a purpose-trained compact model could learn. B remains the baseline, U the native-final-confirmed reader, J the fixed candidate replicated on calibration. No final used, no scale nomination.

[Protocol](DECODE_RECORD.md) · [implementation and retained fixture failure](DECODE_RECORD_IMPLEMENTATION.md) · [verified aggregate](receipts/decode_record_result.json)

## The comparison

All six readers used the same real host prefill and same latest eligible address. B/U/J read the actual past vector. P used a per-byte average. D used the per-byte decoder and the two correctly bound codes; S used the same decoder and a causal same-follower/log-age-bin pair of codes from the wrong occurrence when available. P/D/S applied frozen J trust coefficients to **their own** base scores, not J's full-vector confidence. No target in address, marks, reconstruction or answer. The standalone D path matched the panel's complete D scores with the actual writer disabled on an invented document at each fitted checkpoint; this does not mean the *whole comparison* avoided the full-vector writers.

In 256 new native TRAIN FIT documents the decoder saw **1,048,320 actual teacher writes per seed**, excluding the unreadable first slot, no target loss and no future hit selection. 180 of 256 follower cells were observed, 3 just once; 76 used the global-mean fallback, the largest cell had 170,901 writes. All three table digests were sealed before opening the 128 disjoint native TRAIN PROBE documents. No U/J coefficient or neural-weight updates.

| Reader | Mean NLL | 4901 | 4902 | 4903 |
|---|---:|---:|---:|---:|
| B: full vector, no trust correction | 1.147719 | 1.148679 | 1.146806 | 1.147671 |
| U: full vector, neural trust note | 1.144764 | 1.145672 | 1.144063 | 1.144557 |
| J: full vector, both trust notes | **1.143306** | **1.144213** | **1.142613** | **1.143093** |
| P: byte average, both trust notes | 1.145255 | 1.146334 | 1.144343 | 1.145087 |
| D: decoded value, both trust notes | 1.144516 | 1.145561 | 1.143631 | 1.144356 |
| S: wrong decoder notes, both trust notes correct | 1.145159 | 1.146184 | 1.144277 | 1.145014 |

All 524,288 real next bytes per checkpoint, including misses. Positive gain = control minus candidate loss. 10,000 bootstrap draws of whole documents paired across readers and the three fixed weights, RNG 4913, linear percentile intervals. Seeds do not triple the document sample size.

| Candidate:control | Gain (nats/byte) | Paired 95% interval | 4901 / 4902 / 4903 |
|---|---:|---|---|
| **D:P** | **+.000739** | [.000564,.000925] | .000774 / .000711 / .000731 |
| **D:S** | **+.000643** | [.000503,.000792] | .000624 / .000646 / .000658 |
| J:P | +.001948 | [.001601,.002292] | .002121 / .001730 / .001993 |
| J:D | +.001210 | [.000952,.001457] | .001347 / .001019 / .001263 |
| D:U | +.000248 | [−.000140,.000664] | .000111 / .000432 / .000201 |
| D:B | +.003202 | [.002572,.003992] | .003118 / .003174 / .003315 |
| J:U | +.001458 | [.001109,.001851] | .001458 / .001451 / .001464 |
| U:B | +.002954 | [.002467,.003525] | .003007 / .002742 / .003114 |
| S:P | +.000096 | [−.000014,.000208] | .000150 / .000065 / .000073 |

Per checkpoint there were 512,819 hits, 11,469 misses and 243,955 correct copies. 436,658 donor positions moved; **432,164 / 432,114 / 432,200** note pairs actually changed. Every PROBE document contributed hits, misses, correct copies, displaced slots and changed pairs. Across 16,410,208 actual read-components per seed, MSE to the true vector was P .18312/.18744/.18820; D .16264/.16447/.16684; S .18780/.19195/.19218. D lowered both prediction loss and reconstruction error, but these metrics are not interchangeable: S had *higher* MSE than P while its prediction-loss interval against P crossed zero.

## Price of the smaller record

| Reader | Explicit historical payload / input | Additional fixed decoder table / checkpoint |
|---|---:|---:|
| B / U / J | 128 / 129 / 130 bytes | None |
| P | 2 bytes | 32,768 bytes used |
| D / S | 2 bytes | 100,352 bytes |

At 4,096 positions J's explicit records take 520 KiB; D's records **plus** its installed table take 106 KiB. They are equal at 784 positions if the entire table is charged to one context. The table is shared across documents at the same weight. This comparison excludes the common frozen model, text, byte index, text-count tables, activations, queried payloads and scoring buffers. E.g., the count writer peaked at 8,074 contexts / 10,370 context-follower pairs per PROBE window. No 65× process-memory or serving-speed claim.

Measured mean milliseconds per 4K PROBE window and checkpoint: host prefill **103.25**; shared address and byte code 46.80; direct head 26.60; current host field 1.96; literal/neural tape builds 1.61/.63; true note reads .074. Full vector write/read **1.635/.087** versus **.319** to decode D at hits (.064 for the P lookup). S additionally cost 8.89 for search/read and .431 to decode. Outer B/P/D/S head 7.31/6.63/6.56/6.47; trust responses 10.12/6.16/6.15/5.90 (B's clock includes **both** U and J). Shared component timings in the comparison are not standalone throughput. FIT prefill 107.60 ms/window, full teacher write 1.99, text/neural marks 1.47/4.16, group statistics 2.41; the final table solves took 1.14–1.64 ms per checkpoint. MSE was computed on already obtained panel values only, never fed back.

## Checks and resources

Qualified public source `26fef4a398e3a819891387c761d49436db343ac5` (local `6a2db80720a0c0edaf248e59dc10f3b829016e0a`): 142 CPU tests, 654 native address frontiers, three real strict CPU loads. Metadata excluded 43,320 prior full-document hashes; 23,187 TRAIN documents eligible before reserving the next 256+128. Whole-hash disjointness does not guarantee no repeated passages. FIT-only preparation checked 7,426,635 source bytes in 28.06 wall / 27.97 CPU s. Three synthetic prefills per fitted checkpoint (nine total) verified the full siblings and writer-disabled D before PROBE access. Independent FIT witness at 768 units; PROBE checked 2,349,921 source bytes, no retained PROBE window file. 1,152 natural seed/document prefills to terminal exposure.

One RTX 6000 Ada lease released rc=0, **≤221 incremental card-s** of the 1,200 ceiling; 236.29 s inclusive scheduling wall, 427.82 owner/child CPU s, peak device allocated/reserved 715,992,064 / 746,586,112 bytes, peak RSS 2,636,548 KiB. The close audit independently rehashed three weight files, verified the table/fit seals and all 256 cells at every seed against a second ridge solution, rebuilt the nine paired intervals and the MSE denominators, and checked charge/release without reopening text. Aggregate SHA-256 `d7d826da0d41f90fececc1262a62ad9b278617e506c9c8b447ca86f106c6f948`. This DR cohort is spent.
