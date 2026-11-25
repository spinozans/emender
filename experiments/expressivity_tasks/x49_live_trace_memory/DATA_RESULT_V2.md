# X49 v2 data gate — TRAIN prepared, evaluation closed

The [amendment](DATA_V2.md) was fast-forwarded to public `892cc5b0409828a220a861828e98d8b3de457b85` **before** v2 selection or extraction. At private source `193479fc5989bcf086e925cbdff7aaf0888db4b6`, five selector/boundary tests passed. The v2 metadata gate found exactly the v1 eligible counts again. The frozen exclusion union contains **38,517 unique full-document SHAs**. A second, set-first selector checked deduplication, ranks, starts, document and split disjointness against the prospective algorithm. This excludes full-document overlap with the registered historical memberships, not all repeated substrings.

| Native split | Eligible after exclusions | Reserved | Payloads read by this extraction |
|---|---:|---:|---:|
| TRAIN | 26,515 | 1,024 | 1,024 full documents |
| Calibration | 110 | 110 | 0 |
| Final | 265 | 128 | 0 |

The extractor independently regenerated the reservation before opening the corpus, positioned reads only at the selected TRAIN documents, checked the pinned open-file identity, and matched each entire document's SHA before taking its registered C14 shift. **4,194,304 real next-byte targets**, 4,195,328 bytes in the private window stream. No training, no evaluation, no model comparison.

| Aggregate binding | SHA-256 |
|---|---|
| Private v2 reservation | `cdec56106c43b9b5abcdfe0d42a24888ec2c62c925fc1df9cb40802a952c2f1b` |
| TRAIN window stream | `13d3a91132ff6fdca8b9b449defd96f73fdd95162ab882edf23bdc95420c8aa6` |
| All TRAIN input/window/target hash receipts | `10d03082afd8338d408907d012f3da85b7ad57b1f3a3c192c7ebf00cadf8ddf8` |
| TRAIN identity (includes all three order digests) | `f6ea7725ada25ce23b304e0af1a64588669bf13b3ce3a1c11a8743da43dc961b` |

Shared successful preparation on lambda01 (GNU time, hundredth-second resolution): reserve **2.11 s wall / 2.11 CPU s**, max RSS 258,144 KiB, no corpus reads; revalidate+extract **2.34 s wall / 2.23 CPU s**, max RSS 256,776 KiB, 19,212,278 corpus bytes read. Total **4.45 s wall / 4.34 CPU s**; attribute equal shares in the eventual training report. Not a card charge or an inclusive cost for earlier development/the v1 refusal.

Private files remain under `output/x49-live/v2/`. The version must be explicit (`panels reserve --protocol v2`, then `panels train --protocol v2`, both one-shot). Do not rerun or overwrite the frozen stream. `train_stream.TrainStream` verifies the frozen identity, stream, all shifts and the exact seed-specific order before exposing a single TRAIN sample by 1-based step. It does not open the corpus or supply eval bytes.
