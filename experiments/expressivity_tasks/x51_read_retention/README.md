# X51 — read for now, or save the answer?

**[Completed result](RESULT.md) · [redacted aggregate](receipts/result.json).** On 1,024 fresh TRAIN-probe documents, making the current read reduced loss from 3.09848 to 1.40494 nats/byte versus the matched small byte-only reader. Keeping earlier answers when training improved on discarding by a further .000087 [.000036,.000137]. But erasing old answers at the *same* fitted KEEP weights had mixed signs and an interval spanning zero. The current lookup was valuable; reliable use of stored answers at inference wasn't established. The lookup cost about 22× the batched CPU evaluation time of the no-read model. Both data roles are spent.

[Prospective test and interpretation](PROTOCOL.md) · [parent: working XSA tape and learned follow-up result](../x50_read_tape/RESULT.md)

The question isn't how often to query. Both reading arms get one identical P100 answer at each real byte. They differ only in whether the *content* of a past answer is still accessible. A third arm never consults the archive. It has a FIT-only unigram prior and can learn from all the observed text. The comparison is for the same small neural reader and training allowance, not an unrestricted byte-only language model.

[`reader.py`](reader.py) has the whole visibility rule. `acquire` has no label argument; NONE is not even given a source capability. `predict` forms the current query with the current answer if present. Each old answer slot has exactly the same byte/type/position embedding in KEEP and DISCARD, but only KEEP adds that old histogram's projected value. Attention sees history through the current observation plus a single current answer slot, never future rows. Comparing KEEP to DISCARD is not comparing a longer sequence to a shorter one. During the additional fixed-KEEP-weight DROP diagnostic, hiding old content cannot change the first prediction in a window.

[`test_retention.py`](test_retention.py) checks all 256 scores against literal single-frontier attention, KEEP's forward and backward parity with X50 ONE, interventions and gradients on old answers, genuine no-read, FIT-only counts, C14 rank continuity, freeze safety and paired inference. [`study.py`](study.py) handles fitting, diagnostic clocks and document bootstrap. [`data.py`](data.py) and [`run.py`](run.py) refuse natural work without the reviewed public qualification and a single fresh detached reservation. The ordinary tests need no corpus, GPU or native worker:

```bash
env -u GPU_LEASE_HELD -u PYTHONOPTIMIZE CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q experiments/expressivity_tasks/x51_read_retention
```
