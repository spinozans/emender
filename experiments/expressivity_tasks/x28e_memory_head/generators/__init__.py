"""X28E stage generators — the frozen data contract (M2 engineering freeze).

Three stage generators per the registration's frozen data contract
(REGISTRATION.md, "Data (immutable, seed-fixed at freeze)" section):

  * ``stage1_mqar``            — associative recall (reviewed MQAR family;
                                 mean-per-class-recall metric)
  * ``stage2_append_necessity``— fresh-key selective copy + slot-occupancy
                                 receipt instrumentation
  * ``stage3_ladder``          — the a^k b^k ladder (balanced classes, graded
                                 bins, malformed reject shapes, binary
                                 balanced-accuracy reporting, X26c discipline)

Every pool is generated from a FIXED SEED with hash-receiptable determinism:
the same seed produces byte-identical data (receipted by
``check_determinism.py`` -> ``receipts/determinism_check.json``). Pool
hashes are the freeze's data-identity receipts; the pools are immutable and
never regenerated with different constants (any deviation requires a NEW
separately-reviewed registration).
"""
from .stage1_mqar import MQARSplits, STAGE1_SPLITS, stage1_pools, stage1_hash
from .stage2_append_necessity import (
    AppendNecessitySplits, STAGE2_SPLITS, OCCUPANCY_GRID, stage2_pools, stage2_hash,
)
from .stage3_ladder import (
    LadderSplits, STAGE3_SPLITS, LADDER_VOCAB, ladder_bin_pool, ladder_train_pool,
    ladder_validation_pool, ladder_sweep_pool, ladder_hash, binary_report,
)

__all__ = [
    "MQARSplits", "STAGE1_SPLITS", "stage1_pools", "stage1_hash",
    "AppendNecessitySplits", "STAGE2_SPLITS", "OCCUPANCY_GRID", "stage2_pools",
    "stage2_hash",
    "LadderSplits", "STAGE3_SPLITS", "LADDER_VOCAB", "ladder_train_pool",
    "ladder_validation_pool", "ladder_bin_pool", "ladder_sweep_pool",
    "ladder_hash", "binary_report",
]
