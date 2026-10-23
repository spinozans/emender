#!/usr/bin/env python3
"""
NO-MOD 3x3 matrix multiplication dataset over {-1,0,1} with FINAL binary label.
NumPy version.

Format:
  src: "T|a1,...,a9|...|a1,...,a9"
  tgt: "0" or "1"

Label:
  y = 1 iff (P_T)[0,0] == 0
"""

from __future__ import annotations
import os
import argparse
import random
from typing import List, Tuple

import numpy as np


# -------------------------
# Matrix ops (NumPy)
# -------------------------
def eye3() -> np.ndarray:
    return np.eye(3, dtype=np.int64)


def sample_pm1_matrix(rng: random.Random) -> np.ndarray:
    # uniform over {-1,0,1}
    return np.array([rng.choice((-1, 0, 1)) for _ in range(9)],
                    dtype=np.int64).reshape(3, 3)


def label_for_seq(mats: List[np.ndarray]) -> int:
    P = eye3()
    for M in mats:
        P = P @ M
    return 1 if P[0, 0] == 0 else 0


def serialize_src(T: int, mats: List[np.ndarray]) -> str:
    parts = [str(T)]
    for M in mats:
        flat = M.reshape(-1)
        parts.append(",".join(str(int(x)) for x in flat))
    return "|".join(parts)


# -------------------------
# Sampling
# -------------------------
def gen_split_examples(
    rng: random.Random,
    T_min: int,
    T_max: int,
    n_per_len: int,
    balanced: bool,
    max_tries: int,
) -> Tuple[List[str], List[str]]:

    src_lines: List[str] = []
    tgt_lines: List[str] = []

    for T in range(T_min, T_max + 1):
        need = n_per_len

        if not balanced:
            for _ in range(need):
                mats = [sample_pm1_matrix(rng) for _ in range(T)]
                y = label_for_seq(mats)
                src_lines.append(serialize_src(T, mats))
                tgt_lines.append(str(y))

        else:
            need1 = need // 2
            need0 = need - need1
            got0 = got1 = 0
            tries = 0

            while (got0 < need0) or (got1 < need1):
                tries += 1
                if max_tries > 0 and tries > max_tries:
                    rem = (need0 - got0) + (need1 - got1)
                    for _ in range(rem):
                        mats = [sample_pm1_matrix(rng) for _ in range(T)]
                        y = label_for_seq(mats)
                        src_lines.append(serialize_src(T, mats))
                        tgt_lines.append(str(y))
                    break

                mats = [sample_pm1_matrix(rng) for _ in range(T)]
                y = label_for_seq(mats)

                if y == 0 and got0 < need0:
                    got0 += 1
                    src_lines.append(serialize_src(T, mats))
                    tgt_lines.append("0")

                elif y == 1 and got1 < need1:
                    got1 += 1
                    src_lines.append(serialize_src(T, mats))
                    tgt_lines.append("1")

    return src_lines, tgt_lines


def write_split(out_dir: str, split: str,
                src_lines: List[str],
                tgt_lines: List[str]) -> None:

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, f"{split}_src.txt"), "w") as f:
        f.write("\n".join(src_lines) + "\n")

    with open(os.path.join(out_dir, f"{split}_tgt.txt"), "w") as f:
        f.write("\n".join(tgt_lines) + "\n")


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T_train", type=int, default=300)
    ap.add_argument("--T_gap", type=int, default=100)
    ap.add_argument("--n_train_per_len", type=int, default=200)
    ap.add_argument("--n_val_per_len", type=int, default=50)
    ap.add_argument("--n_test_per_len", type=int, default=50)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--max_tries", type=int, default=200000)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    T0_min, T0_max = 1, args.T_train
    T1_min, T1_max = args.T_train + 1, args.T_train + args.T_gap
    T2_min, T2_max = args.T_train + args.T_gap + 1, args.T_train + 2 * args.T_gap

    print(f"[out] {args.out_dir}")

    splits = [
        ("train",     T0_min, T0_max, args.n_train_per_len),
        ("val_bin0",  T0_min, T0_max, args.n_val_per_len),
        ("test_bin0", T0_min, T0_max, args.n_test_per_len),
        ("test_bin1", T1_min, T1_max, args.n_test_per_len),
        ("test_bin2", T2_min, T2_max, args.n_test_per_len),
    ]

    for name, Tmin, Tmax, n in splits:
        print(f"[gen] {name} T=[{Tmin},{Tmax}]")
        src, tgt = gen_split_examples(
            rng, Tmin, Tmax, n, args.balanced, args.max_tries
        )
        write_split(args.out_dir, name, src, tgt)


if __name__ == "__main__":
    main()
