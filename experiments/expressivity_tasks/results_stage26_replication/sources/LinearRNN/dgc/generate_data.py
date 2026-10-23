#!/usr/bin/env python3
"""
Two-bucket reachability dataset generator with TXT outputs.

Encoding (one per line):
  src;i1-j1;i2-j2;...;tgt

Labels:
  0 or 1, one per line

Splits:
  train:        N in [2, n]
  val_bin0:     N in [2, n]
  val_bin1:     N in [n+1, 2n]
  val_bin2:     N in [2n+1, 3n]
"""

import os
import random
import argparse
from typing import List, Tuple


# ------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------

def build_edges_from_partition(N: int, bucket: List[int]) -> List[Tuple[int, int]]:
    """Connect consecutive nodes inside each bucket."""
    edges: List[Tuple[int, int]] = []
    for b in (0, 1):
        nodes = [i for i in range(N + 1) if bucket[i] == b]
        nodes.sort()
        for u, v in zip(nodes, nodes[1:]):
            edges.append((u, v))
    edges.sort()
    return edges

def binary_encode(n: int) -> str: 
    return bin(n)[2:]
    # return str(n)

def encode_instance(edges: List[Tuple[int, int]], src: int, tgt: int) -> str:
    return ";".join(
        [binary_encode(src)] + [f"{binary_encode(u)}-{binary_encode(v)}" for (u, v) in edges] + [binary_encode(tgt)]
    )
    
def generate_instance(N: int, p: float, label: int) -> Tuple[str, int]:
    """
    Guarantees:
      - If label == 1, there is a path 0 -> ... -> N
      - If label == 0, there is no such path
    """

    bucket = [0] * (N + 1)

    b0 = random.randint(0, 1)

    if label == 1:
        # force 0, N, and at least one predecessor into same bucket
        bucket[0] = b0
        bucket[N] = b0

        u = random.randint(1, N - 1)
        bucket[u] = b0

    else:
        # negative: separate 0 and N
        bucket[0] = b0
        bucket[N] = 1 - b0

    # sample remaining nodes
    for i in range(1, N):
        if bucket[i] != 0 and bucket[i] != 1:
            bucket[i] = 0 if random.random() < p else 1

    edges = build_edges_from_partition(N, bucket)
    text = encode_instance(edges, src=0, tgt=N)
    return text, label


# ------------------------------------------------------------
# Dataset generation
# ------------------------------------------------------------

def generate_balanced_split(
    num_samples: int,
    N_low: int,
    N_high: int,
    p: float,
) -> List[Tuple[str, int]]:
    """Generate exactly balanced positive / negative samples."""
    pos = num_samples // 2
    neg = num_samples - pos

    data: List[Tuple[str, int]] = []

    for _ in range(pos):
        N = random.randint(N_low, N_high)
        data.append(generate_instance(N, p, label=1))

    for _ in range(neg):
        N = random.randint(N_low, N_high)
        data.append(generate_instance(N, p, label=0))

    random.shuffle(data)
    return data


# ------------------------------------------------------------
# Writing helpers
# ------------------------------------------------------------

def write_txt(src_path: str, tgt_path: str, data: List[Tuple[str, int]]) -> None:
    os.makedirs(os.path.dirname(src_path), exist_ok=True)

    with open(src_path, "w", encoding="utf-8") as f_src, \
         open(tgt_path, "w", encoding="utf-8") as f_tgt:
        for text, label in data:
            f_src.write(text + "\n")
            f_tgt.write(str(label) + "\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./data/two_bucket_txt")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--train_size", type=int, default=100000)
    ap.add_argument("--val_size", type=int, default=30000)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    n = args.n
    out_dir = f'data/n{n}'
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Train
    # -------------------------
    train = generate_balanced_split(args.train_size, 2, n, args.p)
    write_txt(
        os.path.join(out_dir, "train_src.txt"),
        os.path.join(out_dir, "train_tgt.txt"),
        train,
    )

    # -------------------------
    # Validation bins
    # -------------------------
    val0 = generate_balanced_split(args.val_size, 2, n, args.p)
    val1 = generate_balanced_split(args.val_size, n + 1, 2 * n, args.p)
    val2 = generate_balanced_split(args.val_size, 2 * n + 1, 3 * n, args.p)

    write_txt(
        os.path.join(out_dir, "val_src_bin0.txt"),
        os.path.join(out_dir, "val_tgt_bin0.txt"),
        val0,
    )

    write_txt(
        os.path.join(out_dir, "val_src_bin1.txt"),
        os.path.join(out_dir, "val_tgt_bin1.txt"),
        val1,
    )

    write_txt(
        os.path.join(out_dir, "val_src_bin2.txt"),
        os.path.join(out_dir, "val_tgt_bin2.txt"),
        val2,
    )

    print("[✓] Dataset written to", out_dir)
    print("[✓] Files:")
    print("    train_src.txt / train_tgt.txt")
    print("    val_src_bin0.txt / val_tgt_bin0.txt")
    print("    val_src_bin1.txt / val_tgt_bin1.txt")
    print("    val_src_bin2.txt / val_tgt_bin2.txt")
    print("[✓] Example:")
    print("    ", train[0][0], "->", train[0][1])


if __name__ == "__main__":
    main()
