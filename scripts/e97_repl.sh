#!/usr/bin/env bash
# e97_repl.sh — operator convenience wrapper for the E97 GGML CPU REPL
# (chat with the promoted E97 4B Pi-native agent on CPU while the GPUs train;
# never touches the GPUs).
#
# Chat mode (default; canonical Pi-native episode codec, temperature 0.7,
# top_p 0.95 — greedy repetition-collapses this model):
#   scripts/e97_repl.sh
#
# Raw LM continuation (feel the base capability):
#   scripts/e97_repl.sh --mode raw
#
# Full f32 weights instead of the q8_0 default:
#   scripts/e97_repl.sh --gguf e97-4b-pi-f32-aligned.gguf
#
# Reproducible sampling: add --seed N. Turn budget: --max-tokens 512 (default).
# REPL commands: /reset (new episode) | /raw (toggle chat<->raw) | /exit.
#
# Environment overrides:
#   E97_REPL_PORT_DIR   port dir (default /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-llama-cpp-port-v1)
#   E97_REPL_THREADS    CPU threads (default 32 — measured sweet spot; the graph is bandwidth-bound)
set -euo pipefail
PORT_DIR=${E97_REPL_PORT_DIR:-/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-llama-cpp-port-v1}
[[ -x "$PORT_DIR/e97-runner" ]] || { echo "e97-runner not found in $PORT_DIR (run $PORT_DIR/build.sh first)" >&2; exit 66; }
cd "$PORT_DIR"   # default GGUF path + prefix-state snapshot are relative to the port dir
exec "$PORT_DIR/e97-runner" --interactive --threads "${E97_REPL_THREADS:-32}" "$@"
