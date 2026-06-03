#!/usr/bin/env python3
"""CPU self-check for the S5-symmetric CMA-ES pre-flight (protocol §D.8).

NO GPU. NO TRAINING. This imports the modified driver (cmaes_search_s5.py),
dry-constructs each arm's seed config + search space, confirms the REAL
parameter estimator keeps each arm at ~8M, confirms the fitness adapter wires to
train_hybrid's real eval, and confirms the E88 BL-1 knobs (linear_state,
use_gate) are searched.

Run:  python scripts/s5_symmetric_preflight_check.py
Exit code 0 = all checks pass.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import cmaes_search_s5 as C

# Force the S5 objective for the whole check (as the launch commands do).
C.OBJECTIVE = 's5_acc@T128'

TARGET = 8_000_000
TOL = 0.10

# Ground-truth seeds = run_separation_suite.py MODEL_CONFIG centers (8M arms).
SEEDS = {
    'e88':         {'dim': 384, 'depth': 4, 'n_heads': 32, 'n_state': 32, 'lr': 3e-4,
                    'linear_state': 0, 'use_gate': 1},
    'm2rnn':       {'dim': 384, 'depth': 4, 'n_heads': 32, 'n_state': 32, 'lr': 3e-4},
    'fla-gdn':     {'dim': 640, 'depth': 4, 'n_heads': 32, 'n_state': 32, 'lr': 3e-4},
    'm2rnn-paper': {'dim': 608, 'depth': 4, 'n_heads': 32, 'n_state': 32, 'lr': 3e-4},
}

ANCHOR_FILE = os.path.join(
    ROOT, 'experiments', 'expressivity_tasks', 'results',
    's5_symmetric_20260603', 'seeds_s5_symmetric.json')

failures = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ''))
    if not cond:
        failures.append(name)


print("=" * 72)
print("S5-SYMMETRIC PRE-FLIGHT CPU SELF-CHECK (no GPU, no training)")
print("=" * 72)

# ---------------------------------------------------------------------------
print("\n[1] LM-loss path is UNCHANGED (the §5 path the protocol must not mutate)")
C.OBJECTIVE = 'lm_loss'
lm_space = C.get_search_space('e88')
check("LM e88 search space is the 1.3B-scale space (dim up to 4096)",
      lm_space['dim'][1] == 4096, f"dim hi={lm_space['dim'][1]}")
check("LM fitness extractor parse_average_loss still present + LM semantics",
      abs(C.parse_average_loss("step 1 loss 2.0\nstep 2 loss 4.0") - 3.0) < 1e-9)
check("LM param gate uses analytic estimate_params_for_config",
      C.estimate_params_for_config({'dim': 384, 'depth': 4, 'n_heads': 32, 'n_state': 32}, 'e88') > 0)
C.OBJECTIVE = 's5_acc@T128'

# ---------------------------------------------------------------------------
print("\n[2] Seed loading: load_anchor_configs seeds each arm at MODEL_CONFIG center")
check("anchor seed file exists", os.path.exists(ANCHOR_FILE), ANCHOR_FILE)
for arm, seed in SEEDS.items():
    anchors = C.load_anchor_configs(ANCHOR_FILE, arm)
    ok = len(anchors) == 1
    if ok:
        a = anchors[0]
        for k, v in seed.items():
            if k == 'lr':
                ok = ok and abs(float(a.get('lr', -1)) - v) < 1e-12
            else:
                ok = ok and a.get(k) == v
    check(f"{arm}: anchor == MODEL_CONFIG center", ok,
          f"{anchors[0] if anchors else None}")

# ---------------------------------------------------------------------------
print("\n[3] Param estimator keeps each arm at ~8M (REAL constructed count, ±10%)")
for arm, seed in SEEDS.items():
    n_real = C.real_param_count_s5(seed, arm)
    n_est = C.estimate_params_for_config(seed, arm)
    valid = C.is_valid_param_count(seed, arm, TARGET, TOL)
    check(f"{arm}: seed valid @8M±10% via REAL count", valid,
          f"real={n_real:,} ({n_real/1e6:.2f}M)  [analytic est={n_est/1e6:.2f}M]")

# ---------------------------------------------------------------------------
print("\n[4] Search space per arm (protocol §D.2): VARY dim/depth/n_heads/n_state/lr;"
      " E88 also linear_state & use_gate")
for arm in SEEDS:
    sp = C.get_search_space(arm)
    keys = set(sp)
    base = {'dim', 'depth', 'n_heads', 'n_state', 'lr'}
    check(f"{arm}: varies {sorted(base)}", base <= keys, f"space keys={sorted(keys)}")
    check(f"{arm}: depth band is 3-6", sp['depth'][0] == 3 and sp['depth'][1] == 6)
    check(f"{arm}: n_state restricted to {{16,32}} (e88_n_state snap)",
          sp['n_state'][2] == 'e88_n_state')
    check(f"{arm}: lr is log-scaled", sp['lr'][2] == 'log')
e88_sp = C.get_search_space('e88')
check("E88 ONLY: linear_state searched (binary)",
      e88_sp.get('linear_state', (0, 0, None))[2] == 'binary')
check("E88 ONLY: use_gate searched (binary)",
      e88_sp.get('use_gate', (0, 0, None))[2] == 'binary')
for arm in ('m2rnn', 'fla-gdn', 'm2rnn-paper'):
    sp = C.get_search_space(arm)
    check(f"{arm}: does NOT search linear_state/use_gate (E88-only knobs)",
          'linear_state' not in sp and 'use_gate' not in sp)

# encode(seed) -> decode round-trips the seed (CMA starts AT the seed center)
print("\n    Round-trip: encode(seed) -> decode == seed (CMA warm-starts at center)")
for arm, seed in SEEDS.items():
    x0 = C.encode_params(seed, arm)
    dec = C.decode_params(x0, arm)
    ok = (dec['dim'] == seed['dim'] and dec['depth'] == seed['depth']
          and dec['n_heads'] == seed['n_heads'] and dec['n_state'] == seed['n_state'])
    if arm == 'e88':
        ok = ok and dec['linear_state'] == seed['linear_state'] and dec['use_gate'] == seed['use_gate']
    check(f"{arm}: seed round-trips through CMA encoding", ok, f"decoded={dec}")

# Decode several CMA vectors and confirm the param gate genuinely filters and
# that surviving candidates are buildable ~8M models (no synthetic shortcut).
print("\n    Param re-match actually filters off-target candidates:")
import numpy as np  # noqa: E402
rng = np.random.default_rng(0)
for arm in SEEDS:
    ndim = C.get_search_dim(arm)
    n_valid = n_total = 0
    for _ in range(200):
        x = rng.uniform(0, 1, ndim)
        cfg = C.decode_params(x, arm)
        n_total += 1
        if C.is_valid_param_count(cfg, arm, TARGET, TOL):
            n_valid += 1
            nr = C.real_param_count_s5(cfg, arm)
            assert nr is not None and abs(nr - TARGET) / TARGET <= TOL
    check(f"{arm}: gate accepts a non-trivial subset and rejects the rest",
          0 < n_valid < n_total, f"{n_valid}/{n_total} random configs pass 8M±10%")

# ---------------------------------------------------------------------------
print("\n[5] Budget/convergence knobs wired (protocol §D.3)")
check("S5_STEPS == 5000 (per-candidate cap)", C.S5_STEPS == 5000)
check("S5_SEQ_LEN == 128", C.S5_SEQ_LEN == 128)
check("S5_BATCH == 32", C.S5_BATCH == 32)
check("S5_OPTIMIZER == schedulefree", C.S5_OPTIMIZER == 'schedulefree')
check("S5_WINDOW_STEPS == 1000 (final-window mean)", C.S5_WINDOW_STEPS == 1000)
check("S5_EVAL_LENGTHS == {128,256,512,1024}", C.S5_EVAL_LENGTHS == [128, 256, 512, 1024])
check("S5 search seed == 42", C.S5_SEED == 42)

# ---------------------------------------------------------------------------
print("\n[6] Fitness adapter wires to train_hybrid's REAL eval (no mock)")
for arm, seed in SEEDS.items():
    cmd, label = C.build_s5_train_command(seed, arm, '/tmp/_s5_preflight_eval', 0)
    s = ' '.join(cmd)
    ok = (cmd[1].endswith('experiments/expressivity_tasks/train_hybrid.py')
          and '--task' in cmd and 's5_permutation' in cmd
          and '--steps' in cmd and '5000' in cmd
          and '--seq_len' in cmd and '128' in cmd
          and '--batch_size' in cmd and '32' in cmd
          and '--optimizer' in cmd and 'schedulefree' in cmd
          and '--eval_lengths' in cmd)
    check(f"{arm}: command runs the real train_hybrid s5_permutation harness", ok)
    # layer level matches the 8M MODEL_CONFIG arm
    lvl_idx = cmd.index('--layer_pattern') + 1
    check(f"{arm}: layer level == {C.S5_LAYER_LEVEL[arm]}", cmd[lvl_idx] == C.S5_LAYER_LEVEL[arm])

# E88 command carries the searched BL-1 flags
cmd_e88, _ = C.build_s5_train_command(SEEDS['e88'], 'e88', '/tmp/_x', 0)
check("E88 command forwards --linear_state and --use_gate (searched knobs)",
      '--linear_state' in cmd_e88 and '--use_gate' in cmd_e88)
# Non-E88 arms do NOT carry the E88-only flags
cmd_m2, _ = C.build_s5_train_command(SEEDS['m2rnn'], 'm2rnn', '/tmp/_x', 0)
check("M2RNN command does NOT carry E88-only flags",
      '--linear_state' not in cmd_m2 and '--use_gate' not in cmd_m2)

# parse_s5_fitness reads the REAL eval_acc field train_hybrid writes. Verify
# against an actual committed train_hybrid output (real harness JSON, real
# eval_acc series) — NOT a fabricated file.
real_json = os.path.join(ROOT, 'paper', 'results', 'figure_4_hybrid',
                         'canon_pure_E88__fsm_tracking__seed42.json')
if os.path.exists(real_json):
    with open(real_json) as f:
        real_log = json.load(f)
    has_schema = (isinstance(real_log.get('steps'), list) and real_log['steps']
                  and 'eval_acc' in real_log['steps'][-1] and 'step' in real_log['steps'][-1])
    check("real train_hybrid JSON has the steps[].eval_acc schema the adapter reads",
          has_schema)
    fit, mean_acc, npts = C.parse_s5_fitness(real_json)
    # Independently recompute from the same real file over the final 1000 steps.
    steps = real_log['steps']
    mx = max(int(s['step']) for s in steps)
    win = [float(s['eval_acc']) for s in steps if int(s['step']) >= mx - C.S5_WINDOW_STEPS]
    exp_mean = sum(win) / len(win)
    check("parse_s5_fitness == 1 - mean(real eval_acc over final 1000 steps)",
          abs(fit - (1.0 - exp_mean)) < 1e-9 and abs(mean_acc - exp_mean) < 1e-9,
          f"fitness={fit:.4f} mean_acc={mean_acc:.4f} npts={npts}")
else:
    check("real train_hybrid JSON available for wiring test", False, real_json)

# NaN / divergence / missing-file -> worst fitness 1.0
check("missing candidate JSON -> fitness 1.0 (divergence rule)",
      C.parse_s5_fitness('/tmp/_s5_does_not_exist_zzz.json')[0] == 1.0)

# ---------------------------------------------------------------------------
print("\n[7] GPU pinning sanity (allowed set 2,3,4,5; never 0,1)")
for g in (2, 3, 4, 5):
    env = C.prepare_worker_env('e88', g)
    check(f"worker on gpu {g}: CUDA_VISIBLE_DEVICES=={g}", env['CUDA_VISIBLE_DEVICES'] == str(g))

print("\n" + "=" * 72)
if failures:
    print(f"RESULT: {len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED — no GPU used, no training run.")
sys.exit(0)
