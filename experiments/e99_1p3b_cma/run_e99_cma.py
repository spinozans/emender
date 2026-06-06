#!/usr/bin/env python3
"""E99 1.3B-class LM-CMA top-up driver (task run-e99-1-3b).

Runs the BOUNDED short-run CMA-ES search for the E99 PRIMARY candidate
``typed-gdn2-lm`` (typed Emender: native GDN-2 heads + nonlinear specialist mix,
the typed-gdn-2-head 5:1 ratio preserved at scale) on the REAL production LM
path (train.py + ndm/models/ladder_lm.py LadderLM, real Pile, schedule-free
AdamW). No mock data, no synthetic fitness.

This driver IMPORTS the production CMA harness ``scripts/cmaes_search_v2.py`` and
REGISTERS a new model_type ``typed-gdn2-lm`` into it WITHOUT editing the shared
file (other tasks/agents share that harness). All counting, output-dir scheme,
.done/pickle crash-resume, GPU-pool parallelism, AvgLoss/Final parsing, and
top-3 checkpoint retention are reused verbatim from the harness, so results line
up with docs/HANDOFF_E97_GDN2_CMAES_20260528.md.

Comparability mapping (see paper/review/E99_1P3B_LM_CMA96.md):
  - candidate counting: popsize 8 x 12 generations = 96 candidate-EVALS (not 96
    full trainings). Continues the prior popsize-8 counting (handoff: 8x8=64).
  - fields: dim,n_heads,n_state,depth,lr,batch_size (handoff field names).
  - fitness: AvgLoss over the train window (handoff convention) is the CMA
    objective; Final (last-100 avg) reported alongside. Held-out BPB computed for
    promoted configs in the bounded pilot, not per-candidate (budget).
  - budget: train_minutes=15 wallclock-matched (handoff), chunk_size=2048,
    params=1270M, data/tokenizer identical. fp32 (typed-gdn2-lm sanity dtype).

HARD STOPS (enforced in-process, logged to stop_reason.json):
  - total candidate-evals >= --max_total_evals (default 96)
  - cumulative TRAINING gpu-minutes >= --gpu_minute_ceiling (default 1440 = 24 GPU-h)
  - wallclock watchdog >= --wall_ceiling_minutes (safety, default 480)
  - cost-projection wrong: after gen 1, if measured median per-eval training
    minutes materially exceeds the 15-min projection (> --proj_tol x 15)
  - instability: if > --max_inf_frac of a generation's evals are inf/NaN/failed
"""
import os
import sys
import json
import time
import pickle
import argparse
import glob
from datetime import datetime

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import numpy as np
import cma
import cmaes_search_v2 as cs

# ---------------------------------------------------------------------------
# E99 typed-gdn2-lm FIXED architecture identity (the discovered form). These
# preserve the 5:1 GDN2:nonlinear head ratio at any n_heads and are NOT searched
# — only width/depth/head-shape/lr move. From candidate_configs.json (the
# typed-gdn-2-head CMA winner) and the wire-e99-e98 sanity.
# ---------------------------------------------------------------------------
TYPED_HEAD_TYPE_LOGITS = [3.9995, -1.9008, -0.9211, -2.8866, 2.4146]
TYPED_LAM_MAX = 1.585
TYPED_BETA_MAX = 2.747
TYPED_GDN_ALLOW_NEG_EIGVAL = 1
TYPED_KNOB_LR_MULT = 1.0
TYPED_EXPANSION = 1.0

# Anchor (warm start) = the sanity/discovered 1.3B-class operating point.
ANCHOR = dict(dim=3072, depth=22, n_heads=96, n_state=32, lr=9.95e-4, batch_size=2)

# 5D search space (batch_size fixed -> see --fixed_batch_size). Centered on the
# anchor; param-tolerance filter keeps everything at 1.27B-class.
TYPED_SEARCH_SPACE = {
    'dim':        (2048, 3584, 'int_mult128', 'Model/residual dim'),
    'n_heads':    (48, 160, 'int', 'Total typed heads (5:1 GDN2:nonlin preserved by fixed logits)'),
    'n_state':    (16, 64, 'e88_n_state', 'Per-head state/head_dim (snaps to {16,32})'),
    'depth':      (16, 30, 'int', 'Number of layers'),
    'lr':         (3e-4, 1.5e-3, 'log', 'Learning rate'),
    'batch_size': (1, 8, 'int_log', 'Batch size (fixed at 2 for fp32 1.3B memory budget)'),
}

# Param-count cache: build ONE TypedHeadMixtureLayer on CPU and count (matches
# the measured 1.2775B anchor to 1.4e-5). Cached by (dim,n_heads,n_state).
_PARAM_CACHE = {}


def typed_layer_params(dim, n_heads, n_state):
    key = (int(dim), int(n_heads), int(n_state))
    if key in _PARAM_CACHE:
        return _PARAM_CACHE[key]
    import torch
    from ndm.models.typed_head_mixture import TypedHeadMixtureLayer
    layer = TypedHeadMixtureLayer(
        dim=int(dim), n_heads=int(n_heads), n_state=int(n_state),
        head_type_logits=TYPED_HEAD_TYPE_LOGITS,
        gdn_allow_neg_eigval=bool(TYPED_GDN_ALLOW_NEG_EIGVAL),
        lam_max=TYPED_LAM_MAX, beta_max=TYPED_BETA_MAX)
    lp = sum(p.numel() for p in layer.parameters())
    del layer
    _PARAM_CACHE[key] = lp
    return lp


def typed_total_params(params, vocab_size):
    dim = int(params['dim'])
    depth = int(params['depth'])
    lp = typed_layer_params(dim, params['n_heads'], params['n_state'])
    # LadderLM: tied embedding (vocab*dim) + depth*(RMSNorm(dim)+layer) + final RMSNorm
    return vocab_size * dim + depth * lp + dim * (depth + 1)


# ---- Register typed-gdn2-lm into the harness (no edits to the shared file) ----
def register_typed_gdn2_lm():
    cs.SEARCH_SPACES['typed-gdn2-lm'] = dict(TYPED_SEARCH_SPACE)

    _orig_estimate = cs.estimate_params_for_config

    def estimate(params, model_type):
        if model_type == 'typed-gdn2-lm':
            return typed_total_params(params, cs.PARAM_VOCAB_SIZE)
        return _orig_estimate(params, model_type)
    cs.estimate_params_for_config = estimate

    _orig_build = cs.build_train_command

    def build(params, model_type, train_minutes, output_dir):
        if model_type != 'typed-gdn2-lm':
            return _orig_build(params, model_type, train_minutes, output_dir)
        actual_params = typed_total_params(params, cs.PARAM_VOCAB_SIZE)
        bs = params.get('batch_size', 2)
        cmd = [
            sys.executable, os.path.join(_ROOT, 'train.py'),
            '--data', cs.DATA_PATH,
            '--level', 'typed-gdn2-lm',
            '--dim', str(int(params['dim'])),
            '--depth', str(int(params['depth'])),
            '--n_heads', str(int(params['n_heads'])),
            '--n_state', str(int(params['n_state'])),
            '--expansion', str(TYPED_EXPANSION),
            '--head_type_logits', ','.join(str(x) for x in TYPED_HEAD_TYPE_LOGITS),
            '--gdn_allow_neg_eigval', str(TYPED_GDN_ALLOW_NEG_EIGVAL),
            '--lam_max', str(TYPED_LAM_MAX),
            '--beta_max', str(TYPED_BETA_MAX),
            '--knob_lr_mult', str(TYPED_KNOB_LR_MULT),
            '--lr', str(params.get('lr', 9.95e-4)),
            # NO --bf16: typed-gdn2-lm runs fp32 (the wire-e99-e98 sanity dtype).
            '--batch_size', str(bs),
            '--chunk_size', str(cs.CHUNK_SIZE),
            '--train_minutes', str(train_minutes),
            '--output', output_dir,
            '--optimizer', 'schedulefree',
            '--seed', '42',
            '--save_every', '999999',
            '--keep_checkpoints', '1',
        ]
        if cs.TOKENIZER_NAME:
            cmd.extend(['--tokenizer', cs.TOKENIZER_NAME])
        return cmd, actual_params
    cs.build_train_command = build


# ---------------------------------------------------------------------------
# Controlled CMA loop with explicit budget guards. Reuses cs.evaluate_batch (so
# .done files / output layout / parsing are identical to the production harness)
# but owns the generation loop to enforce hard stops.
# ---------------------------------------------------------------------------
def eval_training_minutes(eval_dir):
    """Recover REAL training minutes for one eval from its stdout (elapsed_h)."""
    out = os.path.join(eval_dir, 'stdout.txt')
    if not os.path.exists(out):
        return None
    last_h = None
    try:
        with open(out) as f:
            for line in f:
                if line.startswith('step') and 'elapsed_h' in line:
                    import re
                    m = re.search(r'elapsed_h\s+([0-9.]+)', line)
                    if m:
                        last_h = float(m.group(1))
    except OSError:
        return None
    return last_h * 60.0 if last_h is not None else None


def run(args):
    register_typed_gdn2_lm()
    model_type = 'typed-gdn2-lm'

    # Harness globals (mirror the handoff conventions).
    cs.TOKENIZER_NAME = args.tokenizer
    cs.PARAM_VOCAB_SIZE = cs.resolve_vocab_size(args.tokenizer)
    cs.DATA_PATH = args.data
    cs.CHUNK_SIZE = args.chunk_size
    cs.SKIP_MEMORY_PROBE = True   # fix bs -> no separate probe; OOM step-down fallback
    cs.PARAM_TOLERANCE = args.param_tolerance
    cs.DEFAULT_GPUS = [int(g) for g in args.gpus.split(',')]

    target_params = int(args.params.lower().replace('m', '000000').replace('b', '000000000'))
    fixed_params = {'batch_size': args.fixed_batch_size}

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    cs.GPU_FILE = os.path.abspath(args.gpu_file) if args.gpu_file else None
    if cs.GPU_FILE and not os.path.exists(cs.GPU_FILE):
        os.makedirs(os.path.dirname(cs.GPU_FILE), exist_ok=True)
        with open(cs.GPU_FILE, 'w') as f:
            f.write(args.gpus + '\n')

    state_file = os.path.join(output_dir, 'cmaes_state.pkl')
    stop_file = os.path.join(output_dir, 'stop_reason.json')

    # ---- Resume: recover completed evals + CMA state ----
    recovered, max_eval_id = cs.recover_completed_evals(output_dir)
    all_results = list(recovered)
    eval_counter = (max_eval_id + 1) if recovered else 0
    start_gen = 0
    es = None
    best_loss, best_params = float('inf'), None
    if recovered:
        print(f"RESUME: {len(recovered)} completed evals (max eval_id={max_eval_id})")
        for r in recovered:
            if r.get('loss', float('inf')) < best_loss:
                best_loss, best_params = r['loss'], r['params']
    if os.path.exists(state_file):
        try:
            with open(state_file, 'rb') as f:
                st = pickle.load(f)
            es = st['es']
            start_gen = st['gen'] + 1
            best_loss = st.get('best_loss', best_loss)
            best_params = st.get('best_params', best_params)
            eval_counter = max(eval_counter, st.get('eval_counter', eval_counter))
            print(f"RESUME: CMA state restored, continuing from gen {start_gen + 1}")
        except Exception as e:
            print(f"WARNING: could not load CMA state ({e}); fresh CMA")
            es = None

    if es is None:
        x0 = cs.encode_params(ANCHOR, model_type, fixed_params)
        es = cma.CMAEvolutionStrategy(x0, args.sigma, {
            'popsize': args.popsize, 'bounds': [0, 1], 'seed': 42, 'verbose': -1,
        })

    def write_stop(reason, extra=None):
        rec = {'reason': reason, 'evals_completed': len(all_results),
               'wallclock_utc': datetime.utcnow().isoformat() + 'Z'}
        if extra:
            rec.update(extra)
        with open(stop_file, 'w') as f:
            json.dump(rec, f, indent=2, default=str)
        print(f"\n*** HARD STOP: {reason} | evals_completed={len(all_results)} ***")

    wall_start = time.time()
    gen_train_minutes = []   # measured per-eval training minutes (for projection gate)

    for gen in range(start_gen, args.max_generations):
        # ---- budget guards BEFORE spending a generation ----
        if len(all_results) >= args.max_total_evals:
            write_stop('reached_max_total_evals', {'max_total_evals': args.max_total_evals})
            break
        cum_train_min = sum(m for m in gen_train_minutes if m) if gen_train_minutes else 0.0
        if cum_train_min >= args.gpu_minute_ceiling:
            write_stop('gpu_minute_ceiling', {'cum_training_gpu_minutes': cum_train_min,
                                              'ceiling': args.gpu_minute_ceiling})
            break
        wall_min = (time.time() - wall_start) / 60.0
        if wall_min >= args.wall_ceiling_minutes:
            write_stop('wall_ceiling', {'wall_minutes': wall_min})
            break

        # ---- rejection-sample popsize valid (param-matched) configs ----
        target_evals = min(args.popsize, args.max_total_evals - len(all_results))
        valid_solutions, valid_configs = [], []
        total_generated = 0
        for _attempt in range(cs.CMAES_MAX_VALID_ATTEMPTS):
            batch = es.ask(number=target_evals * 2)
            total_generated += len(batch)
            for sol in batch:
                if len(valid_solutions) >= target_evals:
                    break
                cfg = cs.decode_params(sol, model_type, fixed_params)
                if cs.is_valid_param_count(cfg, model_type, target_params):
                    if not any(np.allclose(sol, vs) for vs in valid_solutions):
                        valid_solutions.append(sol)
                        valid_configs.append(cfg)
            if len(valid_solutions) >= target_evals:
                break

        if not valid_configs:
            write_stop('no_valid_configs', {'gen': gen, 'total_generated': total_generated})
            break

        current_gpus = cs.get_available_gpus()
        print(f"\n=== Generation {gen + 1}/{args.max_generations}: "
              f"{len(valid_configs)} valid configs, {len(current_gpus)} GPUs, "
              f"eval_counter={eval_counter} ===")

        gen_results = cs.evaluate_batch(valid_configs, model_type, args.train_minutes,
                                        output_dir, current_gpus, start_eval_id=eval_counter)
        eval_counter += len(gen_results)
        all_results.extend(gen_results)

        # ---- measured training minutes (projection + instability gates) ----
        inf_count = 0
        for r in gen_results:
            if not (r.get('loss', float('inf')) < 10.0) or not r.get('success', False):
                inf_count += 1
            tm = eval_training_minutes(os.path.join(output_dir, f"eval_{r.get('eval_id')}"))
            if tm:
                gen_train_minutes.append(tm)

        inf_frac = inf_count / max(1, len(gen_results))
        if inf_frac > args.max_inf_frac:
            write_stop('instability', {'gen': gen, 'inf_frac': inf_frac,
                                       'inf_count': inf_count, 'gen_size': len(gen_results)})
            break

        # cost-projection gate (after first completed generation)
        if gen == start_gen and gen_train_minutes:
            med = float(np.median(gen_train_minutes))
            if med > args.proj_tol * args.train_minutes:
                write_stop('cost_projection_wrong',
                           {'median_train_minutes': med, 'projection_minutes': args.train_minutes,
                            'tol_x': args.proj_tol})
                break

        # ---- CMA tell on the valid solutions actually evaluated ----
        fitnesses = [r['loss'] for r in gen_results]
        if len(fitnesses) >= 2:
            try:
                es.tell(valid_solutions[:len(fitnesses)], fitnesses)
            except ValueError as e:
                print(f"  WARNING: CMA update skipped: {e}")

        gen_best = min(fitnesses)
        if gen_best < best_loss:
            best_loss = gen_best
            best_params = valid_configs[fitnesses.index(gen_best)]
            print(f"  *** NEW BEST: {best_loss:.4f} | {cs.format_params(best_params)} ***")

        cs.retain_top_checkpoints(output_dir, all_results, top_n=3)

        # ---- persist state + per-gen snapshot (crash-resume + audit) ----
        with open(state_file, 'wb') as f:
            pickle.dump({'es': es, 'gen': gen, 'best_loss': best_loss,
                         'best_params': best_params, 'eval_counter': eval_counter}, f)
        snap = {
            'gen': gen,
            'wallclock_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'popsize': args.popsize, 'n_valid': len(valid_configs),
            'total_generated': total_generated,
            'gen_best_loss': float(gen_best),
            'gen_fitnesses': [float(x) for x in fitnesses],
            'best_loss_so_far': float(best_loss),
            'best_params_so_far': best_params,
            'evals_completed': len(all_results),
            'cum_training_gpu_minutes': sum(m for m in gen_train_minutes if m),
            'median_eval_train_minutes': float(np.median(gen_train_minutes)) if gen_train_minutes else None,
            'inf_frac_this_gen': inf_frac,
            'sigma': float(es.sigma),
        }
        with open(os.path.join(output_dir, 'generations.jsonl'), 'a') as f:
            f.write(json.dumps(snap, default=str) + '\n')
        print(f"  gen_best={gen_best:.4f} overall_best={best_loss:.4f} "
              f"evals={len(all_results)} cum_train_min={snap['cum_training_gpu_minutes']:.0f}")
    else:
        write_stop('completed_all_generations', {'generations': args.max_generations})

    # ---- final results.json (extends handoff summary columns) ----
    all_results.sort(key=lambda x: x.get('loss', float('inf')))
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'model': model_type,
            'search': 'E99 typed-gdn2-lm 1.3B-class LM-CMA top-up (run-e99-1-3b)',
            'popsize': args.popsize, 'generations_target': args.max_generations,
            'evals_completed': len(all_results),
            'best_loss': all_results[0]['loss'] if all_results else None,
            'best_params': all_results[0]['params'] if all_results else None,
            'train_minutes_per_eval': args.train_minutes,
            'chunk_size': args.chunk_size, 'dtype': 'fp32',
            'fixed_knobs': {'head_type_logits': TYPED_HEAD_TYPE_LOGITS,
                            'lam_max': TYPED_LAM_MAX, 'beta_max': TYPED_BETA_MAX,
                            'gdn_allow_neg_eigval': TYPED_GDN_ALLOW_NEG_EIGVAL,
                            'knob_lr_mult': TYPED_KNOB_LR_MULT},
            'all_results': [{'eval_id': r.get('eval_id'), 'params': r['params'],
                             'actual_params': r.get('actual_params'),
                             'loss': r['loss'], 'final_loss': r.get('final_loss'),
                             'batch_size': r.get('batch_size'),
                             'success': r.get('success'), 'error': r.get('error')}
                            for r in all_results],
        }, f, indent=2, default=str)
    print(f"\nResults -> {results_file}  ({len(all_results)} evals)")
    return all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--gpu_file', default=None)
    ap.add_argument('--data', default='/home/erikg/elman/data/pile.txt')
    ap.add_argument('--tokenizer', default='p50k_base')
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--params', default='1270M')
    ap.add_argument('--param_tolerance', type=float, default=0.10)
    ap.add_argument('--popsize', type=int, default=8)
    ap.add_argument('--max_generations', type=int, default=12)
    ap.add_argument('--sigma', type=float, default=0.14)
    ap.add_argument('--train_minutes', type=float, default=15.0)
    ap.add_argument('--fixed_batch_size', type=int, default=2)
    # hard-stop guards
    ap.add_argument('--max_total_evals', type=int, default=96)
    ap.add_argument('--gpu_minute_ceiling', type=float, default=1440.0)  # 24 GPU-h training
    ap.add_argument('--wall_ceiling_minutes', type=float, default=480.0)
    ap.add_argument('--proj_tol', type=float, default=1.6)  # median eval > 1.6x15min -> stop
    ap.add_argument('--max_inf_frac', type=float, default=0.5)
    args = ap.parse_args()
    run(args)


if __name__ == '__main__':
    main()
