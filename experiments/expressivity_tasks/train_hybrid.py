"""Train a HybridLadderLM (per-layer architecture) on a task. Mirrors
train_task.py but uses HybridLadderLM directly so we can pass layer_pattern.

Usage:
    python train_hybrid.py --task parity --layer_pattern E88 fla-gdn \\
        --dim 128 --depth 4 --steps 500 --label hybrid_parity
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from ndm.models.hybrid_ladder import HybridLadderLM
from experiments.expressivity_tasks.tasks import ALL_TASKS


def evaluate(model, task, B, T, n_batches, rng, device):
    model.eval()
    correct = total = 0
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            inp, tgt, mask = task.generate_batch(B, T, rng)
            x = torch.from_numpy(inp).to(device)
            y = torch.from_numpy(tgt).to(device)
            m = torch.from_numpy(mask).to(device)
            use_autocast = device == 'cuda' and not getattr(model, 'disable_autocast', False)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
                logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += ((preds == y) & m).sum().item()
            total += m.sum().item()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                    y.view(-1), reduction='none').view_as(m)
            losses.append((loss * m).sum().item() / max(m.sum().item(), 1))
    return correct / max(total, 1), float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=list(ALL_TASKS.keys()))
    ap.add_argument('--layer_pattern', nargs='+', required=True,
                    help='List of layer levels, e.g. E88 fla-gdn')
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--n_heads', type=int, default=4)
    ap.add_argument('--n_state', type=int, default=16)
    ap.add_argument('--rank', type=int, default=None)
    ap.add_argument('--expansion', type=float, default=1.0)
    ap.add_argument('--use_triton_e88', action='store_true',
                    help='Route E88 layers through the Triton fwd/bwd kernels.')
    # E88 structural (BL-1) knobs. Default None = use the E88 layer's own
    # constructor defaults (unchanged behavior for run_separation_suite and the
    # §6 probes). When set, they are forwarded per-layer to E88-family layers so
    # the S5-symmetric CMA-ES can search them (paper/review/S5_SYMMETRIC_PROTOCOL.md
    # §D.2: linear_state and use_gate become searched knobs for E88).
    ap.add_argument('--linear_state', type=int, default=None, choices=[0, 1],
                    help='E88 state nonlinearity: 0=tanh, 1=linear. Default: '
                         "layer default (tanh). Forwarded only to E88-family layers.")
    ap.add_argument('--state_activation', type=str, default=None,
                    choices=['tanh', 'identity', 'linear', 'relu', 'softplus'],
                    help='E88 state nonlinearity f in S=f(decay*S+outer): tanh '
                         '(saturating default), identity/linear (affine), relu or '
                         'softplus (NON-SATURATING, unbounded |S| -> can count). '
                         'Forwarded only to E88-family layers; runs the fp32 PyTorch '
                         'reference recurrence for relu/softplus.')
    ap.add_argument('--use_gate', type=int, default=None, choices=[0, 1],
                    help='E88 output gate: 0=off, 1=on. Default: HybridLadderLM '
                         'default. Forwarded only to E88-family layers.')
    ap.add_argument('--decay_mode', type=str, default=None,
                    choices=['mamba', 'simple', 'none', 'constant'],
                    help='E88 recurrence decay mode. mamba=input-dependent '
                         'exp decay (default); constant=learned per-head '
                         'constant (input-INDEPENDENT transition, eigenvalues '
                         'in (0,1)); none=identity (eigenvalue 1); '
                         'simple=input-dependent sigmoid. Forwarded only to '
                         'E88-family layers. Used by the E5 input-dependence '
                         'ablation (paper/review/E5_ABLATE_INPUTDEP.md).')
    # Eigenvalue-causal-test knobs (ARM A / ARM B). Default None = unchanged.
    ap.add_argument('--gdn_allow_neg_eigval', type=int, default=None, choices=[0, 1],
                    help='ARM A: fla GatedDeltaNet allow_neg_eigval (beta*=2 -> beta in (0,2) '
                         '-> along-key eigenvalue g(1-beta) can go negative). fla-gdn layers only.')
    ap.add_argument('--e88_pos_eigval_clamp', type=int, default=None, choices=[0, 1],
                    help='ARM B: clamp E88 along-key eigenvalue >=0 by moving decay onto the '
                         'whole operator decay*(I-kk^T) (was decay*I-kk^T). E88-family layers only.')
    ap.add_argument('--e88_raw_write', type=int, default=None, choices=[0, 1],
                    help='ARM B secondary: E88 raw_write (drop delta-correction; A_t=decay*I, '
                         'all eigenvalues = decay > 0). E88-family layers only.')
    ap.add_argument('--m2rnn_q_heads', type=int, default=None)
    ap.add_argument('--m2rnn_k_heads', type=int, default=None)
    ap.add_argument('--m2rnn_v_heads', type=int, default=None)
    ap.add_argument('--m2rnn_f_heads', type=int, default=None)
    ap.add_argument('--m2rnn_g_heads', type=int, default=None)
    ap.add_argument('--m2rnn_weight_heads', type=int, default=None)
    ap.add_argument('--m2rnn_normalize_qk', action='store_true',
                    help='L2-normalize M2RNN query/key vectors before recurrence')
    ap.add_argument('--m2rnn_no_residual', action='store_true',
                    help='Disable M2RNN D*v direct residual path')
    ap.add_argument('--m2rnn_freeze_state_weight', action='store_true',
                    help='Keep M2RNN state_weight fixed at identity')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--seq_len', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'schedulefree'])
    # UnifiedCell (e98-cma / unified-* / e98-*) meta-config knobs (cma-capability).
    # Default None = use the layer pattern's own defaults (unchanged behaviour for
    # all prior e98/unified sweeps). Forwarded only to UnifiedCell-family layers.
    ap.add_argument('--lam_max', type=float, default=None,
                    help='UnifiedCell lambda gain cap (free gain needs >=1.3 for latch).')
    ap.add_argument('--beta_max', type=float, default=None,
                    help='UnifiedCell beta (delta-correction) cap.')
    ap.add_argument('--igain_max', type=float, default=None,
                    help='UnifiedCell input write-gain cap.')
    ap.add_argument('--head_type_logits', type=str, default=None,
                    help='typed-gdn2: comma-sep 5 unconstrained logits over head '
                         'types [gdn2_recall,e97_track,count,latch,nonlin]; '
                         'softmax+largest-remainder -> per-type head counts.')
    ap.add_argument('--corner_mixture', type=str, default=None,
                    help='Comma-separated 4 head fractions [track,count,latch,nonlin] '
                         'for spread-init/fixed_pop placement. e.g. "0.4,0.2,0.2,0.2". '
                         'Default None = equal 25/25/25/25 round-robin.')
    ap.add_argument('--shell_state_nonlin', type=str, default=None,
                    help='typed-gdn2: bounded nonlinear-in-time state map for the '
                         'gdn2_nonlin_shell control head (e.g. "tanh"). Forwarded '
                         'only to typed-gdn2 layers; ignored when no shell heads.')
    ap.add_argument('--shell_state_chunk', type=int, default=None,
                    help='typed-gdn2: chunk size for the fused shell scan (e.g. 64).')
    ap.add_argument('--knob_lr_mult', type=float, default=1.0,
                    help='LEARNABILITY intervention #2: multiply the base LR for '
                         'the recurrence knobs (lam_raw/beta_raw/igain_raw/'
                         'gamma_raw of every UnifiedCellLayer) by this factor, '
                         'placing them in a SEPARATE optimizer param-group so the '
                         'knobs actually move while projections stay at base LR. '
                         '1.0 = single group (no split).')
    ap.add_argument('--spec_reg', type=str, default=None,
                    choices=['pull', 'anticenter', 'coverage', 'pull_cov', 'anticenter_cov'],
                    help='SPECIALIZATION-PRESSURE regularizer (specialization-study): '
                         'add a penalty on every UnifiedCellLayer that FORCES per-head '
                         'knobs onto the four corners. pull=(a) pull-to-nearest-corner, '
                         'anticenter=(b) repel-center/reward-corner, coverage=(c) '
                         'population diversity, *_cov = combine per-head + coverage. '
                         'Default None = no regularizer.')
    ap.add_argument('--spec_reg_weight', type=float, default=1.0,
                    help='Weight on the specialization regularizer (swept).')
    ap.add_argument('--spec_reg_anneal', type=float, default=0.5,
                    help='Fraction of training over which the reg weight ramps '
                         'linearly 0 -> spec_reg_weight (then held). 0 = full weight '
                         'from step 0. Default 0.5 (anneal in over first half so the '
                         'task is learned before specialization pressure peaks).')
    ap.add_argument('--curriculum', type=str, default=None,
                    help='Optional length curriculum (secondary lever): comma-'
                         'separated train seq_len schedule, e.g. 128,256,512,1024. '
                         'Steps are split evenly across stages; eval still uses '
                         '--eval_lengths. Default None = fixed --seq_len.')
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--label', required=True)
    ap.add_argument('--output_dir', default='experiments/expressivity_tasks/results')
    ap.add_argument('--eval_lengths', type=int, nargs='+', default=None,
                    help='If set, after training, eval at each of these T values '
                         '(Délétang length-extrapolation protocol). Records per-T '
                         "accuracy under log['length_extrap'].")
    ap.add_argument('--eval_interval', type=int, default=None,
                    help='Steps between S5 eval logging. Default: max(50, steps//20). '
                         'Set to 100 for dense candidate-budget calibration curves.')
    ap.add_argument('--eval_lengths_n_batches', type=int, default=8,
                    help='Number of eval batches per length in --eval_lengths.')
    ap.add_argument('--disable_autocast', action='store_true',
                    help='Run forward passes without bf16 autocast. Useful for '
                         'exact algorithmic tasks and fair comparison to M2RNN, '
                         'whose current expressivity path already disables autocast.')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Build task
    task_kwargs = {}
    if args.task == 'modular_counter':       task_kwargs['K'] = args.K
    elif args.task == 'dyck':                 task_kwargs['max_depth'] = args.K
    elif args.task == 'dyck2':                task_kwargs['max_depth'] = args.K
    elif args.task == 'fsm_tracking':         task_kwargs['n_states'] = args.K
    elif args.task == 'selective_copy':       task_kwargs['n_to_copy'] = args.K
    elif args.task == 'assoc_recall':         task_kwargs['n_pairs'] = args.K
    elif args.task in ('overwrite_recall', 'reset_recall'):
        task_kwargs['n_keys'] = args.K
    elif args.task == 'keyed_fsm_memory':
        task_kwargs['n_keys'] = args.K
        task_kwargs['n_states'] = args.K
    elif args.task == 'flag_hold_recall':
        task_kwargs['n_keys'] = args.K
    elif args.task == 'mixed_probe':
        task_kwargs['n_keys'] = args.K
    task = ALL_TASKS[args.task](**task_kwargs)
    print(f"Task: {task.name}, vocab_size={task.vocab_size}", flush=True)

    # Build hybrid model
    m2_kwargs = {}
    if args.m2rnn_q_heads is not None: m2_kwargs['num_q_heads'] = args.m2rnn_q_heads
    if args.m2rnn_k_heads is not None: m2_kwargs['num_k_heads'] = args.m2rnn_k_heads
    if args.m2rnn_v_heads is not None: m2_kwargs['num_v_heads'] = args.m2rnn_v_heads
    if args.m2rnn_f_heads is not None: m2_kwargs['num_f_heads'] = args.m2rnn_f_heads
    if args.m2rnn_g_heads is not None: m2_kwargs['num_g_heads'] = args.m2rnn_g_heads
    if args.m2rnn_weight_heads is not None: m2_kwargs['num_weight_heads'] = args.m2rnn_weight_heads
    if args.m2rnn_normalize_qk: m2_kwargs['normalize_qk'] = True
    if args.m2rnn_no_residual: m2_kwargs['use_residual'] = False
    if args.m2rnn_freeze_state_weight: m2_kwargs['state_weight_trainable'] = False
    # M2RNN raw-write state-nonlinearity ablation: --linear_state drops the tanh
    # in Z = tanh(h W + k v^T) -> Z = h W + k v^T (analogue of E88 linear_state).
    if args.linear_state is not None:
        m2_kwargs['linear_state'] = bool(args.linear_state)
    # E88-family structural overrides (only applied when explicitly passed).
    e88_kwargs = {}
    if args.linear_state is not None:
        e88_kwargs['linear_state'] = bool(args.linear_state)
    if args.state_activation is not None:
        e88_kwargs['state_activation'] = args.state_activation
    if args.use_gate is not None:
        e88_kwargs['use_gate'] = bool(args.use_gate)
    if args.decay_mode is not None:
        e88_kwargs['decay_mode'] = args.decay_mode
    if args.e88_pos_eigval_clamp is not None:
        e88_kwargs['pos_eigval_clamp'] = bool(args.e88_pos_eigval_clamp)
    if args.e88_raw_write is not None:
        e88_kwargs['raw_write'] = bool(args.e88_raw_write)
    # fla-gdn overrides (only applied when explicitly passed).
    gdn_kwargs = {}
    if args.gdn_allow_neg_eigval is not None:
        gdn_kwargs['allow_neg_eigval'] = bool(args.gdn_allow_neg_eigval)
    # UnifiedCell (e98-cma / unified-* / e98-*) meta-config overrides.
    unified_kwargs = {}
    if args.lam_max is not None:
        unified_kwargs['lam_max'] = args.lam_max
    if args.beta_max is not None:
        unified_kwargs['beta_max'] = args.beta_max
    if args.igain_max is not None:
        unified_kwargs['igain_max'] = args.igain_max
    if args.corner_mixture is not None:
        unified_kwargs['corner_mixture'] = [float(x) for x in args.corner_mixture.split(',') if x.strip()]

    # typed-gdn2 (typed-gdn-2-head): native typed-head mixture meta-config. The
    # per-type logits are the CMA search variable; lam_max/beta_max freeze the
    # placed corner personalities at the validated operating points.
    typed_kwargs = {}
    if args.head_type_logits is not None:
        typed_kwargs['head_type_logits'] = [float(x) for x in args.head_type_logits.split(',') if x.strip()]
    if args.lam_max is not None:
        typed_kwargs['lam_max'] = args.lam_max
    if args.beta_max is not None:
        typed_kwargs['beta_max'] = args.beta_max
    if args.igain_max is not None:
        typed_kwargs['igain_max'] = args.igain_max
    if args.shell_state_nonlin is not None:
        typed_kwargs['shell_state_nonlin'] = args.shell_state_nonlin
    if args.shell_state_chunk is not None:
        typed_kwargs['shell_state_chunk'] = args.shell_state_chunk

    def _is_unified_level(level):
        return isinstance(level, str) and (level.startswith('e98') or level.startswith('unified'))

    def _layer_kw(level):
        if level in ('m2rnn', 'm2rnn-paper'):
            return dict(m2_kwargs)
        # E97 = E88FLAHybrid(use_split_edit=True); it is part of the E88 family
        # and accepts the same structural overrides (raw_write, state_activation,
        # decay_mode, ...). The level token 'E97' does not start with 'E88', so
        # forward e88_kwargs to it explicitly — otherwise --e88_raw_write and
        # --state_activation are silently dropped for the E97 split-gate arm.
        if isinstance(level, str) and (level.startswith('E88') or level == 'E97'):
            return dict(e88_kwargs)
        if level == 'fla-gdn':
            return dict(gdn_kwargs)
        if level == 'typed-gdn2':
            return dict(typed_kwargs)
        if _is_unified_level(level):
            return dict(unified_kwargs)
        return {}

    layer_kwargs = [_layer_kw(level) for level in args.layer_pattern]
    model = HybridLadderLM(
        vocab_size=task.vocab_size,
        dim=args.dim, depth=args.depth,
        layer_pattern=args.layer_pattern,
        layer_kwargs=layer_kwargs,
        n_state=args.n_state, n_heads=args.n_heads,
        expansion=args.expansion,
        rank=args.rank,
        use_triton_e88=args.use_triton_e88,
    ).to(device)
    if args.disable_autocast:
        model.disable_autocast = True
    print(f"Pattern: {model.actual_pattern}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}", flush=True)

    # typed-gdn2: dump the deterministic per-type head allocation (same for every
    # typed layer) so the report can show whether CMA picks a balanced population
    # or collapses toward GDN-2.
    typed_alloc = None
    for layer in model.layers:
        if hasattr(layer, 'head_alloc'):
            typed_alloc = layer.head_alloc()
            break
    if typed_alloc is not None:
        print(f"Typed-head alloc: {typed_alloc['counts']}", flush=True)

    # Build param groups. With --knob_lr_mult != 1, the recurrence knobs
    # (lam/beta/igain/gamma raw of every UnifiedCellLayer) get a SEPARATE group at
    # a higher LR; everything else stays at the base LR.
    KNOB_SUFFIXES = ('lam_raw', 'beta_raw', 'igain_raw', 'gamma_raw')
    knob_params, base_params, knob_names = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.endswith(s) for s in KNOB_SUFFIXES):
            knob_params.append(p); knob_names.append(name)
        else:
            base_params.append(p)
    use_knob_group = args.knob_lr_mult != 1.0 and len(knob_params) > 0
    if use_knob_group:
        param_groups = [
            {'params': base_params, 'lr': args.lr},
            {'params': knob_params, 'lr': args.lr * args.knob_lr_mult},
        ]
        print(f"Knob-LR group: {len(knob_params)} knob params at lr="
              f"{args.lr * args.knob_lr_mult:.2e} ({args.knob_lr_mult}x base); "
              f"{len(base_params)} base params at lr={args.lr:.2e}", flush=True)
    else:
        param_groups = model.parameters()

    if args.optimizer == 'schedulefree':
        import schedulefree
        optimizer = schedulefree.AdamWScheduleFree(
            param_groups, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
        print(f"Using schedule-free AdamW (lr={args.lr})", flush=True)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0.01)
        print(f"Using vanilla AdamW (lr={args.lr})", flush=True)

    log = {'task': task.name, 'pattern': model.actual_pattern, 'dim': args.dim, 'depth': args.depth,
           'seq_len': args.seq_len, 'batch_size': args.batch_size, 'lr': args.lr,
           'seed': args.seed, 'params': n_params,
           'disable_autocast': bool(args.disable_autocast),
           'use_triton_e88': bool(args.use_triton_e88),
           'linear_state': args.linear_state,
           'state_activation': args.state_activation,
           'use_gate': args.use_gate,
           'decay_mode': args.decay_mode,
           'gdn_allow_neg_eigval': args.gdn_allow_neg_eigval,
           'e88_pos_eigval_clamp': args.e88_pos_eigval_clamp,
           'e88_raw_write': args.e88_raw_write,
           'knob_lr_mult': float(args.knob_lr_mult),
           'lam_max': args.lam_max,
           'beta_max': args.beta_max,
           'igain_max': args.igain_max,
           'corner_mixture': args.corner_mixture,
           'head_type_logits': args.head_type_logits,
           'typed_alloc': typed_alloc,
           'spec_reg': args.spec_reg,
           'spec_reg_weight': float(args.spec_reg_weight),
           'spec_reg_anneal': float(args.spec_reg_anneal),
           'curriculum': args.curriculum,
           'random_baseline_acc': task.random_baseline_acc(),
           'steps': []}

    # Snapshot per-head knobs at INIT (before any optimizer step) so drift from
    # spread-init can be measured directly against the trained values.
    init_knobs = []
    for li, layer in enumerate(model.layers):
        if hasattr(layer, 'knob_values'):
            kv = layer.knob_values()
            init_knobs.append({
                'layer': li,
                'lambda': kv['lambda'].tolist(), 'beta': kv['beta'].tolist(),
                'igain': kv['igain'].tolist(), 'gamma': kv['gamma'].tolist(),
                'eig_along': kv['eig_along'].tolist(),
            })
    if init_knobs:
        log['unified_knobs_init'] = init_knobs

    # Specialization-pressure regularizer (specialization-study): collect every
    # layer that exposes specialization_loss (the UnifiedCellLayers). The penalty
    # ramps in over --spec_reg_anneal of training so the task is learned before
    # the corner-pressure peaks.
    spec_layers = [l for l in model.layers if hasattr(l, 'specialization_loss')] if args.spec_reg else []
    if args.spec_reg:
        print(f"Spec-reg: variant={args.spec_reg} weight={args.spec_reg_weight} "
              f"anneal={args.spec_reg_anneal} over {len(spec_layers)} unified layers", flush=True)

    def _spec_weight(step):
        if not args.spec_reg:
            return 0.0
        if args.spec_reg_anneal <= 0:
            return args.spec_reg_weight
        frac = min(1.0, step / max(1.0, args.spec_reg_anneal * args.steps))
        return args.spec_reg_weight * frac

    # Length curriculum (optional secondary lever): split total steps across stages.
    curriculum_stages = None
    if args.curriculum:
        curriculum_stages = [int(s) for s in args.curriculum.split(',') if s.strip()]
        log['curriculum_stages'] = curriculum_stages

    def _seq_len_for_step(step):
        if not curriculum_stages:
            return args.seq_len
        n = len(curriculum_stages)
        idx = min(n - 1, (step * n) // max(args.steps, 1))
        return curriculum_stages[idx]

    t0 = time.time()
    eval_interval = args.eval_interval if args.eval_interval is not None else max(50, args.steps // 20)
    model.train()
    if hasattr(optimizer, 'train'): optimizer.train()
    for step in range(args.steps):
        cur_seq_len = _seq_len_for_step(step)
        inp, tgt, mask = task.generate_batch(args.batch_size, cur_seq_len, rng)
        x = torch.from_numpy(inp).to(device)
        y = torch.from_numpy(tgt).to(device)
        m = torch.from_numpy(mask).to(device)
        use_autocast = device == 'cuda' and not getattr(model, 'disable_autocast', False)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
            logits = model(x)
        loss_per = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                    y.view(-1), reduction='none').view_as(m)
        loss = (loss_per * m).sum() / m.sum().clamp_min(1)
        if spec_layers:
            sw = _spec_weight(step)
            if sw > 0:
                reg = sum(l.specialization_loss(args.spec_reg) for l in spec_layers) / len(spec_layers)
                loss = loss + sw * reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == args.steps - 1:
            if hasattr(optimizer, 'eval'): optimizer.eval()
            acc, eval_loss = evaluate(model, task, args.batch_size, args.seq_len, 4, rng, device)
            if hasattr(optimizer, 'train'): optimizer.train()
            elapsed = time.time() - t0
            print(f"  step {step:>5d}  train_loss={loss.item():.4f}  eval_acc={acc:.4f}  eval_loss={eval_loss:.4f}  ({elapsed:.0f}s)", flush=True)
            log['steps'].append({'step': step, 'train_loss': float(loss.item()),
                                  'eval_acc': float(acc), 'eval_loss': float(eval_loss),
                                  'elapsed_s': float(elapsed)})
            model.train()

    if hasattr(optimizer, 'eval'): optimizer.eval()
    acc, eval_loss = evaluate(model, task, args.batch_size, args.seq_len, 16, rng, device)
    log['final_acc'] = float(acc); log['final_loss'] = float(eval_loss)

    # Emergent-specialization logging (Run C): dump per-head learned knobs for
    # any UnifiedCellLayer in the model (lambda gain, beta correction, gamma phi,
    # igain, and the along-key eigenvalue lambda-beta).
    unified_knobs = []
    for li, layer in enumerate(model.layers):
        if hasattr(layer, 'knob_values'):
            kv = layer.knob_values()
            entry = {
                'layer': li,
                'knob_mode': getattr(layer, 'knob_mode', None),
                'preset': getattr(layer, 'preset', None),
                'phi_mode': getattr(layer, 'phi_mode', None),
                'lambda': kv['lambda'].tolist(),
                'beta': kv['beta'].tolist(),
                'igain': kv['igain'].tolist(),
                'gamma': kv['gamma'].tolist(),
                'eig_along': kv['eig_along'].tolist(),
            }
            # TYPE-DICTIONARY interpretability: dump the K shared prototype knobs
            # and each head's argmax prototype (the type it leans on most).
            if getattr(layer, 'knob_mode', None) == 'dictionary':
                with torch.no_grad():
                    plam = (layer.lam_max * torch.sigmoid(layer.proto_lam_raw)).cpu().tolist()
                    pbeta = (layer.beta_max * torch.sigmoid(layer.proto_beta_raw)).cpu().tolist()
                    pgam = torch.sigmoid(layer.proto_gamma_raw).cpu().tolist()
                    w = torch.softmax(layer.proto_weight, dim=1)  # [H,K]
                    entry['proto'] = {'lambda': plam, 'beta': pbeta, 'gamma': pgam}
                    entry['proto_argmax'] = w.argmax(dim=1).cpu().tolist()
                    entry['proto_weight'] = w.cpu().tolist()
            unified_knobs.append(entry)
    if unified_knobs:
        log['unified_knobs'] = unified_knobs

    # Final specialization-regularizer value (per-layer mean), for diagnostics.
    if spec_layers:
        with torch.no_grad():
            log['spec_reg_final'] = float(
                sum(l.specialization_loss(args.spec_reg) for l in spec_layers) / len(spec_layers))
    log['elapsed_total_s'] = float(time.time() - t0)
    print(f"\nFINAL: acc={acc:.4f}  loss={eval_loss:.4f}  baseline={task.random_baseline_acc():.4f}", flush=True)

    # Length-extrapolation eval (Délétang protocol): test at lengths the
    # model never trained on. A model that learned the algorithm
    # extrapolates; a model that memorized the training-length
    # distribution does not.
    if args.eval_lengths is not None:
        log['length_extrap'] = {}
        # Use a smaller per-batch B at very long T to avoid OOM.
        for T_eval in args.eval_lengths:
            B_eval = args.batch_size
            # Cap memory: scale batch down for very long sequences.
            if T_eval > 4 * args.seq_len:
                B_eval = max(2, args.batch_size // (T_eval // (4 * args.seq_len)))
            try:
                acc_T, loss_T = evaluate(
                    model, task, B_eval, T_eval,
                    args.eval_lengths_n_batches, rng, device,
                )
                print(f"  length_extrap T={T_eval:>5d} (B={B_eval}): "
                      f"acc={acc_T:.4f}  loss={loss_T:.4f}", flush=True)
                log['length_extrap'][str(T_eval)] = {
                    'acc': float(acc_T),
                    'loss': float(loss_T),
                    'B_eval': int(B_eval),
                }
            except Exception as e:
                print(f"  length_extrap T={T_eval}: ERROR {type(e).__name__}: {e}",
                      flush=True)
                log['length_extrap'][str(T_eval)] = {'error': str(e)}

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'{args.label}.json')
    json.dump(log, open(out_path, 'w'), indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
