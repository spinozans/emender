"""emender-real-cap final synthesis: one doc from all REAL results.

Pulls: results_cma/cma_result.json (CMA-FOUND mixture, the core deliverable),
results_cap/*.json (expressivity battery, separation vs BOTH controls), throughput.json,
results_lm_tie/tie_result.json (convergent-loss tie). Emits RESULTS_EMENDER_REAL_CAP.md
+ synth_summary.json. Robust to partial data.

Separation is reported vs TWO controls:
  * gdn2     = pure GDN-2 incumbent (fla-gdn, neg-eig)             -- the headline incumbent
  * gdn2typed= all-gdn2_recall on the SAME typed path as Emender  -- the CLEAN control
    (gdn2typed vs emenderN isolates the emendment HEADS from the typed plumbing).
"""
import json, os, re, sys
from collections import defaultdict
from statistics import mean, pstdev

THIS = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(THIS, 'results_cap')
EVALS = ['128', '256', '512']
ARMS = ['gdn2', 'gdn2typed', 'emender4', 'emender8', 'shell4']
TASKS = ['modular_quadratic', 'iterated_nonlinear_map', 's5_permutation',
         'modular_counter', 'mqar_recall']
FN = re.compile(r'emc_(?P<task>.+?)(?:_K\d+)?__(?P<arm>[a-z0-9]+)__seed(?P<seed>\d+)\.json$')


def load_cap():
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if not os.path.isdir(CAP):
        return acc
    for fn in os.listdir(CAP):
        m = FN.match(fn)
        if not m:
            continue
        d = json.load(open(os.path.join(CAP, fn)))
        le = d.get('length_extrap') or {}
        for L in EVALS:
            if L in le and 'acc' in le[L]:
                acc[m['task']][m['arm']][L].append(le[L]['acc'])
    return acc


def f(xs):
    xs = [x for x in xs if x is not None]
    return f'{mean(xs):.3f}' if xs else '  -  '


def main():
    P = []
    add = P.append
    summary = {}

    # ---- CMA-found mixture (core deliverable) ----
    cma_path = os.path.join(THIS, 'results_cma', 'cma_result.json')
    add("# emender-real-cap — RESULTS\n")
    add("**The REAL Emender, at matched precision, with the mixture DISCOVERED by CMA-ES "
        "(not hand-picked).**\n")
    add("## 0. Precision finding (flagged, verified on real GPU)\n")
    add("The fused E97 split-edit Triton kernel (the Emender's nonlinear EMENDMENT head, "
        "`e97_delta` with per-step bounded `tanh`) is **bf16-ONLY**: pure fp16 raises "
        "`RuntimeError` (it refuses rather than silently running the eager T-scan), and "
        "fp16-autocast silently casts the emendment head to bf16 (a hidden dtype mismatch — "
        "the very thing the task forbids). So **fp16-uniform-fused is impossible** for the "
        "Emender. Per the task's escape clause, all arms run **bf16 UNIFORM** (half precision, "
        "uniform, fused, no fp32, no mismatch) — the matched-precision fix to the opt-1p3b "
        "fp32 strawman (which ran the Emender in fp32 = 4.3x token-starved vs bf16 controls).\n")

    if os.path.exists(cma_path):
        c = json.load(open(cma_path))
        fm = c['found_mixture']
        summary['cma_found'] = fm
        add("## 1. CMA-ES search — the FOUND mixture (core deliverable)\n")
        add(f"- **Search space**: {c['search']}")
        add(f"- **Fitness**: {c['fitness']}")
        add(f"- **Param/FLOP-locked**: target {c['param_target']:.0f} params, dim derived per candidate")
        add(f"- **popsize/gens**: {c['popsize']}/{c['generations']}; proxy {c['proxy_shape']}\n")
        add(f"**FOUND mixture** = `{fm['name']}`: held-out bpb **{fm['heldout_bpb']:.5f}**, "
            f"e97_delta frac **{fm['f_delta']:.4f}**, e97_track frac **{fm['f_track']:.4f}**, "
            f"total nonlinear fraction **{fm['nonlinear_fraction']:.4f}**.")
        add(f"\n**VERDICT (loss-CMA)**: {fm['verdict']}\n")
        add("Leaderboard (lowest held-out bpb first):\n")
        add("| rank | name | bpb | f_delta | f_track | counts |")
        add("|---|---|---|---|---|---|")
        for i, lb in enumerate(c['leaderboard'][:8]):
            nz = {k: v for k, v in (lb['counts'] or {}).items() if v}
            add(f"| {i+1} | {lb['name']} | {lb['bpb']:.5f} | {lb['f_delta']:.3f} | "
                f"{lb['f_track']:.3f} | {nz} |")
        add("")
        # per-generation fraction trajectory
        gp = os.path.join(THIS, 'results_cma', 'generations.jsonl')
        if os.path.exists(gp):
            add("CMA generation trajectory (gen-best):\n")
            add("| gen | best bpb | f_delta | f_track |")
            add("|---|---|---|---|")
            for line in open(gp):
                g = json.loads(line)
                add(f"| {g['gen']} | {g['gen_best']:.5f} | {g['gen_best_f_delta']:.3f} | "
                    f"{g['gen_best_f_track']:.3f} |")
            add("")
    else:
        add("## 1. CMA-ES search — (running; cma_result.json not yet present)\n")

    # ---- expressivity separation ----
    acc = load_cap()
    if acc:
        add("## 2. Expressivity battery — separation vs BOTH controls (3 seeds, eval-length extrap)\n")
        add("`sepF` = emender − fla-gdn (incumbent); `sepT` = emender − gdn2typed "
            "(CLEAN same-path control = isolates the emendment HEADS).\n")
        summary['expressivity'] = {}
        for task in TASKS:
            if task not in acc:
                continue
            add(f"### {task}")
            add("| arm | T=128 | T=256 | T=512 (cliff) |")
            add("|---|---|---|---|")
            for arm in ARMS:
                if arm in acc[task]:
                    r = [f(acc[task][arm].get(L, [])) for L in EVALS]
                    add(f"| {arm} | {r[0]} | {r[1]} | {r[2]} |")
            seps = {}
            for arm in ('emender4', 'emender8'):
                if arm not in acc[task]:
                    continue
                e512 = acc[task][arm].get('512', [])
                g512 = acc[task]['gdn2'].get('512', [])
                t512 = acc[task]['gdn2typed'].get('512', [])
                if e512 and g512:
                    seps[f'{arm} sepF@512'] = round(mean(e512) - mean(g512), 3)
                if e512 and t512:
                    seps[f'{arm} sepT@512'] = round(mean(e512) - mean(t512), 3)
            if seps:
                add("\n*" + ", ".join(f"`{k}={v:+.3f}`" for k, v in seps.items()) + "*")
            add("")
            summary['expressivity'][task] = {
                arm: {L: round(mean(v), 4) for L, v in acc[task][arm].items() if v}
                for arm in ARMS if arm in acc[task]}

    # ---- convergent-loss tie ----
    tie_path = os.path.join(THIS, 'results_lm_tie_lr5e4', 'tie_result.json')
    if os.path.exists(tie_path):
        t = json.load(open(tie_path))
        s = t['summary']
        add("## 3. Convergent-loss tie — held-out BPB (real-Pile, bf16 uniform, fused)\n")
        add("| budget | gdn2 bpb | emender(found) bpb | Δ (emender−gdn2) | gdn2 tok | emender tok |")
        add("|---|---|---|---|---|---|")
        for mode in ('token', 'wall'):
            g = s.get(f'gdn2_{mode}', {})
            e = s.get(f'emender_found_{mode}', {})
            if g.get('bpb_mean') is not None and e.get('bpb_mean') is not None:
                d = e['bpb_mean'] - g['bpb_mean']
                add(f"| {mode}-matched | {g['bpb_mean']:.5f} | {e['bpb_mean']:.5f} | "
                    f"{d:+.5f} | {g.get('tokens_mean')} | {e.get('tokens_mean')} |")
        summary['loss_tie'] = s
        add("")

    # ---- throughput ----
    tp = os.path.join(THIS, 'throughput.json')
    if os.path.exists(tp):
        d = json.load(open(tp))
        add("## 4. Throughput @ 1.3B head shape (fwd+bwd bf16, ratio vs GDN-2)\n")
        add("| config | tok/s | ratio |")
        add("|---|---|---|")
        for r in d['rows']:
            add(f"| {r['label']} | {r['tok_s']:.0f} | {r['ratio_vs_gdn2']:.3f} |")
        summary['throughput'] = {r['label']: r['ratio_vs_gdn2'] for r in d['rows']}
        add(f"\n*(base shape: dim={d['dim']}, {d['base']})*\n")

    out = '\n'.join(P)
    open(os.path.join(THIS, 'RESULTS_EMENDER_REAL_CAP.md'), 'w').write(out)
    json.dump(summary, open(os.path.join(THIS, 'synth_summary.json'), 'w'), indent=2)
    print(out)
    print(f"\n[wrote RESULTS_EMENDER_REAL_CAP.md + synth_summary.json]", file=sys.stderr)


if __name__ == '__main__':
    main()
