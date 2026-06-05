"""UnifiedCellLayer — ONE parameterized matrix-recurrence cell.

Wraps the unified Triton kernel (``ndm.triton.unified_cell``). Per head, with
matrix state S [N, V]:

    u_t   = k_t^T S_{t-1}
    pre_t = lambda_t S_{t-1} - beta_t (k_t outer u_t) + i_t (k_t outer v_t)
    S_t   = phi_gamma(pre_t)
    o_t   = S_t^T q_t

The three scalar knobs per head (and the phi shape) select capability:

    lambda  gain/decay of the self-loop.  RANGE FREE (the "un-cribbing"):
            <1 leaky, =1 integrate/count, >1 runaway -> bounded by phi -> latch.
            ``lam_max`` caps it: 1.0 == cribbed/clamped (E88 regime), 1.5 == free.
    beta    delta-correction in [0, beta_max].  0 = pure accumulate, beta>lambda
            -> along-key eigenvalue (lambda-beta) goes negative -> reflection/track.
    igain   input write gain.
    phi     interpolates identity (linear) <-> tanh via gamma (gamma_mix), or a
            fixed identity/tanh/relu/softplus.

Knob modes:
    'pinned'  : knobs are fixed buffers at a capability corner (preset arms).
    'learned' : knobs are per-head learnable parameters (the LEARNED cell). With
                lam_max=1.0 this is the CLAMPED un-cribbing-demo arm; lam_max=1.5
                is the FREE arm.

L2-normalized k,q make the along-key eigenvalue exactly (lambda - beta).
Optional per-head output RMSNorm (``head_norm``) is the mLSTM-style readout
stabilizer for the high-gain / counting regimes where |S| grows.
"""
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ndm.triton.unified_cell_backward import unified_cell
from ndm.triton.unified_cell_forward import (
    PHI_NAME_TO_CODE, PHI_IDENTITY, PHI_TANH, PHI_GAMMA_MIX, PHI_RELU, PHI_SOFTPLUS,
)


# Capability-corner centroids in (lambda, beta, gamma) space. These are the four
# specialization targets used by BOTH the specialization-pressure regularizer
# (specialization_loss below) and the aggregator's nearest-corner classifier
# (aggregate_specialization_study.py). They match the spread-init corners.
SPEC_CORNERS = {
    'track':  (0.9, 1.8, 0.05),   # linear, along-key eig -0.9 (reflection/S5)
    'count':  (1.0, 0.0, 0.05),   # linear, pure integration
    'latch':  (1.3, 0.0, 0.95),   # tanh, bistable +/-1
    'nonlin': (0.9, 0.5, 0.95),   # tanh, state-nonlinear phi
    # E98 FIVE-CORNER 5th corner: the leaky-linear ASSOCIATIVE-MEMORY workhorse.
    # lambda<1, small beta -> POSITIVE along-key eig (0.9-0.1)=0.8 in (0,1), linear
    # phi=identity: a fading key-value store (the GDN/Mamba regime that does real
    # LM recall). NOT one of the exotic primitives.
    'leaky':  (0.9, 0.1, 0.05),   # linear, along-key eig +0.8 (fading recall)
}

# Ordered corner lists for SPREAD placement. spread-4 = the four exotic primitives
# (the current population); spread-5 appends the leaky-linear workhorse so a
# 5-type population places ~1/5 of the heads on associative recall.
SPREAD_CORNERS_4 = ['track', 'count', 'latch', 'nonlin']
SPREAD_CORNERS_5 = ['track', 'count', 'latch', 'nonlin', 'leaky']
# The generic-init compromise regime (sigmoid-midpoint-ish): heads collapse here
# when under-pressured. The anti-center variant repels from this point.
SPEC_CENTER = (0.95, 0.5, 0.5)
# Per-axis scale to normalize the three knob axes (lambda~O(1), beta~O(2),
# gamma in [0,1]) before computing corner distances.
SPEC_SCALE = (0.4, 1.8, 1.0)


# Capability-corner presets: (phi_name, lam0, beta0, igain0, lam_max, beta_max, head_norm)
PRESETS = {
    # track (S5): contractive orthogonal, strong along-key reflection (lam-beta<0), linear
    'track':   dict(phi='identity', lam0=0.9, beta0=1.8, igain0=1.0, lam_max=1.0, beta_max=2.0, head_norm=False),
    # count (a^n b^n c^n): pure integration, lambda exactly 1, no correction, linear + readout norm
    'count':   dict(phi='identity', lam0=1.0, beta0=0.0, igain0=1.0, lam_max=1.0, beta_max=2.0, head_norm=True),
    # latch (flag-hold): gain>1 with tanh -> bistable +/-1 attractors
    'latch':   dict(phi='tanh',     lam0=1.3, beta0=0.0, igain0=1.0, lam_max=1.5, beta_max=2.0, head_norm=False),
    # nonlinear (iterated map): genuinely state-nonlinear phi
    'nonlin':  dict(phi='tanh',     lam0=0.9, beta0=0.5, igain0=1.0, lam_max=1.0, beta_max=2.0, head_norm=False),
    # E88-baseline: cribbed contractive delta (lambda<1, beta=1 full delta, tanh) == E88 recurrence
    'e88base': dict(phi='tanh',     lam0=0.9, beta0=1.0, igain0=1.0, lam_max=1.0, beta_max=1.0, head_norm=False),
    # E98 FIVE-CORNER: leaky-linear associative memory (the LM workhorse). lambda<1
    # + small beta -> positive along-key eig in (0,1), identity phi -> fading
    # key-value recall. head_norm on (mLSTM-style readout stabilizer aids recall).
    'leaky-linear': dict(phi='identity', lam0=0.9, beta0=0.1, igain0=1.0, lam_max=1.0, beta_max=2.0, head_norm=True),
}


class UnifiedCellLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_state: int = 32,
        n_heads: int = 8,
        expansion: float = 1.0,
        knob_mode: str = 'learned',   # 'pinned' | 'learned' | 'dictionary' | 'fixed_pop'
        preset: Optional[str] = None,  # for pinned mode, name in PRESETS
        phi: str = 'gamma_mix',        # for learned mode
        spread_init: bool = False,     # learned mode: spread per-head knobs ACROSS corners
        n_spread_corners: int = 4,     # 4 = exotic corners (spread-4); 5 adds leaky-linear (spread-5)
        corner_mixture: Optional[list] = None,  # spread_init/fixed_pop: per-corner head
        #   fractions over the n_spread_corners corners ([track,count,latch,nonlin]
        #   for spread-4, +leaky for spread-5). None == equal round-robin. When
        #   given, heads are assigned in contiguous blocks sized by the
        #   (renormalized) fractions via largest-remainder rounding. Tuned by the
        #   cma-capability meta-search; default None preserves the validated
        #   spread/fixedpop behaviour exactly.
        n_proto: int = 4,              # dictionary mode: number of shared prototype knobs
        lam_max: float = 1.5,          # learned cap; 1.0 == clamped (cribbed)
        beta_max: float = 2.0,
        igain_max: float = 2.0,
        head_norm: bool = True,        # per-head readout RMSNorm (mLSTM-style stabilizer)
        use_gate: bool = True,
        gate_activation: str = 'silu',
        split_gate: bool = False,      # E97/E98: decoupled erase (b*k) + value-write (w*v) gates
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.n_heads = n_heads
        N = n_state
        V = int(round(n_state * expansion))
        self.N, self.V = N, V
        self.key_dim = n_heads * N
        self.value_dim = n_heads * V
        self.knob_mode = knob_mode

        if knob_mode == 'pinned':
            assert preset in PRESETS, f"unknown preset {preset!r}"
            cfg = PRESETS[preset]
            self.preset = preset
            phi = cfg['phi']; lam_max = cfg['lam_max']; beta_max = cfg['beta_max']
            head_norm = cfg['head_norm']
            self._lam0, self._beta0, self._igain0 = cfg['lam0'], cfg['beta0'], cfg['igain0']
        else:
            self.preset = None
        self.phi_mode = PHI_NAME_TO_CODE[phi]
        self.lam_max, self.beta_max, self.igain_max = lam_max, beta_max, igain_max
        self.head_norm = head_norm
        self.spread_init = bool(spread_init) and knob_mode == 'learned'
        self.n_spread_corners = int(n_spread_corners)
        self.corner_mixture = self._normalize_mixture(corner_mixture)
        self.n_proto = int(n_proto)

        # projections (fused qkv)
        self.qkv_proj = nn.Linear(dim, 2 * self.key_dim + self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, self.value_dim, bias=False) if use_gate else None
        self.gate_activation = gate_activation

        # E97/E98 SPLIT-GATE (GDN-2-inspired decoupling): the correction term
        # becomes  pre = lambda*S - beta*k ((b*k)^T S) + i*k (w*v)^T,  i.e. the
        # read/erase direction is the input-dependently-gated key b*k while the
        # write value is the gated w*v; the outer-product (write) key stays the
        # ungated k. With b=w=1 this reduces EXACTLY to the E88-based unified cell,
        # so all four capability corners remain reachable (b->1 opens the erase so
        # the along-key eigenvalue is again lambda-beta; the track/reflection corner
        # is reached by beta>lambda WITH b open, and b<1 is a strictly richer
        # input-dependent partial-erase the E88 cell could not express).
        self.split_gate = bool(split_gate)
        if self.split_gate:
            self.erase_gate_proj = nn.Linear(dim, self.key_dim, bias=False)
            self.value_write_gate_proj = nn.Linear(dim, self.value_dim, bias=False)
        else:
            self.erase_gate_proj = None
            self.value_write_gate_proj = None

        if head_norm:
            self.head_norm_weight = nn.Parameter(torch.ones(n_heads, V))
        self.norm_eps = 1e-5

        # --- knobs ---
        H = n_heads
        if knob_mode == 'pinned':
            # fixed (non-learnable) raw values realising the exact corner.
            self.register_buffer('lam_raw', self._inv_scaled(torch.full((H,), self._lam0), lam_max))
            self.register_buffer('beta_raw', self._inv_scaled(torch.full((H,), self._beta0), beta_max))
            self.register_buffer('igain_raw', self._inv_scaled(torch.full((H,), self._igain0), igain_max))
            self.register_buffer('gamma_raw', torch.full((H,), 4.0 if phi == 'tanh' else (-4.0 if phi == 'identity' else 0.0)))
        elif knob_mode == 'fixed_pop':
            # FIXED-TYPE POPULATION (floor baseline): heads HARD-assigned to the
            # four corners round-robin as NON-learnable buffers. Only the q/k/v/o
            # projections (and gate) learn; the recurrence types are fixed. This
            # is the population analogue of the pinned single-corner presets, and
            # the floor the learnable approaches must beat to justify their cost.
            lam0, beta0, gam0 = self._spread_corner_values(H, lam_max, self.n_spread_corners, self.corner_mixture)
            self.register_buffer('lam_raw', self._inv_scaled(lam0, lam_max))
            self.register_buffer('beta_raw', self._inv_scaled(beta0, beta_max))
            self.register_buffer('igain_raw', self._inv_scaled(torch.full((H,), min(1.0, 0.5 * igain_max)), igain_max))
            self.register_buffer('gamma_raw', torch.log(gam0 / (1.0 - gam0)))
        elif knob_mode == 'dictionary':
            # TYPE-DICTIONARY: K SHARED learnable prototype knobs + a per-head soft
            # weight over prototypes. knob_h = sum_k softmax(w_h)_k * proto_k (mixed
            # in raw/pre-sigmoid space). Strong shared gradient (all heads pull the
            # same K prototypes) + per-head flexibility (each head picks its mix).
            # Prototypes init spread across the corners (K>=4: corners then jitter)
            # so the shared gradient REFINES a diverse dictionary rather than
            # discovering it from scratch; per-head weights init near-uniform with
            # small symmetry-breaking noise.
            K = self.n_proto
            plam, pbeta, pgam = self._dict_proto_values(K, lam_max)
            self.proto_lam_raw = nn.Parameter(self._inv_scaled(plam, lam_max))
            self.proto_beta_raw = nn.Parameter(self._inv_scaled(pbeta, beta_max))
            self.proto_igain_raw = nn.Parameter(
                self._inv_scaled(torch.full((K,), min(1.0, 0.5 * igain_max)), igain_max))
            self.proto_gamma_raw = nn.Parameter(torch.log(pgam / (1.0 - pgam)))
            # per-head soft weights [H, K]: near-uniform logits + deterministic
            # per-(head,proto) jitter so heads don't start identical.
            base_w = torch.zeros(H, K)
            for h in range(H):
                for kk in range(K):
                    base_w[h, kk] = 0.1 * math.sin(1.0 + h * 1.3 + kk * 2.7)
            self.proto_weight = nn.Parameter(base_w)
        elif self.spread_init:
            # SPREAD INIT (learnability intervention #1): partition the H heads
            # ACROSS the four capability corners so descent REFINES specialization
            # rather than discovering it from a generic center. Requires the FREE
            # gain cap (lam_max>=~1.35) so the latch corner (lambda=1.3) is
            # reachable; gamma_mix phi lets gamma in {~0 linear, ~1 tanh} select
            # linear vs saturating per head.
            #   track  : lambda 0.9,  beta 1.8, gamma 0.05 (linear)  -> along-key eig -0.9 (reflection/S5)
            #   count  : lambda 1.0,  beta 0.0, gamma 0.05 (linear)  -> pure integration
            #   latch  : lambda 1.3,  beta 0.0, gamma 0.95 (tanh)    -> bistable +/-1 attractors
            #   nonlin : lambda 0.9,  beta 0.5, gamma 0.95 (tanh)    -> state-nonlinear phi
            lam0, beta0, gam0 = self._spread_corner_values(H, lam_max, self.n_spread_corners, self.corner_mixture)
            self.lam_raw = nn.Parameter(self._inv_scaled(lam0, lam_max))
            self.beta_raw = nn.Parameter(self._inv_scaled(beta0, beta_max))
            self.igain_raw = nn.Parameter(self._inv_scaled(torch.full((H,), min(1.0, 0.5 * igain_max)), igain_max))
            # gamma_raw is a plain logit (gamma=sigmoid(raw)); set per-head.
            self.gamma_raw = nn.Parameter(torch.log(gam0 / (1.0 - gam0)))
        else:
            # learned per-head knobs. Init: lam~0.95, beta~0.5, igain~1.0, gamma~0.5.
            self.lam_raw = nn.Parameter(self._inv_scaled(torch.full((H,), min(0.95, 0.99 * lam_max)), lam_max))
            self.beta_raw = nn.Parameter(self._inv_scaled(torch.full((H,), min(0.5, 0.49 * beta_max)), beta_max))
            self.igain_raw = nn.Parameter(self._inv_scaled(torch.full((H,), min(1.0, 0.5 * igain_max)), igain_max))
            self.gamma_raw = nn.Parameter(torch.zeros(H))  # gamma=sigmoid(0)=0.5

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.g_proj is not None:
            nn.init.xavier_uniform_(self.g_proj.weight)
        if self.split_gate:
            # Small init -> gates start near sigmoid(0)=0.5; the erase gate is then
            # free to open (->1, recovering the unified corners) or partially close.
            nn.init.xavier_uniform_(self.erase_gate_proj.weight, gain=0.1)
            nn.init.xavier_uniform_(self.value_write_gate_proj.weight, gain=0.1)

    @staticmethod
    def _normalize_mixture(mixture):
        """Validate/normalize a per-corner head-fraction spec to a list of floats
        summing to 1, or None. Accepts counts or fractions; clamps negatives to 0.
        Length must be 4 (spread-4 [track,count,latch,nonlin]) or 5 (spread-5,
        +leaky); consistency with n_spread_corners is checked at placement time.
        """
        if mixture is None:
            return None
        vals = [max(0.0, float(v)) for v in mixture]
        if len(vals) not in (4, 5):
            raise ValueError(
                f"corner_mixture must have 4 or 5 entries, got {len(vals)}")
        s = sum(vals)
        if s <= 0:
            return None  # degenerate -> fall back to equal round-robin
        return [v / s for v in vals]

    @staticmethod
    def _mixture_counts(H, mixture, C):
        """Heads-per-corner (length C) from fractions via largest-remainder
        rounding; every corner with fraction>0 gets >=1 head when H>=C so coverage
        is preserved."""
        if len(mixture) != C:
            raise ValueError(
                f"corner_mixture has {len(mixture)} entries but n_spread_corners={C}")
        raw = [f * H for f in mixture]
        counts = [int(c) for c in raw]
        # guarantee >=1 head for any nonzero-fraction corner (coverage floor)
        for i, f in enumerate(mixture):
            if f > 0 and counts[i] == 0 and H >= C:
                counts[i] = 1
        deficit = H - sum(counts)
        if deficit > 0:  # distribute by largest fractional remainder
            rema = sorted(range(C), key=lambda i: raw[i] - int(raw[i]), reverse=True)
            for j in range(deficit):
                counts[rema[j % C]] += 1
        elif deficit < 0:  # over-allocated by the coverage floor: trim largest
            for _ in range(-deficit):
                i = max(range(C), key=lambda i: counts[i])
                counts[i] -= 1
        return counts

    @staticmethod
    def _spread_corner_values(H, lam_max, n_corners=4, mixture=None):
        """Per-head (lambda, beta, gamma) init partitioned across the corners.

        n_corners selects the corner population: 4 -> the four exotic primitives
        (spread-4, the validated default); 5 -> appends the leaky-linear
        associative-memory workhorse (spread-5). mixture is None -> heads are
        assigned round-robin (head_idx % C), the balanced default. When a per-corner
        `mixture` (length C) is given (cma-capability meta-search), heads are
        assigned in contiguous blocks sized by those fractions instead. Returns
        float tensors [H]; lambda is clamped to the achievable cap (latch's 1.3
        needs lam_max>=1.3).
        """
        names = SPREAD_CORNERS_5 if n_corners >= 5 else SPREAD_CORNERS_4
        corners = [SPEC_CORNERS[n] for n in names]
        C = len(corners)
        lam = torch.empty(H)
        beta = torch.empty(H)
        gam = torch.empty(H)
        lam_cap = min(1.49, 0.99 * lam_max)
        if mixture is None:
            assign = [h % C for h in range(H)]
        else:
            counts = UnifiedCellLayer._mixture_counts(H, mixture, C)
            assign = []
            for ci, n in enumerate(counts):
                assign.extend([ci] * n)
        for h in range(H):
            l, b, g = corners[assign[h]]
            lam[h] = min(l, lam_cap)
            beta[h] = b
            gam[h] = g
        return lam, beta, gam

    @staticmethod
    def _dict_proto_values(K, lam_max):
        """K prototype (lambda, beta, gamma) inits spread across the four corners.

        K==4 -> exactly the four corners. K>4 -> corners then round-robin repeats
        with a small deterministic jitter so duplicate prototypes are distinct.
        """
        corners = [
            (0.9, 1.8, 0.05),  # track
            (1.0, 0.0, 0.05),  # count
            (1.3, 0.0, 0.95),  # latch
            (0.9, 0.5, 0.95),  # nonlin
        ]
        lam = torch.empty(K); beta = torch.empty(K); gam = torch.empty(K)
        lam_cap = min(1.49, 0.99 * lam_max)
        for k in range(K):
            l, b, g = corners[k % 4]
            jit = 0.05 * math.sin(0.7 + k * 1.9)  # 0 for k<4 only if sin!=0; small
            rep = k // 4  # 0 for first pass, >0 for repeats
            l = l + (jit if rep else 0.0)
            b = max(0.0, b + (jit if rep else 0.0))
            g = min(0.99, max(0.01, g + (jit if rep else 0.0)))
            lam[k] = min(max(0.05, l), lam_cap)
            beta[k] = b
            gam[k] = g
        return lam, beta, gam

    @staticmethod
    def _inv_scaled(val, vmax):
        """Inverse of vmax*sigmoid(raw): raw = logit(val/vmax), clamped."""
        r = (val / vmax).clamp(1e-4, 1 - 1e-4)
        return torch.log(r / (1 - r))

    def _raws(self):
        """Effective per-head raw knobs [H] for (lam, beta, igain, gamma).

        For 'dictionary' mode these are mixed on-the-fly from the K shared
        prototypes via the per-head soft weights; for all other modes they are the
        stored per-head parameters/buffers directly.
        """
        if self.knob_mode == 'dictionary':
            w = torch.softmax(self.proto_weight, dim=1)  # [H, K]
            lam = w @ self.proto_lam_raw
            beta = w @ self.proto_beta_raw
            igain = w @ self.proto_igain_raw
            gamma = w @ self.proto_gamma_raw
            return lam, beta, igain, gamma
        return self.lam_raw, self.beta_raw, self.igain_raw, self.gamma_raw

    # knob getters (scaled to range)
    def _lam(self):
        return self.lam_max * torch.sigmoid(self._raws()[0])

    def _beta(self):
        return self.beta_max * torch.sigmoid(self._raws()[1])

    def _igain(self):
        return self.igain_max * torch.sigmoid(self._raws()[2])

    def _gamma(self):
        return torch.sigmoid(self._raws()[3])

    def knob_values(self):
        """Return current per-head (lambda, beta, igain, gamma) as detached numpy-friendly tensors."""
        with torch.no_grad():
            return {
                'lambda': self._lam().detach().cpu(),
                'beta': self._beta().detach().cpu(),
                'igain': self._igain().detach().cpu(),
                'gamma': self._gamma().detach().cpu(),
                'eig_along': (self._lam() - self._beta()).detach().cpu(),
            }

    def specialization_loss(self, variant: str = 'pull', tau: float = 0.1,
                            sigma2: float = 0.25, kappa: float = 1.0):
        """SPECIALIZATION-PRESSURE regularizer on this layer's per-head knobs.

        Computes a differentiable scalar that, when minimized alongside the task
        loss, FORCES heterogeneous per-head specialization onto the four capability
        corners (track/count/latch/nonlin) instead of collapsing to the generic
        center. All distances are in normalized (lambda,beta,gamma) space.

        variant:
          'pull'        (a) per-head pull-to-nearest-corner: mean_h min_c d2(h,c).
                        Quadratic well -> strong gradient when far, attracts each
                        head to its closest corner.
          'anticenter'  (b) per-head anti-center: bounded Gaussian bumps,
                        mean_h [ exp(-d2_center/s2) - exp(-min_c d2/s2) ].
                        Penalizes occupying the center, rewards being near ANY
                        corner. Bounded in [-1, 1] (safe even when init==center).
          'coverage'    (c) population coverage/diversity: KL(p || uniform) where
                        p = mean_h softmax(-d2/tau) is the head->corner assignment
                        distribution. Minimized when heads SPREAD across all four
                        corners (anti-collapse). Layer-level term.
          'pull_cov'        = pull + coverage
          'anticenter_cov'  = anticenter + coverage

        Returns a scalar tensor on the knob device. For non-learnable knob modes
        (pinned/fixed_pop) the gradient is zero (knobs are buffers); the term is
        only meaningful for 'learned'/'dictionary' arms.
        """
        lam, beta, gam = self._lam(), self._beta(), self._gamma()  # each [H]
        z = torch.stack([lam, beta, gam], dim=-1)  # [H, 3]
        dev = z.device
        corners = torch.tensor([SPEC_CORNERS[c] for c in ('track', 'count', 'latch', 'nonlin')],
                               dtype=z.dtype, device=dev)  # [C, 3]
        center = torch.tensor(SPEC_CENTER, dtype=z.dtype, device=dev)  # [3]
        scale = torch.tensor(SPEC_SCALE, dtype=z.dtype, device=dev)    # [3]

        diff = (z[:, None, :] - corners[None, :, :]) / scale  # [H, C, 3]
        d2 = diff.pow(2).sum(dim=-1)                            # [H, C]
        d2_min = d2.min(dim=1).values                          # [H]
        d2_center = (((z - center) / scale) ** 2).sum(dim=-1)  # [H]

        def _pull():
            return d2_min.mean()

        def _anticenter():
            return (torch.exp(-d2_center / sigma2) - kappa * torch.exp(-d2_min / sigma2)).mean()

        def _coverage():
            a = torch.softmax(-d2 / tau, dim=1)        # [H, C] soft assignment
            p = a.mean(dim=0)                          # [C] population distribution
            C = p.shape[0]
            ent = -(p * (p + 1e-9).log()).sum()
            return math.log(C) - ent                  # KL(p || uniform) >= 0

        if variant == 'pull':
            return _pull()
        if variant == 'anticenter':
            return _anticenter()
        if variant == 'coverage':
            return _coverage()
        if variant == 'pull_cov':
            return _pull() + _coverage()
        if variant == 'anticenter_cov':
            return _anticenter() + _coverage()
        raise ValueError(f"unknown specialization variant {variant!r}")

    def forward(self, x: torch.Tensor):
        # x: [B, T, dim]
        B, T, D = x.shape
        H, N, V = self.n_heads, self.N, self.V
        qkv = self.qkv_proj(x)  # [B,T, 2*key_dim+value_dim]
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(B, T, H, N)
        k = k.view(B, T, H, N)
        v = v.view(B, T, H, V)
        # L2-normalize k, q  -> along-key eigenvalue is exactly (lambda - beta)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        # -> [T, B, H, *]
        k = k.permute(1, 0, 2, 3).contiguous()
        q = q.permute(1, 0, 2, 3).contiguous()
        v = v.permute(1, 0, 2, 3).contiguous()

        lam = self._lam().to(x.dtype)      # [H]
        beta = self._beta().to(x.dtype)
        igain = self._igain().to(x.dtype)
        gamma = self._gamma().to(torch.float32)
        # broadcast per-head -> [T,B,H]
        lam_t = lam.view(1, 1, H).expand(T, B, H).contiguous()
        beta_t = beta.view(1, 1, H).expand(T, B, H).contiguous()
        igain_t = igain.view(1, 1, H).expand(T, B, H).contiguous()

        # E97/E98 split gates (input-dependent): b erase [T,B,H,N], w value [T,B,H,V].
        b_gate = w_gate = None
        if self.split_gate:
            bg = torch.sigmoid(self.erase_gate_proj(x)).view(B, T, H, N)
            wg = torch.sigmoid(self.value_write_gate_proj(x)).view(B, T, H, V)
            b_gate = bg.permute(1, 0, 2, 3).contiguous().to(x.dtype)
            w_gate = wg.permute(1, 0, 2, 3).contiguous().to(x.dtype)

        S0 = torch.zeros(B, H, N, V, device=x.device, dtype=x.dtype)
        out, _ = unified_cell(k, v, q, lam_t, beta_t, igain_t, gamma, S0,
                              phi_mode=self.phi_mode, b_gate=b_gate, w_gate=w_gate)
        # out: [T, B, H, V] -> [B, T, H, V]
        out = out.permute(1, 0, 2, 3).contiguous()

        if self.head_norm:
            # per-head RMSNorm over V (mLSTM-style readout stabilizer)
            rms = out.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
            out = out * rms * self.head_norm_weight.view(1, 1, H, V)

        out = out.reshape(B, T, H * V)
        if self.g_proj is not None:
            g = self.g_proj(x)
            if self.gate_activation in ('silu', 'swish'):
                out = out * F.silu(g)
            else:
                out = out * torch.sigmoid(g)
        out = self.o_proj(out)
        return self.dropout(out)
