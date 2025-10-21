# Contextual Thompson Sampler - Hyperparameter Guide

This document explains the CTS hyperparameters and their relationship to parameter scales and weight distributions.

## Understanding the Model

The CTS learns to weight value signals based on context:

```
Context c_t → Model parameters (U, b) → Logits z_t → Softmax + Mixing → Weights w_t
```

The key insight: **hyperparameters control the SCALE of randomness and updates relative to the signal.**

---

## Hyperparameters and Their Scales

### `lr` - Learning Rate
**Scale**: `0.001` to `0.1`

**What it controls**: Size of parameter updates after each gradient step.

**Formula**: `U = U - lr * grad_U`

**Relationship to parameters**:
- Gradients have scale O(1) (from backprop through softmax and loss)
- If `lr = 0.01`, parameters change by ~0.01 per update
- Starting from `U = 0`, after 100 updates with consistent gradient, `U` could reach O(1)

**Tuning logic**:
```
lr = 0.001  → Takes ~1000 steps to build up U to O(1) scale
lr = 0.01   → Takes ~100 steps to build up U to O(1) scale  ✓ CHOSEN
lr = 0.1    → Takes ~10 steps, but risks overshooting
```

**Why 0.01?**
- Warm-start has ~7000 samples, 3 epochs → ~21,000 gradient steps
- Want to converge in ~1000-5000 steps, leaving room for fine-tuning
- Faster than 0.001, more stable than 0.1

---

### `expl_scale` - Exploration Noise Variance Scale
**Scale**: `0.01` to `1.0`

**What it controls**: Variance of Gaussian noise added to parameters during Thompson Sampling.

**Formula**:
```python
noise_std_U = sqrt(expl_scale / h_U)
U_tilde = U + N(0, noise_std_U)
```

**Critical relationship**:
```
expl_scale = 0.01, h0 = 1.0  →  initial noise_std = 0.1  (low exploration)
expl_scale = 0.1,  h0 = 1.0  →  initial noise_std = 0.316 (moderate) ✓ CHOSEN
expl_scale = 1.0,  h0 = 1.0  →  initial noise_std = 1.0  (high exploration)
expl_scale = 1.0,  h0 = 0.001→  initial noise_std = 31.6 (CHAOS!)
```

**Relationship to logits and weights**:
- Logits `z = U @ c + b` initially O(noise_std) since U~noise
- With `noise_std = 0.316` and context features O(1):
  - Logits `z ~ N(0, 0.316)` → range approximately [-1, 1]
  - After temperature `z/tau = z/10` → range [-0.1, 0.1]
  - softmax([-0.1, 0.05, 0.1, ...]) ≈ [0.15, 0.17, 0.18, ...] (near uniform)

- With `noise_std = 31.6`:
  - Logits `z ~ N(0, 31.6)` → range approximately [-100, 100]
  - Even with `tau=10`: `z/10` → range [-10, 10]
  - softmax([-10, 5, 8, ...]) ≈ [0.0001, 0.18, 0.82] (BIMODAL!)

**Why 0.1?**
- Keeps initial logits small O(0.3), preventing softmax from being too peaked
- Allows meaningful exploration without drowning out learning
- As model learns, `h_U` grows with squared gradients, reducing exploration naturally

---

### `h0` - Initial Hessian Diagonal Value
**Scale**: `0.001` to `10.0`

**What it controls**: Initial value for diagonal Hessian approximation, directly controls initial exploration noise magnitude.

**Formula**:
```python
h_U = h0 * ones_like(U)  # Initially
noise_std = sqrt(expl_scale / h_U)
```

**This is the CRITICAL parameter for initialization!**

**Scale analysis**:
```
h0 = 0.001  →  noise_std = sqrt(0.1/0.001) = 10.0   (massive noise)
h0 = 0.01   →  noise_std = sqrt(0.1/0.01)  = 3.16   (large noise)
h0 = 0.1    →  noise_std = sqrt(0.1/0.1)   = 1.0    (moderate noise)
h0 = 1.0    →  noise_std = sqrt(0.1/1.0)   = 0.316  (controlled noise) ✓ CHOSEN
h0 = 10.0   →  noise_std = sqrt(0.1/10.0)  = 0.1    (low noise)
```

**Relationship to learning**:
- `h_U` accumulates squared gradients: `h_U = 0.999 * h_U + 0.001 * grad^2`
- If gradients are O(1), after 100 steps: `h_U ≈ 1.0 + 0.1 = 1.1`
- Noise decreases over time: `noise_std = sqrt(0.1 / 1.1) ≈ 0.3` (slightly less exploration)

**Why 1.0 instead of 0.001?**
- With `h0 = 0.001`, initial noise_std = 10.0 is **100x larger** than desired
- Logits become O(10), softmax becomes extremely peaked regardless of tau/alpha
- With `h0 = 1.0`, initial noise_std = 0.316 is reasonable
- Model starts near-uniform and gradually learns structure from data

---

### `tau` - Softmax Temperature
**Scale**: `1.0` to `20.0`

**What it controls**: How peaked vs flat the softmax distribution is.

**Formula**: `q = softmax(z / tau)`

**Relationship to logits and weights**:

For logits `z = [z1, z2, ..., z6]`, softmax gives:
```
q_i = exp(z_i / tau) / sum(exp(z_j / tau))
```

**Scale examples** (logits = [-1, 0, 0.5, 1, 0, -0.5]):
```
tau = 1:   softmax → [0.04, 0.11, 0.18, 0.30, 0.11, 0.07]  (peaked at z=1)
tau = 2:   softmax → [0.08, 0.13, 0.16, 0.21, 0.13, 0.10]  (moderate)
tau = 5:   softmax → [0.13, 0.16, 0.17, 0.19, 0.16, 0.14]  (flatter)
tau = 10:  softmax → [0.15, 0.17, 0.17, 0.18, 0.17, 0.16]  (very flat) ✓ CHOSEN
tau = 20:  softmax → [0.16, 0.17, 0.17, 0.17, 0.17, 0.16]  (nearly uniform)
```

**Intuition**:
- Low tau (1-2): "Confident" - strongly favors highest logit
- High tau (10-20): "Uncertain" - hedges bets across all signals
- tau → ∞: Always uniform (1/6 for each signal)

**Why 10.0?**
- At initialization with U=0, b=0: logits are O(noise_std) ≈ O(0.3)
- With `tau=10`: tempered logits O(0.03) → softmax ≈ [0.166, 0.167, 0.167, ...]
- Allows model to start near-uniform
- Model can still learn preferences (as U grows, logits grow, softmax becomes more peaked)
- After learning, if `z = [0, 0, 0, 5, 0, 0]` (one signal clearly best):
  - `z/10 = [0, 0, 0, 0.5, 0, 0]`
  - softmax ≈ [0.14, 0.14, 0.14, 0.21, 0.14, 0.14] (still diversified)

---

### `alpha` - Uniform Mixing Weight
**Scale**: `0.0` to `1.0`

**What it controls**: How much to mix softmax output with uniform distribution.

**Formula**: `w = (1 - alpha) * softmax(z/tau) + alpha * (1/num_signals)`

**Relationship to final weights**:

**Scale examples** (softmax output = [0.1, 0.15, 0.4, 0.2, 0.1, 0.05]):
```
alpha = 0.0:   w = [0.10, 0.15, 0.40, 0.20, 0.10, 0.05]  (pure softmax, peaked)
alpha = 0.2:   w = [0.11, 0.15, 0.35, 0.19, 0.11, 0.07]  (slight smoothing)
alpha = 0.5:   w = [0.13, 0.16, 0.28, 0.18, 0.13, 0.11]  (significant smoothing) ✓ CHOSEN
alpha = 0.8:   w = [0.15, 0.16, 0.21, 0.17, 0.15, 0.14]  (mostly uniform)
alpha = 1.0:   w = [0.167, 0.167, 0.167, 0.167, 0.167, 0.167]  (always uniform)
```

**Weight bounds**:
```
Min weight = alpha / num_signals = 0.5 / 6 ≈ 0.083
Max weight = (1 - alpha) * 1.0 + alpha/num_signals = 0.5 + 0.083 ≈ 0.583
```

**Why 0.5?**
- At initialization (softmax = uniform): `w = 0.5 * 1/6 + 0.5 * 1/6 = 1/6` (perfect uniform)
- Even after learning, no signal can be ignored (min 8.3%) or completely dominate (max 58.3%)
- Promotes portfolio approach: consider multiple signals, not just the "best" one
- Reflects RTS curator values: balance multiple objectives (audience, diversity, etc.)

---

### `ema_decay` - Exponential Moving Average Decay
**Scale**: `0.9` to `0.9999`

**What it controls**: How quickly Hessian approximation forgets old gradient information.

**Formula**: `h_U = ema_decay * h_U + (1 - ema_decay) * grad_U^2`

**Effective memory window**: `1 / (1 - ema_decay)`

**Scale analysis**:
```
ema_decay = 0.9    →  memory ≈ 10 updates  (fast adaptation, noisy)
ema_decay = 0.99   →  memory ≈ 100 updates  (moderate)
ema_decay = 0.999  →  memory ≈ 1000 updates (slow adaptation, smooth) ✓ CHOSEN
ema_decay = 0.9999 →  memory ≈ 10000 updates (very slow)
```

**Relationship to exploration decay**:
- As gradients accumulate, `h_U` grows
- Larger `h_U` → smaller exploration noise: `noise_std = sqrt(expl_scale / h_U)`
- With `ema_decay = 0.999`:
  - Initially: `h_U = 1.0`, `noise_std = 0.316`
  - After 1000 updates with grad≈1: `h_U ≈ 2.0`, `noise_std ≈ 0.224` (less exploration)
  - After 5000 updates: `h_U ≈ 5.0`, `noise_std ≈ 0.141` (even less)

**Why 0.999?**
- Warm-start has ~21,000 gradient steps (7000 samples × 3 epochs)
- Want `h_U` to grow gradually over training, not jump immediately
- Smooth exploration decay: start with moderate noise, end with less noise
- Not too slow (0.9999 would barely change over 20k steps)

---

### `match_loss_weight` - Signal Matching Loss Weight
**Scale**: `0.0` to `2.0`

**What it controls**: Weight for auxiliary loss that encourages matching curator-indicated signal preferences.

**Formula**: `total_loss = reward_loss + match_loss_weight * signal_matching_loss`

**Relationship to gradients**:
- Reward loss gradient: `dL/dw = (p - r) * s_chosen` where `p` = predicted success prob, `r` = actual
- Signal matching gradient: `dL/dw = (w - y_curator)` where `y_curator` = curator signal preferences
- Total gradient is sum weighted by `match_loss_weight`

**Scale reasoning**:
```
match_loss_weight = 0.0  →  Only learn from success/failure (pure RL)
match_loss_weight = 0.5  →  Balance both objectives ✓ CHOSEN
match_loss_weight = 1.0  →  Equal weight to both
match_loss_weight = 2.0  →  Prioritize signal matching over outcomes
```

**Why 0.5?**
- During warm-start: we only have binary "selected/not selected", no explicit signal preferences
- This loss is mainly useful for online learning with explicit curator feedback
- Moderate weight allows it to influence learning without dominating
- Can be tuned higher (0.7-1.0) if curator provides explicit signal feedback

---

## Initialization Strategy

### Zero Initialization: U = 0, b = 0

**Why zeros?**

1. **Deterministic starting point**: `z = U @ c + b = 0` for all signals (before noise)

2. **Scale of initial weights**:
   - Logits after noise: `z ~ N(0, noise_std^2)` where `noise_std = sqrt(expl_scale/h0) = 0.316`
   - After temperature: `z/tau ~ N(0, (0.316/10)^2) ≈ N(0, 0.001)`
   - softmax([small values close to 0]) ≈ [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
   - After mixing with `alpha=0.5`: still ≈ [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

3. **Contrast with random initialization**:
   ```python
   # Random init: U ~ N(0, 1), context features ~ O(1)
   z = U @ c  # O(sqrt(context_dim)) ≈ O(5) if context_dim ≈ 25
   z/tau = O(5/10) = O(0.5)
   softmax([values around 0.5]) → PEAKED distribution
   ```

   Even scaling down by 0.01:
   ```python
   z = 0.01 * U @ c  # O(0.05)
   # Still larger than noise O(0.03), creates variation
   ```

**Summary**:
```
                Initial Logits z    After tau=10    Softmax Output     After alpha=0.5 Mix
Zero init       0 + noise(0.3)      noise(0.03)     ~[1/6, 1/6, ...]   ~[1/6, 1/6, ...]  ✓
Random*0.01     O(0.05) + noise     O(0.005+0.03)   Slight variation   Near uniform
Random*1.0      O(5) + noise        O(0.5)          PEAKED             Bimodal even with mixing
```

---

## Summary: Scale Relationships

| Hyperparameter | Value | Controls | Scale Reasoning |
|---------------|-------|----------|-----------------|
| `lr` | 0.01 | Parameter update size | Need ~1000 steps to converge, have ~20k steps available |
| `expl_scale` | 0.1 | Exploration noise variance | With h0=1.0, gives noise_std=0.316 (reasonable for O(1) logits) |
| `h0` | 1.0 | Initial uncertainty | Makes initial noise O(0.3) not O(10), prevents bimodal initialization |
| `tau` | 10.0 | Softmax peakedness | Logits/10 keeps softmax near uniform initially, allows learning |
| `alpha` | 0.5 | Uniform mixing | Bounds weights to [0.08, 0.58], ensures diversification |
| `ema_decay` | 0.999 | Gradient memory | Window of ~1000 steps, smooth exploration decay over training |

**The key insight**: All scales are chosen relative to each other to ensure:
1. Initial weights near 1/6 (uniform)
2. Moderate exploration noise (not chaos)
3. Gradual learning over warm-start (~1000-5000 steps to converge)
4. Natural exploration decay as uncertainty reduces

---

## Recommended Settings

```python
cts = ContextualThompsonSampler(
    num_signals=6,
    context_dim=context_dim,
    lr=0.01,
    expl_scale=0.1,
    ema_decay=0.999,
    h0=1.0,
    tau=10.0,
    alpha=0.5,
    match_loss_weight=0.5,
    random_state=42
)

# Zero initialization for uniform start
cts.U = np.zeros_like(cts.U)
cts.b = np.zeros_like(cts.b)
```

**Expected behavior**:
- Initial weights: ~0.167 ± 0.05 for all signals (near uniform)
- After warm-start (3 epochs, ~21k gradient steps): weights reflect learned preferences from data
- Exploration noise decreases naturally as model gains confidence
