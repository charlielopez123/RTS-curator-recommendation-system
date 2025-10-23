# Contextual Thompson Sampler - Hyperparameter Guide

This document explains the CTS hyperparameters and their relationship to parameter scales and weight distributions.

## Understanding the Model

The CTS learns to weight value signals based on context:

```
Context c_t → Model parameters (U, b) → Logits z_t → Softmax + Mixing → Weights w_t
```

The key insight: **hyperparameters control the SCALE of randomness and updates relative to the signal.**

---

## Recommended Production Settings

Based on experiments in [experiments/notebooks/IL_testing.ipynb](../experiments/notebooks/IL_testing.ipynb):

### CTS Initialization

```python
from cts_recommender.models.contextual_thompson_sampler import ContextualThompsonSampler
from cts_recommender.imitation_learning.IL_constants import PSEUDO_REWARD_WEIGHTS

cts = ContextualThompsonSampler(
    num_signals=6,              # 5 value signals + 1 curator signal
    context_dim=18,             # Context feature dimension
    lr=0.05,                    # Learning rate for online updates
    expl_scale=0.001,           # Exploration noise scale
    ema_decay=0.999,            # EMA decay for Hessian approximation
    h0=0.1,                     # Initial Hessian value
    tau=2.0,                    # Temperature for softmax
    alpha=0.3,                  # Uniform mixing weight
    weight_decay=1e-4,          # L2 regularization
    match_loss_weight=0.5,      # Weight for signal matching loss
    random_state=cfg.random_seed
)

# Initialize with target weights based on curator objectives
gamma = 0.3  # Curator signal weight (30%)
pseudo_weights = np.array([
    PSEUDO_REWARD_WEIGHTS['audience'],      # 0.4
    PSEUDO_REWARD_WEIGHTS['competition'],   # 0.15
    PSEUDO_REWARD_WEIGHTS['diversity'],     # 0.1
    PSEUDO_REWARD_WEIGHTS['novelty'],       # 0.2
    PSEUDO_REWARD_WEIGHTS['rights']         # 0.15
])

target_weights = np.zeros(6)
target_weights[:5] = (1 - gamma) * pseudo_weights  # [0.28, 0.105, 0.07, 0.14, 0.105]
target_weights[5] = gamma                          # 0.3

cts.initialize_with_target_weights(target_weights)
```

### Warm-Start Training

```python
history = cts.warm_start(
    contexts=context_features,
    signals=train_signals,
    rewards=reward_targets,
    epochs=1,
    lr=0.01,           # Reduced for stability (vs 0.05 for online)
    expl_scale=1e-4,   # Reduced for deterministic learning (vs 0.001)
    ema_decay=0.9999,  # Increased for smooth curvature (vs 0.999)
    weight_decay=cts.weight_decay,
    verbose=True
)
```

### Key Parameter Explanations

#### Learning & Exploration
- **`lr=0.05`**: Learning rate for online updates after warm-start
- **`expl_scale=0.001`**: Exploration noise variance scale. With `h0=0.1`, initial noise std ≈ 0.1
- **`h0=0.1`**: Initial Hessian diagonal value. Controls starting exploration magnitude

#### Softmax & Weight Distribution
- **`tau=2.0`**: Temperature for softmax. Higher values flatten the distribution, promoting more uniform weights
- **`alpha=0.3`**: Uniform mixing weight. **Prevents complete diminishing of any signal weight** - ensures all signals maintain minimum influence even when model heavily favors others. Final weights = 0.7×softmax + 0.3×uniform

#### EMA Decay
- **`ema_decay=0.999`**: EMA decay for Hessian approximation. **Extremely close to 1.0 because squared gradient updates are typically very small (≈1e-4 to 1e-6 scale)**, requiring slow accumulation to build accurate curvature estimates. Memory window ≈ 1/(1-0.999) = 1000 updates

#### Regularization
- **`weight_decay=1e-4`**: L2 regularization to prevent overfitting
- **`match_loss_weight=0.5`**: Weight for signal matching loss (when curator signals provided)

#### Weight Initialization
- **`gamma=0.3`**: Curator signal weight (30%). **Changed from initial gamma=0.6 to better align with curator objectives**
  - The original **gamma=0.6** was used during **IL training data generation** to put more emphasis on positive samples for extracting pseudo-rewards
  - For CTS deployment, **gamma=0.3** better reflects actual curator decision-making priorities
- Remaining 70% distributed proportionally across 5 value signals:
  - **Audience: 28%** (0.7 × 0.4) - Highest priority among value signals
  - **Novelty: 14%** (0.7 × 0.2)
  - **Competition: 10.5%** (0.7 × 0.15)
  - **Rights: 10.5%** (0.7 × 0.15)
  - **Diversity: 7%** (0.7 × 0.1)

#### Warm-Start Configuration
**These hyperparameters are temporarily lowered during offline warm-start to ensure stability throughout the many training samples (7,206 samples), then automatically restored for online learning:**

- **`lr=0.01`** (vs 0.05 online): Reduced learning rate prevents overshooting across large batch
- **`expl_scale=1e-4`** (vs 0.001 online): 10× less exploration noise for deterministic offline learning
- **`ema_decay=0.9999`** (vs 0.999 online): Even slower curvature adaptation prevents instability from rapid changes
- **`epochs=1`**: Single pass through historical data is sufficient with proper learning rate

---

## Hyperparameters and Their Scales

### `lr` - Learning Rate
**Scale**: `0.001` to `0.1`

**What it controls**: Size of parameter updates after each gradient step.

**Formula**: `U = U - lr * grad_U`

**Relationship to parameters**:
- Gradients have scale O(0.01-0.1) (from backprop through softmax and loss)
- If `lr = 0.01`, parameters change by ~0.001 per update
- Starting from `U = 0`, after 1000 updates with consistent gradient, `U` could reach O(1)

**Tuning logic**:
```
lr = 0.001  → Takes ~10000 steps to build up U to O(1) scale
lr = 0.01   → Takes ~1000 steps to build up U to O(1) scale  ✓ CHOSEN (warm-start)
lr = 0.05   → Takes ~200 steps, good for online adaptation  ✓ CHOSEN (online)
lr = 0.1    → Takes ~10 steps, but risks overshooting
```

**Why 0.01 for warm-start, 0.05 for online?**
- Warm-start has ~7000 samples, 1 epoch → ~7,000 gradient steps
- Want stable convergence over these samples without overshooting
- Online learning needs faster adaptation to new patterns → higher lr

---

### `expl_scale` - Exploration Noise Variance Scale
**Scale**: `0.0001` to `1.0`

**What it controls**: Variance of Gaussian noise added to parameters during Thompson Sampling.

**Formula**:
```python
noise_std_U = sqrt(expl_scale / h_U)
U_tilde = U + N(0, noise_std_U)
```

**Critical relationship**:
```
expl_scale = 0.0001, h0 = 0.1  →  initial noise_std = 0.032 (minimal, warm-start) ✓ CHOSEN (warm-start)
expl_scale = 0.001,  h0 = 0.1  →  initial noise_std = 0.1   (moderate, online) ✓ CHOSEN (online)
expl_scale = 0.01,   h0 = 0.1  →  initial noise_std = 0.316 (high exploration)
expl_scale = 1.0,    h0 = 0.1  →  initial noise_std = 3.16  (very high)
```

**Relationship to logits and weights**:
- Logits `z = U @ c + b` initially O(noise_std) since U~noise
- With `noise_std = 0.1` and context features O(1):
  - Logits `z ~ N(0, 0.1)` → range approximately [-0.3, 0.3]
  - After temperature `z/tau = z/2` → range [-0.15, 0.15]
  - softmax([-0.15, 0.05, 0.1, ...]) ≈ moderate variation around uniform

**Why 0.001 for online, 0.0001 for warm-start?**
- Keeps initial logits moderate, allowing meaningful Thompson sampling
- Warm-start uses 10× less noise for deterministic learning from historical data
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

**Scale analysis (with expl_scale=0.001)**:
```
h0 = 0.001  →  noise_std = sqrt(0.001/0.001) = 1.0   (massive noise)
h0 = 0.01   →  noise_std = sqrt(0.001/0.01)  = 0.316 (large noise)
h0 = 0.1    →  noise_std = sqrt(0.001/0.1)   = 0.1   (moderate noise) ✓ CHOSEN
h0 = 1.0    →  noise_std = sqrt(0.001/1.0)   = 0.032 (low noise)
h0 = 10.0   →  noise_std = sqrt(0.001/10.0)  = 0.01  (very low noise)
```

**Relationship to learning**:
- `h_U` accumulates squared gradients: `h_U = 0.999 * h_U + 0.001 * grad^2`
- If gradients are O(0.01-0.1), squared gradients are O(0.0001-0.01)
- After many steps, `h_U` converges to average squared gradient magnitude
- Noise decreases as `h_U` grows, naturally reducing exploration over time

**Why 0.1?**
- With `h0 = 0.1`, initial noise_std = 0.1 provides moderate exploration
- Logits remain O(0.1-0.3), preventing extreme peaked or flat softmax
- Balances exploration (via noise) with learning (signal from data)

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
tau = 2:   softmax → [0.08, 0.13, 0.16, 0.21, 0.13, 0.10]  (moderate) ✓ CHOSEN
tau = 5:   softmax → [0.13, 0.16, 0.17, 0.19, 0.16, 0.14]  (flatter)
tau = 10:  softmax → [0.15, 0.17, 0.17, 0.18, 0.17, 0.16]  (very flat)
tau = 20:  softmax → [0.16, 0.17, 0.17, 0.17, 0.17, 0.16]  (nearly uniform)
```

**Intuition**:
- Low tau (1): "Confident" - strongly favors highest logit
- Medium tau (2-5): "Balanced" - allows differentiation while maintaining diversity
- High tau (10-20): "Uncertain" - hedges bets across all signals
- tau → ∞: Always uniform (1/6 for each signal)

**Why 2.0?**
- Allows learned preferences to influence weights while preventing over-concentration
- Balances exploitation (following learned preferences) with maintaining signal diversity
- Works well with `alpha=0.3` mixing to ensure all signals remain influential

---

### `alpha` - Uniform Mixing Weight
**Scale**: `0.0` to `1.0`

**What it controls**: How much to mix softmax output with uniform distribution. **Prevents complete diminishing of any signal weight.**

**Formula**: `w = (1 - alpha) * softmax(z/tau) + alpha * (1/num_signals)`

**Weight bounds (alpha=0.3)**:
```
Min weight = 0.3 / 6 = 0.05  (5%)
Max weight = 0.7 + 0.05 = 0.75 (75%)
```

**Why 0.3?**
- Ensures all signals maintain minimum influence even when model heavily favors others
- Allows meaningful differentiation while preventing complete dominance

---

### `ema_decay` - Exponential Moving Average Decay
**Scale**: `0.9` to `0.9999`

**What it controls**: How quickly Hessian approximation adapts to gradient magnitude changes. **Extremely close to 1.0 because squared gradient updates are typically very small (≈1e-4 to 1e-6 scale)**, requiring slow accumulation.

**Formula**: `h_U = ema_decay * h_U + (1 - ema_decay) * grad_U^2`

**Effective memory window**: `1 / (1 - ema_decay)`

**Scale analysis**:
```
ema_decay = 0.9    →  memory ≈ 10 updates
ema_decay = 0.99   →  memory ≈ 100 updates
ema_decay = 0.999  →  memory ≈ 1000 updates (online) ✓ CHOSEN
ema_decay = 0.9999 →  memory ≈ 10000 updates (warm-start) ✓ CHOSEN
```

**Why 0.999 for online, 0.9999 for warm-start?**
- Gradients squared are O(1e-4 to 1e-6), accumulate very slowly
- Higher decay = smoother curvature estimates = more stable exploration noise
- Warm-start uses even higher decay for maximum stability across large batch

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

## Summary: Production Settings

| Hyperparameter | Online | Warm-Start | Reasoning |
|---------------|--------|------------|-----------|
| `lr` | 0.05 | 0.01 | Faster online adaptation vs stable offline learning |
| `expl_scale` | 0.001 | 0.0001 | Thompson sampling vs deterministic learning |
| `h0` | 0.1 | 0.1 | Moderate initial exploration |
| `tau` | 2.0 | 2.0 | Balanced differentiation with diversity |
| `alpha` | 0.3 | 0.3 | Prevents signal weight collapse (min 5%) |
| `ema_decay` | 0.999 | 0.9999 | Slow accumulation for small squared gradients |
| `weight_decay` | 1e-4 | 1e-4 | L2 regularization |
| `gamma` (init) | 0.3 | 0.3 | Curator signal weight (changed from 0.6 in IL data generation) |

**Key insights**:
1. Weight initialization uses `gamma=0.3` (vs 0.6 for IL training) to align with curator objectives
2. Warm-start hyperparameters are lowered for stability across 7,206 samples
3. All signals maintain minimum 5% influence via `alpha=0.3` mixing
4. `ema_decay` near 1.0 due to small squared gradient scale (1e-4 to 1e-6)
