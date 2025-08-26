# Parameter Format

`Preset.to_array()` packs all synth state into a single **(145,) float32** vector.

The first 123 values are **continuous floats in [0, 1]**. The last 22 values are **integers** (stored as float32 in the array, recovered by rounding on `from_array()`). The integer maxes are documented in the table below, making it straightforward to specify independent sampling distributions for each section.

You can create a Preset from a Patch (`patch.to_preset()` or `Preset.from_patch(patch)`), from a flat array (`Preset.from_array(arr)`), or by constructing one directly with keyword arguments.

All 145 values are JAX PyTree data leaves — changing any value, including `algorithm`, never triggers JIT recompilation.

## Layout

### Continuous floats — indices 0–122

#### Global (15 values)

| Index | Field | Native range |
|-------|-------|-------------|
| 0 | feedback | 0–7 |
| 1 | transpose | 0–48 |
| 2 | pitch_mod_sensitivity | 0–7 |
| 3 | lfo_speed | 0–99 |
| 4 | lfo_delay | 0–99 |
| 5 | lfo_pitch_mod_depth | 0–99 |
| 6 | lfo_amp_mod_depth | 0–99 |
| 7–10 | pitch_env_rates (4) | 0–99 |
| 11–14 | pitch_env_levels (4) | 0–99 |

#### Per-operator (108 values, 6 ops × 18 floats each)

The 6-operator fields are packed in **op-major order**: all values for op 0 come first, then op 1, and so on through op 5. For example, `op_env_rates` occupies indices 15–38 as:

```
[op0_r0, op0_r1, op0_r2, op0_r3,  op1_r0, ..., op1_r3,  ...,  op5_r0, ..., op5_r3]
 ─────── op 0 ──────────           ─────── op 1 ───           ─────── op 5 ───
```

The same pattern applies to every per-operator field in the tables below.

| Index range | Field | Native range |
|-------------|-------|-------------|
| 15–38 | op_env_rates (6 × 4) | 0–99 |
| 39–62 | op_env_levels (6 × 4) | 0–99 |
| 63–68 | op_output_level (6) | 0–99 |
| 69–74 | op_frequency_coarse (6) | 0–31 |
| 75–80 | op_frequency_fine (6) | 0–99 |
| 81–86 | op_detune (6) | 0–14 |
| 87–92 | op_velocity_sensitivity (6) | 0–7 |
| 93–98 | op_amp_mod_sensitivity (6) | 0–3 |
| 99–104 | op_rate_scaling (6) | 0–7 |
| 105–110 | op_breakpoint (6) | 0–99 |
| 111–116 | op_left_depth (6) | 0–99 |
| 117–122 | op_right_depth (6) | 0–99 |

### Integers — indices 123–144

#### Global (4 values)

| Index | Field | Max |
|-------|-------|-----|
| 123 | osc_key_sync | 1 |
| 124 | lfo_sync | 1 |
| 125 | algorithm | 31 |
| 126 | lfo_wave | 5 |

#### Per-operator (18 values, 6 ops × 3 ints each)

| Index range | Field | Max |
|-------------|-------|-----|
| 127–132 | op_frequency_mode (6) | 1 |
| 133–138 | op_left_curve (6) | 3 |
| 139–144 | op_right_curve (6) | 3 |

## Approach A — flat black-box

```python
from dexed import Patch, Preset
import numpy as np

# Patch → Preset
preset = Patch.load_bank("bank.syx")[0].to_preset()

# Round-trip
arr     = preset.to_array()           # (145,) float32
preset2 = Preset.from_array(arr)

# Sample: 123 floats from Beta, 22 ints from Categoricals
floats   = beta_distribution.sample((123,))
integers = np.array([
    np.random.randint(0, mx + 1)
    for mx in Preset.GLOBAL_INT_MAXES             # [1, 1, 31, 5]
] + [
    np.random.randint(0, mx + 1)
    for mx in Preset.OP_INT_MAXES * 6             # [1, 3, 3] × 6 ops
])
preset = Preset.from_array(np.concatenate([floats, integers]).astype(np.float32))

# Bulk storage: 100k presets ≈ 55 MB
bank = np.stack([p.to_array() for p in presets])  # (N, 145) float32
np.save("bank.npy", bank)
presets = [Preset.from_array(row) for row in np.load("bank.npy")]
```

## Approach B — per-operator black-box

A network predicts global params and one bundle per operator. The spec is
self-contained: no DX7 knowledge needed, just array shapes and integer maxes.

```python
# Spec (class constants — no DX7 knowledge required)
Preset.GLOBAL_CONTINUOUS_SIZE  # 15
Preset.GLOBAL_INT_MAXES        # [1, 1, 31, 5]  (osc_key_sync, lfo_sync, algorithm, lfo_wave)
Preset.OP_CONTINUOUS_SIZE      # 18
Preset.OP_INT_MAXES            # [1, 3, 3]       (frequency_mode, left_curve, right_curve)

# Decompose a Preset into bundles
gc = preset.global_continuous()   # (15,)   float32 in [0, 1]
gi = preset.global_ints()         # (4,)    int32
oc = preset.op_continuous()       # (6, 18) float32 in [0, 1]  — row i = operator i
oi = preset.op_ints()             # (6, 3)  int32

# Reconstruct from bundles
preset = Preset.from_operator_bundles(gc, gi, oc, oi)
```

A network in this style:

```python
# Network predicts all four arrays; sample independently per array
global_cont = beta_dist.sample((Preset.GLOBAL_CONTINUOUS_SIZE,))
global_int  = np.array([np.random.randint(0, mx + 1)
                         for mx in Preset.GLOBAL_INT_MAXES], dtype=np.int32)
op_cont     = beta_dist.sample((6, Preset.OP_CONTINUOUS_SIZE))
op_int      = np.array([[np.random.randint(0, mx + 1)
                          for mx in Preset.OP_INT_MAXES]
                         for _ in range(6)], dtype=np.int32)

preset = Preset.from_operator_bundles(global_cont, global_int, op_cont, op_int)
synth.load_preset(preset)
```

The per-operator column ordering within `op_continuous()` is:
env_rates[0..3], env_levels[0..3], output_level, frequency_coarse, frequency_fine,
detune, velocity_sensitivity, amp_mod_sensitivity, rate_scaling, breakpoint, left_depth, right_depth.

## JAX PyTree

All Presets share one treedef (no meta fields):

```python
import jax
from dexed import Preset

_, treedef = jax.tree.flatten(Preset())  # universal

# JIT-traceable flat → Preset
leaves = Preset.array_to_leaves(flat_array)  # 28 arrays/scalars
preset = jax.tree.unflatten(treedef, leaves)
```
