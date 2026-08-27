# Quick Start

## Using the Patch API

`Patch` is the human-friendly interface — parameters use native DX7 ranges (e.g. 0-99 for levels) and string names (e.g. `"sine"` for LFO wave). Use it for sound design, sysex import/export, and interactive exploration.

```python
from dexed import Patch, DexedSynth

# Create a patch with named parameter access (algorithm is 0-indexed)
patch = Patch(name="My Sound")
patch.algorithm = 15        # 0-31
patch.feedback = 5
patch.op[0].output_level = 99
patch.op[0].envelope.rates = [99, 85, 35, 50]
patch.op[0].envelope.levels = [99, 75, 0, 0]
patch.lfo.wave = "sine"

# Load and render
synth = DexedSynth(sample_rate=44100)
synth.load_patch(patch)
audio = synth.render(midi_note=60, velocity=100, note_duration=1.0, render_duration=1.5)

audio2 = synth.render(midi_note=64, velocity=80)
audio3 = synth.render(midi_note=67, velocity=120)
```

## Loading and Saving DX7 Sysex Files

DX7 sysex banks are 4104-byte `.syx` bulk dumps containing 32 patches: a 6-byte header, a 4096-byte payload, a checksum and an end-of-exclusive byte. A raw 4096-byte payload with no framing loads too. Individual voice dumps are 163 bytes (a 6-byte sysex header + 155 data bytes + 2 trailing bytes).

```python
from dexed import Patch

# Load a 32-voice bank
patches = Patch.load_bank("rom1a.syx")
print(f"Loaded {len(patches)} patches")

for i, p in enumerate(patches[:5]):
    print(f"  {i}: {p.name.strip()} (algorithm {p.algorithm})")

# Save patches to a bank file
Patch.save_to_bank("my_bank.syx", patches)

# Load from an individual voice dump (skip 6-byte sysex header)
with open("voice.syx", "rb") as f:
    data = f.read()
patch = Patch.from_sysex(data[6:162])  # 156 bytes of unpacked voice data
```

## ML / JAX Workflow with Preset

`Preset` is the ML-native representation: all parameters in a single flat (145,) float32 array. Continuous params are normalized to [0, 1]; discrete params (algorithm, curves, etc.) are int arrays. All fields are JAX PyTree data leaves — changing any field, including `algorithm`, **never triggers JIT recompilation**.

```python
import numpy as np
from dexed import Patch, Preset, DexedSynth

# Convert a sysex patch to a Preset
preset = Patch.load_bank("rom1a.syx")[0].to_preset()

# Or construct directly
preset = Preset(
    algorithm=15,               # 0-31
    feedback=0.5,               # normalized [0, 1]
    op_output_level=np.full(6, 0.8, dtype=np.float32),
)

# Load and render
synth = DexedSynth()
synth.load_preset(preset)
audio = synth.render(midi_note=60, velocity=100)

# Flat array round-trip — everything in one (145,) float32 vector
arr     = preset.to_array()
preset2 = Preset.from_array(arr)

# Bulk storage: 100k presets ~ 55 MB
bank = np.stack([p.to_array() for p in presets])  # (N, 145)
np.save("bank.npy", bank)
presets = [Preset.from_array(row) for row in np.load("bank.npy")]
```

## Converting Between Patch and Preset

The two representations convert losslessly in both directions:

```python
from dexed import Patch, Preset

# Patch -> Preset
patch = Patch(name="My Sound")
patch.algorithm = 15
preset = patch.to_preset()

# Preset -> Patch
patch2 = preset.to_patch()

# Classmethod alternatives
preset = Preset.from_patch(patch)
```

## JAX pure_callback

```python
import jax
import jax.numpy as jnp
from dexed import DexedSynth, Preset

synth = DexedSynth()

def render_fn(preset):
    synth.load_preset(preset)
    return synth.render(midi_note=60, velocity=100,
                        note_duration=0.5, render_duration=1.0)

@jax.jit
def jitted_render(preset):
    return jax.pure_callback(
        render_fn,
        jax.ShapeDtypeStruct((44100,), jnp.float32),
        preset,
    )

audio = jitted_render(Preset(algorithm=0, feedback=0.5))

# Changing any field — including algorithm — does NOT recompile
audio2 = jitted_render(Preset(algorithm=15, feedback=0.3))
```

For flat-vector policies (e.g. sampling all 145 dims from a Beta distribution):

```python
# One treedef works for every Preset — no meta fields
_, treedef = jax.tree.flatten(Preset())

@jax.jit
def render_from_flat(flat_params):   # (145,) float32
    preset = jax.tree.unflatten(treedef, Preset.array_to_leaves(flat_params))
    return jax.pure_callback(render_fn, jax.ShapeDtypeStruct((44100,), jnp.float32), preset)
```

## Algorithm Metadata

The 32 standard DX7 algorithms define which operators are carriers (output to audio) and which are modulators (modulate other operators). All indices are 0-based.

```python
from dexed import algorithms, get_carriers, get_modulators, get_mod_matrix

alg = algorithms[15]
print(f"Algorithm 15 carriers: {alg.carriers}")
print(f"Algorithm 15 modulators: {alg.modulators}")
print(f"Feedback edge: {alg.feedback_edge}")  # (source, target) — same op for self-loops
print(f"Modulation matrix:\n{alg.mod_matrix}")   # 6x6 int8

carriers = get_carriers(31)   # [0, 1, 2, 3, 4, 5] — all parallel
```

## Individual Operator Outputs

Render each operator's contribution separately — useful for analysis and visualization:

```python
from dexed import Patch, DexedSynth

synth = DexedSynth(sample_rate=44100)
patch = Patch(name="Test")
patch.algorithm = 0
patch.op[0].output_level = 99
synth.load_patch(patch)

# 7-channel output: operators 0-5 + final mix
audio = synth.render_all_ops(midi_note=60)
# audio.shape = (7, num_samples)
# audio[0..5] = Operators 0-5, audio[6] = Final mix
```

## Feedback Normalization

By default, algorithms 3, 5, and 31 (DX7 algorithms 4, 6, 32) have reduced feedback, matching original Dexed/DX7 behavior. For consistent feedback across all algorithms (useful for ML pipelines where feedback should be comparable across algorithms):

```python
synth = DexedSynth()
synth.normalize_feedback = True
synth.load_patch(patch)
audio = synth.render(midi_note=60)
```

## Custom Operator Graphs

`OperatorGraph` lets you build arbitrary FM topologies — not limited to the 32 standard DX7 algorithms, and not limited to 6 operators.

```python
from dexed import OperatorGraph

# Create a 7-operator graph (beyond DX7's 6)
graph = OperatorGraph(num_ops=7)

for i in range(7):
    graph.op[i].output_level = 99
    graph.op[i].frequency_coarse = i + 1

# Chain: op6 -> op5 -> ... -> op1 -> op0 (carrier)
for i in range(6, 0, -1):
    graph.connect(i, i - 1)

graph.set_carriers([0])
graph.set_feedback(6, 6, level=7)   # source, target, level (0-7)

audio = graph.render(sample_rate=44100, midi_note=60, velocity=100,
                     note_duration=1.0, render_duration=1.5)

# Render each operator individually
audio_all = graph.render_all_ops(midi_note=60)  # (8, T) for 7 ops + final
```

### From a Modulation Matrix

```python
import numpy as np
from dexed import OperatorGraph

mod_matrix = np.zeros((4, 4), dtype=np.float32)
mod_matrix[0, 1] = 1.0  # Op 1 modulates Op 0
mod_matrix[1, 2] = 1.0  # Op 2 modulates Op 1
graph = OperatorGraph.from_matrix(mod_matrix, carriers=[0], feedback={(3, 3): 5})
```

### From a Standard DX7 Algorithm

```python
from dexed import OperatorGraph

# Creates a 6-op graph matching the DX7 algorithm (0-indexed)
graph = OperatorGraph.from_algorithm(15)
```

### Visualization

```python
graph = OperatorGraph.from_algorithm(0)

print(graph.summary())     # human-readable text description
print(graph.to_ascii())    # ASCII art of the signal flow
print(graph.to_mermaid())  # paste into any Mermaid renderer
```
