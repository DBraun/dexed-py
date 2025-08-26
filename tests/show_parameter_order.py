from faustax import FaustSynthesizer
from faustax.synthesizer_registry import get_synthesizer_class

from flax import nnx

DX7Algo16 = get_synthesizer_class("DX7Algo16")
rngs = nnx.Rngs(params=0, rng_stream=1)
faust_module = DX7Algo16(
    sample_rate=44100,
    soundfile_dirs=[],
    rngs=rngs
)

synth = FaustSynthesizer(faust_module)

for i, (full_label, blah) in enumerate(synth.get_label_to_param_mapping().items()):
    if blah["type"] == "nentry":
        offset = blah["categorical_offset"]
        size = blah["num_classes"]
        other = offset+size
        print(f"{str(offset).zfill(3)}-{str(other).zfill(3)}: {full_label}")
    else:
        offset = blah["continuous_idx"]
        print(f"{str(offset).zfill(3)}: {full_label}")

import jax
import jax.numpy as jnp
import numpy as np
flat_params = jnp.full((1, 185,), fill_value=0.5)  # todo:
inputs = jnp.ones((1, 3, 44100))
inputs = inputs.at[:, 0, :].set(440)  # set freq value

audio = synth.render_from_flat(flat_params, inputs, unroll=1, rng=jax.random.key(0))
assert audio.ndim == 3
audio = jnp.squeeze(audio, (0, 1))

from scipy.io import wavfile

wavfile.write("faustax_output.wav", 44100, np.array(audio))