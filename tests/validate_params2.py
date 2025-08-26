from faustax import FaustSynthesizer
from faustax.synthesizer_registry import get_synthesizer_class
from flax import nnx

DX7Algo16 = get_synthesizer_class('DX7Algo16')
rngs = nnx.Rngs(params=0, rng_stream=1)
faust_module = DX7Algo16(sample_rate=44100, soundfile_dirs=[], rngs=rngs)
synth = FaustSynthesizer(faust_module)
mapping = synth.get_label_to_param_mapping()

cont_count = 0
cat_count = 0
for p in mapping.values():
    if p.get("type") == "nentry":
        cat_count += 1
    else:
        cont_count += 1

print(f'Total params: {len(mapping)}')
print(f'Continuous params: {cont_count}')
print(f'Categorical params: {cat_count}')
