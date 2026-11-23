# External dependencies

Fairmotion loads and displays motion data. UnderPressure is an optional foot-contact
comparison model; the default single-detection pipeline does not use it.
Both repositories are pinned submodules and retain their upstream licenses.
UnderPressure is an InterDigital product.

## Setup

```sh
git submodule update --init -- third_party/fairmotion
# Optional contact-model comparison:
git submodule update --init -- third_party/UnderPressure
```

Python dependencies are listed in the [project installation](../README.md#installation).
The pinned UnderPressure checkout includes its pretrained weights and reference sample.
See the [Fairmotion license](fairmotion/LICENSE) and
[UnderPressure license](UnderPressure/LICENCE.txt) for their respective terms.

## Run

```sh
python source/visualize_contacts.py --bvh-file data/canonical_bvh/DATE/clip.bvh --contact-config preprocess/training/underpressure_offline_comparison.json --no-mesh
```

Offline UnderPressure uses future frames and is for comparison only. The
`underpressure.json` configuration uses past/current prefixes with the unchanged
model; it is slower and has not been retrained as a causal detector.
