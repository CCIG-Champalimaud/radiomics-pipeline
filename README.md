# Radiomic feature extraction pipeline

Here I include scripts for the automated registration of images belonging to the same study and the consequent extraction of features from them.

## Extraction using `utils/extract-radiomic-features.py`

```
usage: extract-radiomic-features.py [-h] --input_paths INPUT_PATHS [INPUT_PATHS ...] --configs CONFIGS [CONFIGS ...] --mask_path MASK_PATH --target_spacing TARGET_SPACING [TARGET_SPACING ...] --output_path OUTPUT_PATH [--registration {mask,largest,first}] [--assume_same]

Extracts features for a given set of sequences and a single mask. In case the sequences or the mask are of different sizes, it maps all of the sequences to a common space given as input. Optionally, this script also registers all sequences assuming that the first sequence is the reference. The mask may also be registered if its input size is different from that of the first input sequence.

options:
  -h, --help            show this help message and exit
  --input_paths INPUT_PATHS [INPUT_PATHS ...]
                        Paths to sequences in nibabel compatible format.
  --configs CONFIGS [CONFIGS ...]
                        Paths to pyradiomics configuration files.
  --mask_path MASK_PATH
                        Path to mask in nibabel compatible format.
  --target_spacing TARGET_SPACING [TARGET_SPACING ...]
                        Target spacing for the inputs/mask.
  --output_path OUTPUT_PATH
                        Output path.
  --registration {mask,largest,first}
                        Registers all images to an image with the shape of the mask or to the largest image (registration is inferred from the first non-fixed image and
                        applied to other images).
  --assume_same         Assumes that if fixed and moving images/masks have the same shape they are already co-registered (or equivalent). Only for registration ==
                        'first'.
```

## Extraction using Snakemake

To extract features using the Snakemake pipeline (specified in `get-radiomic-features-pi-cai.smk`), one needs to edit `config.yaml` such that it contains the correct values. The one presented here was used on data from PI-CAI.