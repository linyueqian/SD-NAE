# SD-NAE

This repo contains the code for the paper [Generating Natural Adversarial Examples with Stable Diffusion](https://openreview.net/forum?id=D87rimdkGd).

Authors: Yueqian Lin*, Jingyang Zhang*, Yiran Chen, Hai Li

---

## Environment setup

To set up the environment for using SD-NAE:

```bash
conda create -n sdnae python=3.9
conda activate sdnae
python -m pip install torch torchvision torchaudio # Version 2.1.0
python -m pip install xformers diffusers transformers accelerate pandas
```

## Using SD-NAE

Use `generate.py` to synthesize natural adversarial examples yourself!

```bash
# Example: generate adversarial examples for class 20, which is 'jellyfish'
# All default hyperparameters (encoded in generate.py) are used
python generate.py --class_ids 20 20
```