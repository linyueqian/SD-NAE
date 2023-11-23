# SD-NAE

This repo contains the code for the paper [Generating Natural Adversarial Examples with Stable Diffusion]().

Anonymous authors

---

## Overview

Natural Adversarial Examples (NAEs) are samples that naturally arise from the environment (rather than artifically created via pixel perturbation) yet fools the classifier into misclassification. NAEs are valuable for identifying the vulnerability and robustly measuring the performance of a classifier.

Early works collect NAEs by filtering from a huge set of real images. We argue that this is passive and relies on the assumption that NAEs exist in the candidate set in the first place. **In this work, we propose to synthesize NAEs using the powerful Stable Diffusion**.

See below the overview and generated examples by SD-NAE. For more details, please refer to our paper!

![overview of SD-NAE](figures/method.png)

![examples synthesized by SD-NAE](figures/main.png)
<p>
    <em>In each pair, the left one is generated with the initialized token embedding. Importantly, we make sure that all left images are correctly classified by the ImageNet ResNet-50 model in the first place. The right ones are the result of SD-NAE optimization when using the corresponding left one as initialization, and we mark the classifier's prediction in red above the image.</em>
</p>

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

## Reference

If you find our work/code helpful, please consider citing our work.

```
To be updated
```
