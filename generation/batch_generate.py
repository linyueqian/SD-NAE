import os
import sys

sys.path.append("..")
import argparse
import random
import string
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from torchvision import transforms as trn
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoImageProcessor, ResNetForImageClassification


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


from class_names import (cifar10_classes, cifar100_classes, imagenet_classes,
                         in100_classes)

from models.wrn import WideResNet


class CustomEmbedding(nn.Module):
    def __init__(self, embedding_weights, update_index):
        super().__init__()
        num_embeddings = embedding_weights.shape[0]
        self.zero_index = torch.LongTensor([0]).to(embedding_weights.device)
        self.weights = nn.ModuleList(
            [
                nn.Embedding.from_pretrained(
                    embedding_weights[i : i + 1], freeze=i != update_index
                )
                for i in range(num_embeddings)
            ]
        )

    def forward(self, x):
        return torch.cat([self.weights[xx.item()](self.zero_index) for xx in x[0]])


# ----------------------------- #
# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar100",
    choices=["cifar100", "in100", "cifar10", "imagenet"],
)
parser.add_argument(
    "--sigma",
    type=float,
    default=0.07,
    help="the noise variance used when sampling outlier embeddings",
)
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--guidance_scale", type=float, default=7.5)
parser.add_argument("--num_inference_steps", type=int, default=20)
parser.add_argument("--num_samples_per_class", type=int, default=20)
parser.add_argument("--class_ids", type=int, nargs="+", default=[0, 9])
parser.add_argument("--num_grad_steps", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--tag", type=str, default="task1")
parser.add_argument("--gpu", type=int, default=4)
# ----------------------------- #


def forward_diffusion(
    self,
    latents: torch.Tensor,
    all_embeddings: torch.Tensor,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    eta: float = 0.0,
):
    self.scheduler.set_timesteps(num_inference_steps)
    timesteps_tensor = self.scheduler.timesteps.to(self.device)
    extra_step_kwargs = self.prepare_extra_step_kwargs(self, eta)

    for i, t in tqdm(
        enumerate(timesteps_tensor),
        desc="Diffusion",
        total=len(timesteps_tensor),
        leave=False,
    ):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=all_embeddings,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        latents = self.scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs, return_dict=False
        )[0]

    latents = latents / self.vae.config.scaling_factor
    image = self.vae.decode(latents, return_dict=False)[0]

    return self.image_processor.postprocess(
        image, output_type="pt"
    )  # pixel value [0, 1]


def forward_classifier(x, preprocessor, clf, device, dataset=None):
    if dataset == "imagenet":
        inputs = preprocessor(x)
        # to tensor and pt
        # inputs = inputs.unsqueeze(0).to(device)
        x = clf(inputs).logits
    else:
        x = preprocessor(x)
        x = clf(x)
    return x


@torch.no_grad()
def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # preparation
    assert (
        len(args.class_ids) == 2 and args.class_ids[1] >= args.class_ids[0]
    ), "Please specify the start and end class id (end >= start)."
    start, end = args.class_ids
    all_class_labels = eval(f"{args.dataset}_classes")

    # img_save_root = f'./outliers/{args.dataset}/{args.sigma:.2f}-alpha{args.alpha:.1f}_{args.dataset}_img{args.img_size}_steps{args.num_inference_steps}_grad{args.num_grad_steps}_lr{args.lr}_k{args.k}_{args.tag}'
    img_save_root = f"./outliers/{args.tag}/{args.dataset}"
    embed_save_root = img_save_root.replace("outliers", "embeds")
    log_save_root = img_save_root.replace("outliers", "logs")

    # load stable diffusion
    pipe = StableDiffusionPipeline.from_pretrained(
        "bguisard/stable-diffusion-nano-2-1",
        torch_dtype=torch.float32,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    # uncond embedding
    uncond_tokens = [""]
    uncond_input = pipe.tokenizer(
        uncond_tokens,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

    # just for easy calling
    original_tk_embed_layer = deepcopy(
        pipe.text_encoder.text_model.embeddings.token_embedding
    )
    if args.tag == "task4":
        r1 = original_tk_embed_layer.weight.data.min().item()
        r2 = original_tk_embed_layer.weight.data.max().item()
        torch.manual_seed(3407)
        # ood_token_embeds = (r1 - r2) * torch.rand(100, 1000, 1024) + r2
        # assert args.num_samples_per_class <= 1000
        variance_factor = torch.rand(
            1
        ).item()  # This could be any function that provides variability
        dynamic_r1 = r1 * variance_factor
        dynamic_r2 = r2 * variance_factor
        # Generating the embeddings with the dynamic range
        ood_token_embeds = (dynamic_r2 - dynamic_r1) * torch.rand(
            100, 1000, 1024
        ) + dynamic_r1
    # load classifier
    if args.dataset == "cifar100":
        clf = WideResNet(depth=40, num_classes=100, widen_factor=2)
        clf.load_state_dict(torch.load("ckpt/seed_233.pth", map_location="cpu"))
        clf.to(device)
        clf.eval()
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        preprocessor = trn.Compose(
            [trn.Resize((32, 32), antialias=True), trn.Normalize(mean, std)]
        )
    elif args.dataset == "cifar10":
        clf = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
        )
        clf.to(device)
        clf.eval()
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        preprocessor = trn.Compose(
            [trn.Resize((32, 32), antialias=True), trn.Normalize(mean, std)]
        )
    elif args.dataset == "imagenet":
        preprocessor = trn.Compose(
            [
                trn.Resize((224, 224), antialias=True),
                trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # preprocessor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        clf = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        clf.to(device)
        clf.eval()
    else:
        raise NotImplementedError

    cos = nn.CosineSimilarity(dim=-1)

    for i in tqdm(
        range(start, end + 1), total=end - start + 1, desc="Classes", leave=True
    ):
        # prepare prompt
        label = all_class_labels[i]
        if args.dataset == "imagenet":
            label = eval(f"in100_classes")[i]
            i = eval(f"imagenet_classes").index(label)
        # prompt = f"A high-quality image of a {label}"
        if args.tag == "task4":
            # random_text = get_random_string(len(label))
            # label = f"{random_text}"
            # print(f'random_text: {random_text}')
            label = "x"
        prompt = f"{label}"
        # special cases, we need more prompts for help, otherwise the generated images will not look like ImageNet classes.
        # https://github.com/deeplearning-wisc/dream-ood/blob/main/scripts/dream_ood.py
        if label == "kite" or label == "quail":
            prompt += " bird"
        if label == "chest":
            prompt += " box"
        if label == "tick":
            prompt += " bite"
        if label == "stingray":
            prompt += " in the water"
        if label == "ox" or label == "impala":
            prompt += " animal"
        if label == "nail":
            # prompt = 'A high-quality image of the wire nail'
            prompt = "wire nail"

        # prepare for text embeddings
        text_input = pipe.tokenizer(
            [prompt],
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_input.input_ids
        text_embeddings = pipe.text_encoder(text_input_ids.to(device))[0]
        label_id = torch.tensor([i], device=device)
        temp_input = pipe.tokenizer(
            [label],
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        temp_input_ids = temp_input.input_ids
        token_id = temp_input_ids[0][1]
        # save dir
        img_save_dir = os.path.join(img_save_root, f"{i:02d}_{label.replace(' ', '_')}")
        os.makedirs(img_save_dir, exist_ok=True)
        embed_save_dir = os.path.join(
            embed_save_root, f"{i:02d}_{label.replace(' ', '_')}"
        )
        os.makedirs(embed_save_dir, exist_ok=True)
        log_save_dir = os.path.join(log_save_root, f"{i:02d}_{label.replace(' ', '_')}")
        os.makedirs(log_save_dir, exist_ok=True)

        for j in tqdm(
            range(args.num_samples_per_class),
            total=args.num_samples_per_class,
            desc="Samples",
            leave=False,
        ):
            # if the image is already generated, skip
            # if os.path.isfile(os.path.join(log_save_dir, f'sample_{j:03d}.csv')):
            #     continue
            # init latents
            torch.manual_seed(j)
            latents_shape = (
                1,
                pipe.unet.config.in_channels,
                args.img_size // pipe.vae_scale_factor,
                args.img_size // pipe.vae_scale_factor,
            )
            latents = torch.randn(
                latents_shape, device=device, dtype=uncond_embeddings.dtype
            )
            latents = latents * pipe.scheduler.init_noise_sigma
            # replace the token embedding with the dream outlier embedding
            embedding_weights = original_tk_embed_layer.weight.data.clone()
            if args.tag == "task4":
                embedding_weights[token_id : token_id + 1] = ood_token_embeds[
                    i, j : j + 1
                ].to(device)
            pipe.text_encoder.text_model.embeddings.token_embedding = CustomEmbedding(
                embedding_weights, token_id
            )
            # optimization
            if args.tag == "task3":
                all_metrics = {
                    "loss": [],
                    "ce_loss": [],
                    "embed_loss": [],
                    "prob_target_class": [],
                    "prob_max": [],
                    "pred_class": [],
                    "kl_div": [],
                }
            elif args.tag == "task4":
                all_metrics = {
                    "loss": [],
                    "ce_loss": [],
                    # 'embed_loss': [],
                    "prob_target_class": [],
                    "prob_max": [],
                    "pred_class": [],
                }
            else:
                all_metrics = {
                    "loss": [],
                    "ce_loss": [],
                    # 'embed_loss': [],
                    "prob_target_class": [],
                    "prob_max": [],
                    "pred_class": [],
                }
            opt_embeds = []

            with torch.enable_grad():
                variable = (
                    pipe.text_encoder.text_model.embeddings.token_embedding.weights[
                        token_id
                    ].weight
                )
                original_embed = variable.detach().clone()
                optimizer = torch.optim.Adam([variable], lr=args.lr)

                tqdm.write("")
                tqdm.write(f"Class: {i:02d} - {label}, Sample {j:03d}")
                for step in tqdm(
                    range(args.num_grad_steps),
                    total=args.num_grad_steps,
                    desc="Optimize",
                    leave=False,
                ):
                    text_embeddings = pipe.text_encoder(text_input_ids)[0]
                    image = forward_diffusion(
                        pipe,
                        latents,
                        torch.cat([uncond_embeddings, text_embeddings]),
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        eta=0.0,
                    )

                    output = forward_classifier(
                        image, preprocessor, clf, device, dataset=args.dataset
                    )

                    if args.tag == "task1":
                        # task1: negative cross entropy loss
                        ce_loss = F.cross_entropy(output, label_id)
                        loss = -ce_loss
                    if args.tag == "task2" and args.dataset == "cifar100":
                        if label == "apples":
                            desired_label_id = torch.tensor(
                                [cifar100_classes.index("oranges")], device=device
                            )
                        elif label == "aquarium fish":
                            desired_label_id = torch.tensor(
                                [cifar100_classes.index("flatfish")], device=device
                            )
                        elif label == "baby":
                            desired_label_id = torch.tensor(
                                [cifar100_classes.index("boy")], device=device
                            )
                        elif label == "bear":
                            desired_label_id = torch.tensor(
                                [cifar100_classes.index("wolf")], device=device
                            )
                        elif label == "beaver":
                            desired_label_id = torch.tensor(
                                [cifar100_classes.index("otter")], device=device
                            )
                        elif label == "bed":
                            desired_label_id = torch.tensor(
                                [cifar100_classes.index("couch")], device=device
                            )  # or 'chair'
                        elif label == "bee":
                            desired_label_id = torch.tensor(
                                [cifar100_classes.index("butterfly")], device=device
                            )
                        elif label == "beetle":
                            desired_label_id = torch.tensor(
                                [cifar100_classes.index("cockroach")], device=device
                            )  # or 'caterpillar'
                        elif label == "bicycle":
                            desired_label_id = torch.tensor(
                                [cifar100_classes.index("motorcycle")], device=device
                            )
                        elif label == "bottles":
                            desired_label_id = torch.tensor(
                                [cifar100_classes.index("cans")], device=device
                            )
                        ce_loss = F.cross_entropy(output, desired_label_id)
                        loss = ce_loss
                    if args.tag == "task2" and args.dataset == "imagenet":
                        if label == "stingray":
                            desired_label_id = torch.tensor(
                                [imagenet_classes.index("electric ray")], device=device
                            )
                        elif label == "hen":
                            desired_label_id = torch.tensor(
                                [imagenet_classes.index("rooster")], device=device
                            )
                        elif label == "magpie":
                            desired_label_id = torch.tensor(
                                [imagenet_classes.index("jay")], device=device
                            )
                        elif label == "kite":
                            desired_label_id = torch.tensor(
                                [imagenet_classes.index("bald eagle")], device=device
                            )
                        elif label == "vulture":
                            desired_label_id = torch.tensor(
                                [imagenet_classes.index("bald eagle")], device=device
                            )
                        elif label == "agama":
                            desired_label_id = torch.tensor(
                                [imagenet_classes.index("alligator lizard")],
                                device=device,
                            )
                        elif label == "tick":
                            desired_label_id = torch.tensor(
                                [imagenet_classes.index("spider web")], device=device
                            )
                        elif label == "quail":
                            desired_label_id = torch.tensor(
                                [imagenet_classes.index("partridge")], device=device
                            )
                        elif label == "hummingbird":
                            desired_label_id = torch.tensor(
                                [imagenet_classes.index("bee eater")], device=device
                            )
                        elif label == "koala":
                            desired_label_id = torch.tensor(
                                [imagenet_classes.index("wombat")], device=device
                            )
                        ce_loss = F.cross_entropy(output, desired_label_id)
                        loss = ce_loss
                    if args.tag == "task3":
                        # cross entropy loss
                        ce_loss = F.cross_entropy(output, label_id)

                        # Calculate softmax probabilities
                        softmax_probs = F.softmax(output, dim=1)

                        # Create a target uniform distribution
                        uniform_dist = torch.full_like(softmax_probs, 1.0 / 100)

                        # Compute the KL divergence
                        kl_div = F.kl_div(
                            softmax_probs.log(), uniform_dist, reduction="batchmean"
                        )

                        # Hyperparameter to balance the two terms
                        alpha = 1  # This would need to be tuned to achieve the desired effect
                        embed_loss = 1 - cos(variable, original_embed)
                        # Combined loss
                        loss = alpha * kl_div + 0.1 * embed_loss

                    if args.tag == "task4":
                        ce_loss = F.cross_entropy(output, label_id)
                        # Calculate the softmax of the logits
                        softmax_probs = F.softmax(output, dim=1)

                        # We take the maximum softmax probability across classes
                        max_prob, _ = torch.max(softmax_probs, dim=1)

                        # Negate the max probability to maximize it via gradient ascent
                        loss = -torch.mean(max_prob)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # recording
                    all_metrics["loss"].append(loss.item())

                    all_metrics["ce_loss"].append(ce_loss.item())
                    # all_metrics['embed_loss'].append(embed_loss.item())
                    if args.tag == "task3":
                        all_metrics["kl_div"].append(kl_div.item())
                        all_metrics["embed_loss"].append(embed_loss.item())

                    probs = F.softmax(output, dim=1)
                    max_prob, max_class = probs.max(1)
                    all_metrics["prob_target_class"].append(probs[0, label_id].item())
                    all_metrics["prob_max"].append(max_prob.item())
                    all_metrics["pred_class"].append(max_class.item())

                    opt_embeds.append(variable.detach().clone().cpu())

                    # saving image
                    os.makedirs(
                        os.path.join(img_save_dir, f"sample_{j:03d}"), exist_ok=True
                    )
                    to_pil_image(image[0].cpu()).save(
                        os.path.join(
                            img_save_dir, f"sample_{j:03d}", f"step_{step:02d}.png"
                        )
                    )

                    if args.tag == "task3":
                        tqdm.write(
                            f"Step [{step:02d}]: loss - {loss.item():.4f}, ce - {ce_loss.item():.4f}, kl_div - {kl_div.item():.4f}, embed_loss - {embed_loss.item():.4f}, "
                            f"prob_target - {probs[0, label_id].item():.4f}, prob_max - {max_prob.item():.4f}, pred_class - {max_class.item()}"
                        )
                    else:
                        tqdm.write(
                            f"Step [{step:02d}]: loss - {loss.item():.4f}, ce - {ce_loss.item():.4f}, "
                            f"prob_target - {probs[0, label_id].item():.4f}, prob_max - {max_prob.item():.4f}, pred_class - {max_class.item()},  pred_class_name - {all_class_labels[max_class.item()]}"
                        )

            # final evaluation
            step += 1
            text_embeddings = pipe.text_encoder(text_input_ids)[0]
            image = forward_diffusion(
                pipe,
                latents,
                torch.cat([uncond_embeddings, text_embeddings]),
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                eta=0.0,
            )
            output = forward_classifier(
                image, preprocessor, clf, device, dataset=args.dataset
            )

            ce_loss = F.cross_entropy(output, label_id)
            # loss = ce_loss + args.alpha * embed_loss
            if args.tag == "task1":
                # task1: negative cross entropy loss
                ce_loss = F.cross_entropy(output, label_id)
                loss = -ce_loss
            if args.tag == "task2" and args.dataset == "cifar100":
                if label == "apples":
                    desired_label_id = torch.tensor(
                        [cifar100_classes.index("oranges")], device=device
                    )
                elif label == "aquarium fish":
                    desired_label_id = torch.tensor(
                        [cifar100_classes.index("flatfish")], device=device
                    )
                elif label == "baby":
                    desired_label_id = torch.tensor(
                        [cifar100_classes.index("boy")], device=device
                    )
                elif label == "bear":
                    desired_label_id = torch.tensor(
                        [cifar100_classes.index("wolf")], device=device
                    )
                elif label == "beaver":
                    desired_label_id = torch.tensor(
                        [cifar100_classes.index("otter")], device=device
                    )
                elif label == "bed":
                    desired_label_id = torch.tensor(
                        [cifar100_classes.index("couch")], device=device
                    )  # or 'chair'
                elif label == "bee":
                    desired_label_id = torch.tensor(
                        [cifar100_classes.index("butterfly")], device=device
                    )
                elif label == "beetle":
                    desired_label_id = torch.tensor(
                        [cifar100_classes.index("cockroach")], device=device
                    )  # or 'caterpillar'
                elif label == "bicycle":
                    desired_label_id = torch.tensor(
                        [cifar100_classes.index("motorcycle")], device=device
                    )
                elif label == "bottles":
                    desired_label_id = torch.tensor(
                        [cifar100_classes.index("cans")], device=device
                    )
                ce_loss = F.cross_entropy(output, desired_label_id)
                loss = ce_loss
            if args.tag == "task2" and args.dataset == "imagenet":
                if label == "stingray":
                    desired_label_id = torch.tensor(
                        [imagenet_classes.index("electric ray")], device=device
                    )
                elif label == "hen":
                    desired_label_id = torch.tensor(
                        [imagenet_classes.index("rooster")], device=device
                    )
                elif label == "magpie":
                    desired_label_id = torch.tensor(
                        [imagenet_classes.index("jay")], device=device
                    )
                elif label == "kite":
                    desired_label_id = torch.tensor(
                        [imagenet_classes.index("bald eagle")], device=device
                    )
                elif label == "vulture":
                    desired_label_id = torch.tensor(
                        [imagenet_classes.index("bald eagle")], device=device
                    )
                elif label == "agama":
                    desired_label_id = torch.tensor(
                        [imagenet_classes.index("alligator lizard")], device=device
                    )
                elif label == "tick":
                    desired_label_id = torch.tensor(
                        [imagenet_classes.index("spider web")], device=device
                    )
                elif label == "quail":
                    desired_label_id = torch.tensor(
                        [imagenet_classes.index("partridge")], device=device
                    )
                elif label == "hummingbird":
                    desired_label_id = torch.tensor(
                        [imagenet_classes.index("bee eater")], device=device
                    )
                elif label == "koala":
                    desired_label_id = torch.tensor(
                        [imagenet_classes.index("wombat")], device=device
                    )
                ce_loss = F.cross_entropy(output, desired_label_id)
                loss = ce_loss
            if args.tag == "task3":
                # cross entropy loss
                ce_loss = F.cross_entropy(output, label_id)

                # Calculate softmax probabilities
                softmax_probs = F.softmax(output, dim=1)

                # Create a target uniform distribution
                uniform_dist = torch.full_like(softmax_probs, 1.0 / 100)

                # Compute the KL divergence
                kl_div = F.kl_div(
                    softmax_probs.log(), uniform_dist, reduction="batchmean"
                )

                alpha = 1  # This would need to be tuned to achieve the desired effect
                embed_loss = cos(variable, original_embed)
                # Combined loss
                loss = alpha * kl_div + 0.1 * embed_loss

            if args.tag == "task4":
                ce_loss = F.cross_entropy(output, label_id)
                # Calculate the softmax of the logits
                softmax_probs = F.softmax(output, dim=1)

                # We take the maximum softmax probability across classes
                max_prob, _ = torch.max(softmax_probs, dim=1)

                # Negate the max probability to maximize it via gradient ascent
                loss = -torch.mean(max_prob)

            all_metrics["loss"].append(loss.item())
            all_metrics["ce_loss"].append(ce_loss.item())
            # all_metrics['embed_loss'].append(embed_loss.item())
            if args.tag == "task3":
                all_metrics["kl_div"].append(kl_div.item())
                all_metrics["embed_loss"].append(embed_loss.item())

            probs = F.softmax(output, dim=1)
            max_prob, max_class = probs.max(1)
            all_metrics["prob_target_class"].append(probs[0, label_id].item())
            all_metrics["prob_max"].append(max_prob.item())
            all_metrics["pred_class"].append(max_class.item())
            if args.tag == "task3":
                tqdm.write(
                    f"Step [{step:02d}]: loss - {loss.item():.4f}, ce - {ce_loss.item():.4f}, kl_div - {kl_div.item():.4f}, embed_loss - {embed_loss.item():.4f}, "
                    f"prob_target - {probs[0, label_id].item():.4f}, prob_max - {max_prob.item():.4f}, pred_class - {max_class.item()}, all_probs>0.5 - {probs[0][probs[0, :].gt(0.5)]}"
                )
            else:
                tqdm.write(
                    f"Step [{step:02d}]: loss - {loss.item():.4f}, ce - {ce_loss.item():.4f}, "
                    f"prob_target - {probs[0, label_id].item():.4f}, prob_max - {max_prob.item():.4f}, pred_class - {max_class.item()},  pred_class_name - {all_class_labels[max_class.item()]}"
                )
            # Check if all arrays in all_metrics have the same length
            lengths = [len(v) for v in all_metrics.values()]
            if len(set(lengths)) != 1:
                print("Not all arrays are of the same length.")
                for k, v in all_metrics.items():
                    print(f"{k}: Length is {len(v)}")

            all_metrics = pd.DataFrame(all_metrics)
            all_metrics.to_csv(
                os.path.join(log_save_dir, f"sample_{j:03d}.csv"), index=False
            )
            to_pil_image(image[0].cpu()).save(
                os.path.join(img_save_dir, f"sample_{j:03d}", f"step_{step:02d}.png")
            )

            opt_embeds = torch.cat(opt_embeds)
            torch.save(opt_embeds, os.path.join(embed_save_dir, f"sample_{j:03d}.pt"))

        # restore the original token embedding
        pipe.text_encoder.text_model.embeddings.token_embedding = (
            original_tk_embed_layer
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
