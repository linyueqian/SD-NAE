import argparse
from copy import deepcopy
import json
import os
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as trn
from torchvision.transforms.functional import to_pil_image

from diffusers import DDIMScheduler, StableDiffusionPipeline
from transformers import ResNetForImageClassification

from class_names import in_selected_classes
with open('in100_class_index.json', 'r') as f:
    in100_class_index = json.load(f)


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
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--guidance_scale", type=float, default=7.5)
parser.add_argument("--num_inference_steps", type=int, default=20)
parser.add_argument("--num_samples_per_class", type=int, default=20)
parser.add_argument("--class_ids", type=int, nargs="+", default=[0, 9])
parser.add_argument("--num_grad_steps", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--alpha", type=float, default=0.0)
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


def forward_classifier(x, preprocessor, clf):
    inputs = preprocessor(x)
    return clf(inputs).logits
    

@torch.no_grad()
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # preparation
    assert (
        len(args.class_ids) == 2 and args.class_ids[1] >= args.class_ids[0]
    ), "Please specify the start and end class id (end >= start)."
    start, end = args.class_ids
    
    img_save_root = f"./outliers/alpha{args.alpha}_gradsteps{args.num_grad_steps}"
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
    
    # load classifier
    preprocessor = trn.Compose(
        [
            trn.Resize((224, 224), antialias=True),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    clf = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    clf.to(device)
    clf.eval()

    cos = nn.CosineSimilarity(dim=-1)

    # class loop
    for i in tqdm(
        range(start, end + 1), total=end - start + 1, desc="Classes", leave=True
    ):
        # prepare prompt
        label = in_selected_classes[i]
        class_id = torch.tensor([in100_class_index[label][0]], device=device)

        # prompt = f"A high-quality image of a {label}"
        prompt = f"{label}"

        # prepare for text embeddings
        text_input = pipe.tokenizer(
            [prompt],
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_input.input_ids

        # get the token id for the token that we will optimize
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
        img_save_dir = os.path.join(img_save_root, f"{label.replace(' ', '_')}")
        os.makedirs(img_save_dir, exist_ok=True)

        log_save_dir = os.path.join(log_save_root, f"{label.replace(' ', '_')}")
        os.makedirs(log_save_dir, exist_ok=True)

        # sample loop
        for j in tqdm(
            range(args.num_samples_per_class),
            total=args.num_samples_per_class,
            desc="Samples",
            leave=False,
        ):
            # if the image is already generated, skip
            if os.path.isfile(os.path.join(log_save_dir, f'sample_{j:02d}.csv')):
                continue

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

            embedding_weights = original_tk_embed_layer.weight.data.clone()
            pipe.text_encoder.text_model.embeddings.token_embedding = CustomEmbedding(
                embedding_weights, token_id
            )

            # optimization
            all_metrics = {
                "loss": [],
                "adv_loss": [],
                "embed_loss": [],
                "pred": [],
                "prob": []
            }

            with torch.enable_grad():
                variable = (
                    pipe.text_encoder.text_model.embeddings.token_embedding.weights[
                        token_id
                    ].weight
                )
                original_embed = variable.detach().clone()
                optimizer = torch.optim.Adam([variable], lr=args.lr)

                tqdm.write("")
                tqdm.write(f"Class: {i:02d} - {label}, Sample {j:02d}")
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
                        image, preprocessor, clf
                    )

                    # loss
                    adv_loss = -F.cross_entropy(output, class_id)
                    embed_loss = cos(variable, original_embed)
                    loss = adv_loss + args.alpha * embed_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # recording
                    all_metrics["loss"].append(loss.item())
                    all_metrics["adv_loss"].append(adv_loss.item())
                    all_metrics['embed_loss'].append(embed_loss.item())
                    max_prob, max_class = F.softmax(output, dim=1).max(1)
                    all_metrics["prob"].append(max_prob.item())
                    all_metrics["pred"].append(max_class.item())

                    # saving image
                    os.makedirs(
                        os.path.join(img_save_dir, f"sample_{j:02d}"), exist_ok=True
                    )
                    to_pil_image(image[0].cpu()).save(
                        os.path.join(
                            img_save_dir, f"sample_{j:02d}", f"step_{step:02d}.png"
                        )
                    )

                    tqdm.write(
                        f"Step [{step:02d}]: loss - {loss.item():.4f}, adv - {adv_loss.item():.4f}, embed - {embed_loss.item():.4f}, "
                        f"prob_max - {max_prob.item():.4f}, pred_class - {max_class.item()}"
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
                image, preprocessor, clf
            )

            # loss
            adv_loss = -F.cross_entropy(output, class_id)
            embed_loss = cos(variable, original_embed)
            loss = adv_loss + args.alpha * embed_loss

            all_metrics["loss"].append(loss.item())
            all_metrics["adv_loss"].append(adv_loss.item())
            all_metrics['embed_loss'].append(embed_loss.item())
            max_prob, max_class = F.softmax(output, dim=1).max(1)
            all_metrics["prob"].append(max_prob.item())
            all_metrics["pred"].append(max_class.item())
            
            tqdm.write(
                f"Step [{step:02d}]: loss - {loss.item():.4f}, adv - {adv_loss.item():.4f}, embed - {embed_loss.item():.4f}, "
                f"prob_max - {max_prob.item():.4f}, pred_class - {max_class.item()}"
            )

            # Check if all arrays in all_metrics have the same length
            lengths = [len(v) for v in all_metrics.values()]
            if len(set(lengths)) != 1:
                print("Not all arrays are of the same length.")
                for k, v in all_metrics.items():
                    print(f"{k}: Length is {len(v)}")

            all_metrics = pd.DataFrame(all_metrics)
            all_metrics.to_csv(
                os.path.join(log_save_dir, f"sample_{j:02d}.csv"), index=False
            )
            to_pil_image(image[0].cpu()).save(
                os.path.join(img_save_dir, f"sample_{j:02d}", f"step_{step:02d}.png")
            )

        # restore the original token embedding
        pipe.text_encoder.text_model.embeddings.token_embedding = (
            original_tk_embed_layer
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
