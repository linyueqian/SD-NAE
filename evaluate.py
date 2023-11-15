import os

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torchvision import transforms as trn
from tqdm import tqdm
from transformers import AutoImageProcessor  # Assuming this is for ImageNet
from transformers import ResNetForImageClassification

from generation.class_names import (cifar10_classes, cifar100_classes,
                                    imagenet_classes, in100_classes)
from models.wrn import WideResNet  # Assuming this is for CIFAR-100


def load_classifier(dataset, device):
    if dataset == "cifar100":
        classifier = WideResNet(depth=40, num_classes=100, widen_factor=2)
        classifier.load_state_dict(torch.load("ckpt/seed_233.pth", map_location=device))
        preprocessor = trn.Compose(
            [
                trn.Resize((32, 32), antialias=True),
                trn.ToTensor(),
                trn.Normalize(
                    [x / 255 for x in [129.3, 124.1, 112.4]],
                    [x / 255 for x in [68.2, 65.4, 70.4]],
                ),
            ]
        )
    elif dataset == "cifar10":
        classifier = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True
        )
        preprocessor = trn.Compose(
            [
                trn.Resize((32, 32), antialias=True),
                trn.ToTensor(),
                trn.Normalize(
                    [x / 255 for x in [125.3, 123.0, 113.9]],
                    [x / 255 for x in [63.0, 62.1, 66.7]],
                ),
            ]
        )
    elif dataset == "imagenet":
        preprocessor = trn.Compose(
            [
                trn.Resize((224, 224), antialias=True),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        classifier = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        classifier.eval()
    else:
        raise NotImplementedError("Dataset not supported")

    classifier.to(device)
    classifier.eval()
    return classifier, preprocessor


def forward_classifier(x, preprocessor, clf, device):
    x = preprocessor(x).unsqueeze(0).to(device)
    x = clf(x)
    return x


def forward_classifier_in(x, preprocessor, clf, device):
    x = preprocessor(x).unsqueeze(0).to(device)
    x = clf(x).logits
    return x


def evaluate_classifier(classifier, outlier_folder, preprocessor, device):
    predictions = []
    with torch.no_grad():
        for task in ["task1", "task2", "task3"]:
            attack_success = []
            for i, class_folder in tqdm(
                enumerate(os.listdir(os.path.join(outlier_folder, task, "cifar100")))
            ):
                class_folder = os.path.join(
                    outlier_folder, task, "cifar100", class_folder
                )
                for sample_folder in os.listdir(class_folder):
                    # load the biggest step image
                    biggest_step = max(
                        os.listdir(os.path.join(class_folder, sample_folder))
                    )
                    image = Image.open(
                        os.path.join(class_folder, sample_folder, biggest_step)
                    )
                    output = forward_classifier(image, preprocessor, classifier, device)
                    _, predicted = torch.max(output, 1)
                    if predicted.item() != i:
                        attack_success.append(1)
                    else:
                        attack_success.append(0)
            asr = sum(attack_success) / len(attack_success)
            predictions.append(asr)
    return predictions


def evaluate_classifier_in(classifier, outlier_folder, preprocessor, device):
    predictions = []
    with torch.no_grad():
        for task in ["task1", "task2", "task3"]:
            attack_success = []
            for i, class_folder in tqdm(
                enumerate(os.listdir(os.path.join(outlier_folder, task, "cifar100")))
            ):
                class_folder = os.path.join(
                    outlier_folder, task, "cifar100", class_folder
                )
                for sample_folder in os.listdir(class_folder):
                    # load the biggest step image
                    biggest_step = max(
                        os.listdir(os.path.join(class_folder, sample_folder))
                    )
                    image = Image.open(
                        os.path.join(class_folder, sample_folder, biggest_step)
                    )
                    output = forward_classifier_in(
                        image, preprocessor, classifier, device
                    )
                    _, predicted = torch.max(output, 1)
                    if predicted.item() != imagenet_classes.index(in100_classes[i]):
                        # print(predicted.item(), imagenet_classes.index(in100_classes[i]))
                        # print class
                        # print(imagenet_classes[predicted.item()], in100_classes[i])
                        attack_success.append(1)
                    else:
                        attack_success.append(0)
            asr = sum(attack_success) / len(attack_success)
            predictions.append(asr)
    return predictions


def main():
    dataset = "cifar100"  # Change as needed: 'cifar100', 'cifar10', 'imagenet'
    outlier_folder = "outliers"
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    classifier, preprocessor = load_classifier(dataset, device)
    if dataset == "cifar100":
        results = evaluate_classifier(classifier, outlier_folder, preprocessor, device)
    elif dataset == "imagenet":
        results = evaluate_classifier_in(
            classifier, outlier_folder, preprocessor, device
        )
    print(results)


if __name__ == "__main__":
    main()
