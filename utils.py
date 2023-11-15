import os

import numpy as np
import torch
from torchvision.datasets.folder import default_loader, make_dataset
from torchvision.datasets.vision import VisionDataset

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

IN100_wnids = [
    "n01498041",
    "n01514859",
    "n01582220",
    "n01608432",
    "n01616318",
    "n01687978",
    "n01776313",
    "n01806567",
    "n01833805",
    "n01882714",
    "n01910747",
    "n01944390",
    "n01985128",
    "n02007558",
    "n02071294",
    "n02085620",
    "n02114855",
    "n02123045",
    "n02128385",
    "n02129165",
    "n02129604",
    "n02165456",
    "n02190166",
    "n02219486",
    "n02226429",
    "n02279972",
    "n02317335",
    "n02326432",
    "n02342885",
    "n02363005",
    "n02391049",
    "n02395406",
    "n02403003",
    "n02422699",
    "n02442845",
    "n02444819",
    "n02480855",
    "n02510455",
    "n02640242",
    "n02672831",
    "n02687172",
    "n02701002",
    "n02730930",
    "n02769748",
    "n02782093",
    "n02787622",
    "n02793495",
    "n02799071",
    "n02802426",
    "n02814860",
    "n02840245",
    "n02906734",
    "n02948072",
    "n02980441",
    "n02999410",
    "n03014705",
    "n03028079",
    "n03032252",
    "n03125729",
    "n03160309",
    "n03179701",
    "n03220513",
    "n03249569",
    "n03291819",
    "n03384352",
    "n03388043",
    "n03450230",
    "n03481172",
    "n03594734",
    "n03594945",
    "n03627232",
    "n03642806",
    "n03649909",
    "n03661043",
    "n03676483",
    "n03724870",
    "n03733281",
    "n03759954",
    "n03761084",
    "n03773504",
    "n03804744",
    "n03916031",
    "n03938244",
    "n04004767",
    "n04026417",
    "n04090263",
    "n04133789",
    "n04153751",
    "n04296562",
    "n04330267",
    "n04371774",
    "n04404412",
    "n04465501",
    "n04485082",
    "n04507155",
    "n04536866",
    "n04579432",
    "n04606251",
    "n07714990",
    "n07745940",
]


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/[...]/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/[...]/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root,
        loader,
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
        class_list=None,
        num_classes=None,
    ):
        super(DatasetFolder, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.class_list = class_list
        self.num_classes = num_classes
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory,
        class_to_idx,
        extensions=None,
        is_valid_file=None,
    ):
        return make_dataset(
            directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file
        )

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if self.class_list is None:
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [
                d.name
                for d in os.scandir(dir)
                if (d.is_dir() and (d.name in self.class_list))
            ]

        if self.num_classes:
            assert self.num_classes <= len(classes)
            classes = list(np.random.choice(classes, self.num_classes, replace=False))

        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


class ImageFolderSelective(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
        class_list=None,
        num_classes=None,
    ):
        super(ImageFolderSelective, self).__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            class_list=class_list,
            num_classes=num_classes,
        )
        self.imgs = self.samples


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
