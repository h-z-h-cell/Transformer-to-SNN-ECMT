import math
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import ImageFolder,CIFAR10,CIFAR100
from timm.data import create_transform
def build_dataset(is_train, args,root_path='.'):
    transform = build_transform(is_train, args)
    if args.data_set == 'CIFAR10':
        dataset_val = CIFAR10(root=args.eval_data_path, train=False ,transform=transform, download=True)
        if is_train==True:
            dataset_train = CIFAR10(root=args.eval_data_path, train=True ,transform=transform, download=True)
    elif args.data_set == 'CIFAR100':
        dataset_val = CIFAR100(root=args.eval_data_path, train=False ,transform=transform, download=True)
        if is_train==True:
            dataset_train = CIFAR100(root=args.eval_data_path, train=True ,transform=transform, download=True)
    elif args.data_set == "image_folder":
        dataset_val = ImageFolder(args.eval_data_path, transform=transform)
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % args.nb_classes)
    if is_train==True:
        return dataset_train, dataset_val, args.nb_classes
    else:
        return dataset_val, args.nb_classes

def build_transform(is_train, args):
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = (0.48145466, 0.4578275, 0.40821073) if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = (0.26862954, 0.26130258, 0.27577711) if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    return transforms.Compose([transforms.Resize(args.input_size, interpolation=3),transforms.CenterCrop(args.input_size),transforms.ToTensor(),transforms.Normalize(mean, std)])