from .logger import CompleteLogger
from .meter import *
from .data import ForeverDataIterator

__all__ = ['metric', 'analysis', 'meter', 'data', 'logger']

import sys
import time
import timm
import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataset import Subset, ConcatDataset
import wilds
from torch.utils.data import DataLoader
import vision.datasets as datasets
import vision.models as models
from vision.transforms import ResizeImage
from utils.metric import accuracy
from utils.meter import AverageMeter, ProgressMeter


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


def get_model(model_name):
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=True)
    else:
        # else:
            # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=True)
        if 'resnetv2' in model_name:
            backbone.head.in_features = backbone.num_features
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone


def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + wilds.supported_datasets


class ConcatDatasetWithDomainLabel(ConcatDataset):
    """ConcatDataset with domain label"""

    def __init__(self, *args, **kwargs):
        super(ConcatDatasetWithDomainLabel, self).__init__(*args, **kwargs)
        self.index_to_domain_id = {}
        domain_id = 0
        start = 0
        for end in self.cumulative_sizes:
            for idx in range(start, end):
                self.index_to_domain_id[idx] = domain_id
            start = end
            domain_id += 1

    def __getitem__(self, index):
        img, target = super(ConcatDatasetWithDomainLabel, self).__getitem__(index)
        domain_id = self.index_to_domain_id[index]
        return img, target, domain_id


def convert_from_wilds_dataset(dataset_name, wild_dataset):
    metadata_array = wild_dataset.metadata_array
    sample_idxes_per_domain = {}
    for idx, metadata in enumerate(metadata_array):
        if dataset_name == 'iwildcam':
            # In iwildcam dataset, domain id is specified by location
            domain = metadata[0].item()
        elif dataset_name == 'camelyon17':
            # In camelyon17 dataset, domain id is specified by hospital
            domain = metadata[0].item()
        elif dataset_name == 'fmow':
            # In fmow dataset, domain id is specified by (region, year) tuple
            domain = (metadata[0].item(), metadata[1].item())

        if domain not in sample_idxes_per_domain:
            sample_idxes_per_domain[domain] = []
        sample_idxes_per_domain[domain].append(idx)

    class Dataset:
        def __init__(self):
            self.dataset = wild_dataset

        def __getitem__(self, idx):
            x, y, metadata = self.dataset[idx]
            return x, y

        def __len__(self):
            return len(self.dataset)

    dataset = Dataset()
    concat_dataset = ConcatDatasetWithDomainLabel(
        [Subset(dataset, sample_idxes_per_domain[domain]) for domain in sample_idxes_per_domain])
    return concat_dataset


def get_dataset(dataset_name, root, task_list, split='train', download=True, transform=None, seed=0, split_ratio=0.8):
    assert split in ['train', 'val', 'test']
    if dataset_name in datasets.__dict__:
        # load datasets from common.vision.datasets
        # currently only PACS, OfficeHome and DomainNet are supported
        supported_dataset = ['PACS', 'OfficeHome', 'DomainNet', 'CUB']
        assert dataset_name in supported_dataset

        dataset = datasets.__dict__[dataset_name]

        train_split_list = []
        val_split_list = []
        test_split_list = []
        # we follow DomainBed and split each dataset randomly into two parts, with 80% samples and 20% samples
        # respectively, the former (larger) will be used as training set, and the latter will be used as validation set.
        split_ratio = split_ratio
        num_classes = 0

        # under domain generalization setting, we use all samples in target domain as test set
        for task in task_list:
            if dataset_name == 'PACS':
                all_split = dataset(root=root, task=task, split='all', download=download, transform=transform)
                num_classes = all_split.num_classes
            elif dataset_name == 'OfficeHome':
                all_split = dataset(root=root, task=task, download=download, transform=transform)
                num_classes = all_split.num_classes
            elif dataset_name == 'CUB':
                all_split = dataset(root=root, task=task, download=download, transform=transform)
                num_classes = all_split.num_classes
            elif dataset_name == 'DomainNet':
                train_split = dataset(root=root, task=task, split='train', download=download, transform=transform)
                test_split = dataset(root=root, task=task, split='test', download=download, transform=transform)
                num_classes = train_split.num_classes
                all_split = ConcatDataset([train_split, test_split])

            train_split, val_split = split_dataset(all_split, int(len(all_split) * split_ratio), seed)

            train_split_list.append(train_split)
            val_split_list.append(val_split)
            test_split_list.append(all_split)

        train_dataset = ConcatDatasetWithDomainLabel(train_split_list)
        val_dataset = ConcatDatasetWithDomainLabel(val_split_list)
        test_dataset = ConcatDatasetWithDomainLabel(test_split_list)

        dataset_dict = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        return dataset_dict[split], num_classes
    else:
        # load datasets from wilds
        # currently only iwildcam, camelyon17 and fmow are supported
        supported_dataset = ['iwildcam', 'camelyon17', 'fmow']
        assert dataset_name in supported_dataset

        dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True)
        num_classes = dataset.n_classes
        return convert_from_wilds_dataset(dataset_name,
                                          dataset.get_subset(split=split, transform=transform)), num_classes



def get_dataset_class_list(dataset_name, root, task_list, split='train', download=True, transform=None, seed=0, split_ratio=0.8):
    assert split in ['train', 'val', 'test']
    if dataset_name in datasets.__dict__:
        # load datasets from common.vision.datasets
        # currently only PACS, OfficeHome and DomainNet are supported
        supported_dataset = ['PACS', 'OfficeHome', 'DomainNet', 'CUB']
        assert dataset_name in supported_dataset

        dataset = datasets.__dict__[dataset_name]

        train_split_list = []
        val_split_list = []
        test_split_list = []
        # we follow DomainBed and split each dataset randomly into two parts, with 80% samples and 20% samples
        # respectively, the former (larger) will be used as training set, and the latter will be used as validation set.
        split_ratio = split_ratio
        num_classes = 0

        # under domain generalization setting, we use all samples in target domain as test set
        for task in task_list:
            if dataset_name == 'PACS':
                all_split = dataset(root=root, task=task, split='all', download=download, transform=transform)
                num_classes = all_split.num_classes
                class_list = all_split.CLASSES
            elif dataset_name == 'OfficeHome':
                all_split = dataset(root=root, task=task, download=download, transform=transform)
                num_classes = all_split.num_classes
                class_list = all_split.CLASSES
            elif dataset_name == 'CUB':
                all_split = dataset(root=root, task=task, download=download, transform=transform)
                num_classes = all_split.num_classes
                class_list = all_split.CLASSES
            elif dataset_name == 'DomainNet':
                train_split = dataset(root=root, task=task, split='train', download=download, transform=transform)
                test_split = dataset(root=root, task=task, split='test', download=download, transform=transform)
                num_classes = train_split.num_classes
                class_list = train_split.CLASSES
                all_split = ConcatDataset([train_split, test_split])


            train_split, val_split = split_dataset(all_split, int(len(all_split) * split_ratio), seed)

            train_split_list.append(train_split)
            val_split_list.append(val_split)
            test_split_list.append(all_split)

        train_dataset = ConcatDatasetWithDomainLabel(train_split_list)
        val_dataset = ConcatDatasetWithDomainLabel(val_split_list)
        test_dataset = ConcatDatasetWithDomainLabel(test_split_list)

        dataset_dict = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        return dataset_dict[split], num_classes, class_list
    else:
        # load datasets from wilds
        # currently only iwildcam, camelyon17 and fmow are supported
        supported_dataset = ['iwildcam', 'camelyon17', 'fmow']
        assert dataset_name in supported_dataset

        dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True)
        num_classes = dataset.n_classes
        # class_list = dataset.CLASSES
        class_list = None
        return convert_from_wilds_dataset(dataset_name,
                                          dataset.get_subset(split=split, transform=transform)), num_classes, class_list



def get_dataset_class(dataset_name, root, task_list, split='train', download=True, transform=None, seed=0, split_ratio=0.8):
    assert split in ['train', 'val', 'test']
    if dataset_name in datasets.__dict__:
        # load datasets from common.vision.datasets
        # currently only PACS, OfficeHome and DomainNet are supported
        supported_dataset = ['PACS', 'OfficeHome', 'DomainNet', 'CUB']
        assert dataset_name in supported_dataset

        dataset = datasets.__dict__[dataset_name]

        train_split_list = []
        val_split_list = []
        test_split_list = []
        # we follow DomainBed and split each dataset randomly into two parts, with 80% samples and 20% samples
        # respectively, the former (larger) will be used as training set, and the latter will be used as validation set.
        split_ratio = split_ratio
        num_classes = 0

        # under domain generalization setting, we use all samples in target domain as test set
        for task in task_list:
            if dataset_name == 'PACS':
                all_split = dataset(root=root, task=task, split='all', download=download, transform=transform)
                num_classes = all_split.num_classes
            elif dataset_name == 'OfficeHome':
                all_split = dataset(root=root, task=task, download=download, transform=transform)
                num_classes = all_split.num_classes
            elif dataset_name == 'CUB':
                all_split = dataset(root=root, task=task, download=download, transform=transform)
                num_classes = all_split.num_classes
            elif dataset_name == 'DomainNet':
                train_split = dataset(root=root, task=task, split='train', download=download, transform=transform)
                test_split = dataset(root=root, task=task, split='test', download=download, transform=transform)
                num_classes = train_split.num_classes
                all_split = ConcatDataset([train_split, test_split])

            train_split, val_split = split_dataset_class(all_split, int(len(all_split) * split_ratio), seed)

            train_split_list.append(train_split)
            val_split_list.append(val_split)
            test_split_list.append(all_split)

        train_dataset = ConcatDatasetWithDomainLabel(train_split_list)
        val_dataset = ConcatDatasetWithDomainLabel(val_split_list)
        test_dataset = ConcatDatasetWithDomainLabel(test_split_list)

        dataset_dict = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        return dataset_dict[split], num_classes
    else:
        # load datasets from wilds
        # currently only iwildcam, camelyon17 and fmow are supported
        supported_dataset = ['iwildcam', 'camelyon17', 'fmow']
        assert dataset_name in supported_dataset

        dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True)
        num_classes = dataset.n_classes
        return convert_from_wilds_dataset(dataset_name,
                                          dataset.get_subset(split=split, transform=transform)), num_classes



def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n data points in the first dataset and the rest in the last,
    using the given random seed
    """
    assert (n <= len(dataset))
    idxes = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(idxes)
    subset_1 = idxes[:n]
    subset_2 = idxes[n:]
    return Subset(dataset, subset_1), Subset(dataset, subset_2)


def split_dataset_class(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n data points in the first dataset and the rest in the last,
    using the given random seed
    """
    assert (n <= len(dataset))
    idxes = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(idxes)
    subset_1 = idxes[:n]
    subset_2 = idxes[n:]
    return Subset(dataset, subset_1), Subset(dataset, subset_2)



def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, domain_label) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg



def validate_domains(val_loader, model, args, device, logger) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    num_domains = len(args.targets)

    top1_domains = [AverageMeter('Acc@1', ':6.2f') for i in range(num_domains)]
    acc_list = []
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    with torch.no_grad():
        end = time.time()
        for i, (images, target, domain_label) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), target.size(0))
            for d in range(num_domains):
                select = (domain_label == d)
                if select.sum() > 0:
                    acc1_domains = accuracy(output[select], target[select])[0]
                    top1_domains[d].update(acc1_domains.item(), target[select].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))

        for d in range(num_domains):
            logger.write("Domain: {domain}, Acc: {top1.avg:.3f}".format(domain=args.targets[d], top1=top1_domains[d]))
            acc_list.append(top1_domains[d].avg)

    return acc_list, top1.avg






def validate_each_domain(val_loader, model, args, device, logger) -> float:


    # switch to evaluate mode
    model.eval()

    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    datasets = val_loader.dataset.datasets
    acc_list = []
    top1_all = AverageMeter('Acc@1', ':6.2f')
    with torch.no_grad():
        end = time.time()

        for dataset in datasets:

            batch_time = AverageMeter('Time', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')


            domain_val_loader = DataLoader(dataset, batch_size=72, shuffle=False, num_workers=4)
            progress = ProgressMeter(
                len(domain_val_loader),
                [batch_time, losses, top1],
                prefix='Test: ')

            for i, (images, target) in enumerate(domain_val_loader):
                images = images.to(device)
                target = target.to(device)

                # compute output
                output = model(images)
                loss = F.cross_entropy(output, target)

                # measure accuracy and record loss
                acc1 = accuracy(output, target)[0]
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top1_all.update(acc1.item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

            print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))
            logger.write("Domain: {domain}, Acc: {top1.avg:.3f}".format(domain=dataset.data_list_file, top1=top1))
            acc_list.append(top1.avg)

    return acc_list, top1_all.avg



def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=True,
                        random_gray_scale=True):
    """
    resizing mode:
        - default: random resized crop with scale factor(0.7, 1.0) and size 224;
        - cen.crop: take the center crop of 224;
        - res.|cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
        - res2x: resize the image to 448;
        - res.|crop: resize the image to 256 and take a random crop of size 224;
        - res.sma|crop: resize the image keeping its aspect ratio such that the
            smaller side is 256, then take a random crop of size 224;
        – inc.crop: “inception crop” from (Szegedy et al., 2015);
        – cif.crop: resize the image to 224, zero-pad it by 28 on each side, then take a random crop of size 224.
    """
    if resizing == 'default':
        transform = T.RandomResizedCrop(224, scale=(0.7, 1.0))
    elif resizing == 'default_256':
        transform = T.RandomResizedCrop(256, scale=(0.7, 1.0))
    elif resizing == 'cen.crop':
        transform = T.CenterCrop(224)
    elif resizing == 'res.|cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'res':
        transform = ResizeImage(224)
    elif 'res2x' in resizing:
        transform = ResizeImage(448)
    elif resizing == 'res.|crop':
        transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224)
        ])
    elif resizing == "res.sma|crop":
        transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'inc.crop':
        transform = T.RandomResizedCrop(224)
    elif resizing == 'cif.crop':
        transform = T.Compose([
            T.Resize((224, 224)),
            T.Pad(28),
            T.RandomCrop(224),
        ])
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3))
    if random_gray_scale:
        transforms.append(T.RandomGrayscale())
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default'):
    """
    resizing mode:
        - default: resize the image to 224;
        - res2x: resize the image to 448;
        - res.|cen.crop: resize the image to 256 and take the center crop of size 224;
    """
    if resizing == 'default':
        transform = ResizeImage(224)
    elif resizing == 'default_256':
        transform = ResizeImage(256)
    elif 'res2x' in resizing:
        transform = ResizeImage(448)
    elif resizing == 'res.|cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def collect_feature(data_loader, feature_extractor: nn.Module, device: torch.device,
                    max_num_features=None, return_label=False) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features. This function is
    specific for domain generalization because each element in data_loader is a tuple
    (images, labels, domain_labels).

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    labels = []
    with torch.no_grad():
        for i, (images, target, domain_labels) in enumerate(tqdm.tqdm(data_loader)):
            if max_num_features is not None and i >= max_num_features:
                break
            images = images.to(device)
            feature = feature_extractor(images).cpu()
            all_features.append(feature)
            labels.append(target)

    if return_label == True:
        return torch.cat(all_features, dim=0), torch.cat(labels, dim=0)
    else:
        return torch.cat(all_features, dim=0)


import math
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

