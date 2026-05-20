"""
MedMNIST data loader for GHN-3 experiments.

This file mirrors the interface of ppuda's image_loader so train_ddp.py can
branch on the dataset name and call either function. It returns the same
3-tuple: (train_loader, valid_loader, n_classes).

Quick example:

    from medmnist_loader import medmnist_loader

    train_loader, val_loader, n_classes = medmnist_loader(
        dataset='dermamnist',
        data_dir='./data/medmnist',
        batch_size=128,
        load_train_anyway=True,
        im_size=224,
    )

Requirements: pip install medmnist
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ppuda.vision.transforms import transforms_imagenet

import medmnist
from medmnist import INFO


# Maps a lowercase dataset name -> the MedMNIST class that loads it.
# Add more entries here when you want to experiment with additional datasets.
DATASET_REGISTRY = {
    'bloodmnist': medmnist.BloodMNIST,
    'pathmnist': medmnist.PathMNIST,
    'dermamnist': medmnist.DermaMNIST,
    'octmnist': medmnist.OCTMNIST,
    'pneumoniamnist': medmnist.PneumoniaMNIST,
    'retinamnist': medmnist.RetinaMNIST,
    'breastmnist': medmnist.BreastMNIST,
    'tissuemnist': medmnist.TissueMNIST,
    'organamnist': medmnist.OrganAMNIST,
    'organcmnist': medmnist.OrganCMNIST,
    'organsmnist': medmnist.OrganSMNIST,
}


def _flatten_labels(dataset):
    """
    MedMNIST stores labels as shape [N, 1] for single-label classification.
    The training pipeline expects shape [N], so flatten in place.

    Without this fix, every batch would yield labels of shape [batch_size, 1]
    which breaks cross-entropy loss.
    """
    if dataset.labels.ndim == 2 and dataset.labels.shape[1] == 1:
        dataset.labels = dataset.labels.flatten()
    return dataset


def _attach_metadata(dataset):
    """
    image_loader sets `.checksum` and `.num_examples` on each dataset object
    so the training loop can log dataset info cleanly. We add the same
    attributes here. We also expose `.targets` as an alias for `.labels`
    in case any downstream code expects torchvision-style attribute names.
    """
    dataset.targets = dataset.labels
    dataset.num_examples = len(dataset.labels)
    dataset.checksum = float(dataset.imgs.mean())
    return dataset


def medmnist_loader(dataset='dermamnist',
                    data_dir='./data/medmnist',
                    test=True,
                    im_size=224,
                    batch_size=64,
                    test_batch_size=64,
                    num_workers=0,
                    seed=1111,
                    load_train_anyway=False,
                    ddp=False,
                    transforms_train_val=None,
                    verbose=True,
                    **kwargs):
    """
    Build train and eval DataLoaders for a MedMNIST dataset.

    The signature mirrors ppuda.vision.loader.image_loader so it can be
    used as a drop-in replacement.

    :param dataset:            lowercase MedMNIST name, e.g. 'dermamnist'
    :param data_dir:           directory for .npz files (auto-downloaded if missing)
    :param test:               True -> use the test split for eval, False -> use val
    :param im_size:            image resolution (224 recommended for ResNet-50)
    :param batch_size:         training batch size
    :param test_batch_size:    evaluation batch size
    :param num_workers:        DataLoader worker processes
    :param seed:               seed for the eval loader's random generator
    :param load_train_anyway:  build the train split even when test=True
    :param ddp:                True -> use DistributedSampler for multi-GPU training
    :param transforms_train_val: optional (train_transform, valid_transform) tuple.
                                 Defaults to ppuda's transforms_imagenet so GHN-3's
                                 predicted weights see ImageNet-normalised inputs.
    :param verbose:            print a summary of what was loaded
    :param kwargs:             silently absorbs extra args (cutout, noise, n_shots,
                               etc.) that image_loader callers may pass but which
                               don't apply to MedMNIST.
    :return: (train_loader, valid_loader, n_classes)
    """

    name = dataset.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(
            "Unknown MedMNIST dataset '{}'. Supported: {}".format(
                dataset, sorted(DATASET_REGISTRY.keys())
            )
        )

    DatasetClass = DATASET_REGISTRY[name]
    info = INFO[name]
    n_classes = len(info['label'])

    # Default to ppuda's ImageNet transforms. GHN-3's predicted weights expect
    # ImageNet-normalised inputs, so reusing this is correct for fair init
    # comparisons. The transforms also handle resize -> crop -> normalize.
    if transforms_train_val is None:
        transforms_train_val = transforms_imagenet(im_size=im_size)
    train_transform, valid_transform = transforms_train_val

    # MedMNIST splits are 'train', 'val', 'test'. We map test=True/False
    # onto the test/val splits respectively.
    eval_split = 'test' if test else 'val'

    # Always build the eval set.
    valid_data = DatasetClass(
        split=eval_split,
        transform=valid_transform,
        download=True,
        root=data_dir,
        size=im_size,
    )
    valid_data = _flatten_labels(valid_data)
    valid_data = _attach_metadata(valid_data)

    # Build the train set only if we need it.
    train_data = None
    if not test or load_train_anyway:
        train_data = DatasetClass(
            split='train',
            transform=train_transform,
            download=True,
            root=data_dir,
            size=im_size,
        )
        train_data = _flatten_labels(train_data)
        train_data = _attach_metadata(train_data)

    if verbose:
        print(
            "loaded {}: {} classes, {} train samples (checksum={}), "
            "{} {} samples (checksum={:.3f})".format(
                name.upper(),
                n_classes,
                train_data.num_examples if train_data is not None else 'none',
                ('%.3f' % train_data.checksum) if train_data is not None else 'none',
                valid_data.num_examples,
                eval_split,
                valid_data.checksum,
            )
        )

    # Wrap datasets in DataLoaders.
    if train_data is None:
        train_loader = None
    else:
        sampler = DistributedSampler(train_data) if ddp else None
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            pin_memory=True,
            num_workers=num_workers,
        )

    # Seeded generator so eval shuffling (if any) is reproducible.
    generator = torch.Generator()
    generator.manual_seed(seed)

    valid_loader = DataLoader(
        valid_data,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        generator=generator,
    )

    return train_loader, valid_loader, n_classes


#Smoke test to check if medmnist_loader is working properly. This will download DermaMNIST.
if __name__ == '__main__':
    print("Smoke testing medmnist_loader on DermaMNIST (size=224)...")

    train_loader, val_loader, n_classes = medmnist_loader(
        dataset='dermamnist',
        data_dir='./data',
        batch_size=8,
        test_batch_size=8,
        load_train_anyway=True,
        im_size=224,
        num_workers=0,
        verbose=True,
    )

    print("\nn_classes:", n_classes)
    print("train batches:", len(train_loader))
    print("val batches:  ", len(val_loader))

    images, labels = next(iter(train_loader))
    print("\nFirst train batch:")
    print("  images.shape:", images.shape, " dtype:", images.dtype)
    print("  labels.shape:", labels.shape, " dtype:", labels.dtype)
    print("  unique labels in batch:", labels.unique().tolist())

    print("\nSUCCESS: medmnist_loader is working.")