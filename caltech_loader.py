"""
Caltech-101 and Caltech-256 data loader for GHN-3 experiments.

Mirrors the interface of ppuda's image_loader so train_ddp.py can branch
on dataset name and dispatch to either this loader, medmnist_loader, or
the original image_loader.

Caltech datasets ship as one folder per class with no predefined train/test
split. We create a deterministic stratified 80/20 split (per-class) using
a fixed RNG seed so the split is identical across runs.

The "junk" class is dropped:
  - Caltech 101 -> drop 'BACKGROUND_Google'
  - Caltech 256 -> drop '257.clutter'

Returns: (train_loader, valid_loader, n_classes) -- same shape as image_loader.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets as tv_datasets
from ppuda.vision.transforms import transforms_imagenet


# Map our dataset name to the torchvision class and the junk class to drop.
DATASET_REGISTRY = {
    'caltech101': (tv_datasets.Caltech101, 'BACKGROUND_Google'),
    'caltech256': (tv_datasets.Caltech256, '257.clutter'),
}


class _TransformWrapper(Dataset):
    """
    Wraps a Caltech dataset slice so we can apply different transforms to
    train and val splits. Caltech's __getitem__ returns a PIL image, which
    is what torchvision transforms expect.

    Also forces 3-channel RGB output -- some Caltech images are grayscale,
    and ResNet-50 expects 3 channels.
    """
    def __init__(self, base, indices, transform, labels):
        self.base = base
        self.indices = indices
        self.transform = transform
        self.labels = labels  # already remapped to [0, n_classes)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, _ = self.base[self.indices[i]]  # ignore base's original label
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # Return the remapped label (after dropping junk class).
        return img, int(self.labels[i])


def _build_index_mapping(base, dropped_class_name):
    """
    Build:
      - keep_indices: indices into `base` that are NOT in the dropped class
      - new_labels:   remapped contiguous labels [0, n_classes)
      - n_classes:    number of remaining classes
    """
    categories = base.categories  # class folder names, ordered by label
    if dropped_class_name in categories:
        dropped_label = categories.index(dropped_class_name)
    else:
        dropped_label = None

    raw_labels_all = np.asarray(base.y, dtype=np.int64)

    if dropped_label is None:
        keep_indices = np.arange(len(raw_labels_all))
    else:
        keep_indices = np.where(raw_labels_all != dropped_label)[0]

    raw_labels = raw_labels_all[keep_indices]

    # Remap remaining labels to contiguous [0, n_classes).
    unique_labels = sorted(set(raw_labels.tolist()))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    new_labels = np.array([label_map[l] for l in raw_labels.tolist()],
                          dtype=np.int64)

    return keep_indices, new_labels, len(unique_labels)


def _stratified_split(labels, train_frac, seed):
    """
    Within each class, take train_frac for train and the rest for test.
    Deterministic for a given seed. Returns local indices into `labels`.
    """
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        n_train = max(1, int(round(train_frac * len(cls_indices))))
        train_idx.extend(cls_indices[:n_train].tolist())
        test_idx.extend(cls_indices[n_train:].tolist())
    return (np.array(train_idx, dtype=np.int64),
            np.array(test_idx, dtype=np.int64))


def caltech_loader(dataset='caltech101',
                   data_dir='./data',
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
                   train_frac=0.8,
                   **kwargs):
    """
    Build train/eval DataLoaders for Caltech 101 or Caltech 256.

    Signature mirrors ppuda.vision.loader.image_loader so it can be used
    as a drop-in replacement.

    :param dataset:           'caltech101' or 'caltech256'
    :param data_dir:          dir for data (torchvision creates subfolders)
    :param test:              True -> use the test partition for evaluation
                              (Caltech has no separate val split)
    :param im_size:           image resolution (224 recommended for ResNet-50)
    :param batch_size:        training batch size
    :param test_batch_size:   evaluation batch size
    :param num_workers:       DataLoader worker processes
    :param seed:              RNG seed (drives the stratified split)
    :param load_train_anyway: build the train split even when test=True
    :param ddp:               True -> use DistributedSampler
    :param transforms_train_val: optional (train, val) transform tuple.
                                 Defaults to ppuda's transforms_imagenet.
    :param verbose:           print a summary of what was loaded
    :param train_frac:        fraction per class for train (default 0.8)
    :param kwargs:            silently absorbs extra args (cutout, noise, ...)
    :return: (train_loader, valid_loader, n_classes)
    """

    name = dataset.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(
            "Unknown Caltech dataset '{}'. Supported: {}".format(
                dataset, sorted(DATASET_REGISTRY.keys())))

    DatasetClass, dropped_class = DATASET_REGISTRY[name]

    os.makedirs(data_dir, exist_ok=True)

    if transforms_train_val is None:
        transforms_train_val = transforms_imagenet(im_size=im_size)
    train_transform, valid_transform = transforms_train_val

    # Load the underlying Caltech dataset. torchvision downloads to a
    # subfolder of data_dir (caltech101/ or caltech256/) the first time.
    base = DatasetClass(root=data_dir, download=False)

    keep_indices, new_labels, n_classes = _build_index_mapping(base, dropped_class)

    train_local, test_local = _stratified_split(new_labels, train_frac, seed)

    # Map local indices back into the underlying base dataset.
    train_base_idx = keep_indices[train_local]
    test_base_idx = keep_indices[test_local]
    train_labels = new_labels[train_local]
    test_labels = new_labels[test_local]

    valid_data = _TransformWrapper(base, test_base_idx, valid_transform, test_labels)

    train_data = None
    if not test or load_train_anyway:
        train_data = _TransformWrapper(base, train_base_idx,
                                       train_transform, train_labels)

    if verbose:
        print("loaded {}: {} classes, {} train samples, {} test samples".format(
            name.upper(), n_classes,
            len(train_data) if train_data is not None else 'none',
            len(valid_data)))

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


# -------------------------------------------------------------------------
# Smoke test: `python caltech_loader.py [caltech101|caltech256]`
# Downloads + parses + checks a batch. No GHN-3 or training involved.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else 'caltech101'
    print(f"Smoke testing caltech_loader on {name}...")

    train_loader, val_loader, n_classes = caltech_loader(
        dataset=name,
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
    print("  images.shape:", images.shape, "dtype:", images.dtype)
    print("  labels.shape:", labels.shape, "dtype:", labels.dtype)
    print("  unique labels in batch:", torch.unique(labels).tolist())

    print("\nSUCCESS: caltech_loader is working.")