"""
Validation/evaluation utilities for MedMNIST experiments.

Provides one main function, evaluate_model(), which runs a model over an
eval DataLoader and returns a dict of metrics:

    {
        'accuracy':           top-1 accuracy from sklearn
        'balanced_accuracy':  per-class accuracy averaged (immune to imbalance)
        'macro_f1':           per-class F1 averaged
        'per_class_f1':       list of per-class F1 scores
        'confusion_matrix':   2D numpy array
        'medmnist_acc':       MedMNIST official accuracy (matches benchmark)
        'medmnist_auc':       MedMNIST official AUC
    }

The MedMNIST metrics will be None if the dataset isn't a MedMNIST dataset
(the evaluator wouldn't apply).
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


@torch.no_grad()
def evaluate_model(model, loader, device, n_classes, medmnist_evaluator=None):
    """
    Run a full pass over `loader` and compute metrics.

    :param model:               PyTorch model to evaluate (already on `device`)
    :param loader:              DataLoader for the eval split (val or test)
    :param device:              'cuda' or 'cpu'
    :param n_classes:           number of classes in the dataset
    :param medmnist_evaluator:  optional medmnist.Evaluator instance for
                                official metrics. Pass None to skip.
    :return: dict of metrics (see module docstring)
    """

    # Switch to eval mode: disables dropout, freezes BatchNorm running stats.
    # This is critical for fair evaluation — we want deterministic outputs.
    was_training = model.training
    model.eval()

    all_logits = []
    all_preds = []
    all_targets = []

    # The evaluation loop
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_logits.append(probs)
        all_preds.append(preds)
        all_targets.append(targets.numpy())

    if was_training:
        model.train()

    # Concatenate batches into single arrays of shape [N], [N], [N, n_classes].
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_logits = np.concatenate(all_logits)

    # ------------------------------------------------------------------
    # sklearn metrics
    # ------------------------------------------------------------------
    metrics = {
        'accuracy': float(accuracy_score(all_targets, all_preds)),
        'balanced_accuracy': float(balanced_accuracy_score(all_targets, all_preds)),
        'macro_f1': float(f1_score(all_targets, all_preds, average='macro', zero_division=0)),
        'per_class_f1': f1_score(
            all_targets, all_preds, average=None, zero_division=0,
            labels=list(range(n_classes)),
        ).tolist(),
        'confusion_matrix': confusion_matrix(
            all_targets, all_preds, labels=list(range(n_classes))
        ).tolist(),
    }

    # ------------------------------------------------------------------
    # MedMNIST official metrics (optional)
    # ------------------------------------------------------------------
    if medmnist_evaluator is not None:
        try:
            # MedMNIST's evaluator expects probabilities of shape [N, n_classes]
            # and ground truth of shape [N, 1].
            mm_targets = all_targets.reshape(-1, 1)
            mm_auc, mm_acc = medmnist_evaluator.evaluate(all_logits, mm_targets)
            metrics['medmnist_auc'] = float(mm_auc)
            metrics['medmnist_acc'] = float(mm_acc)
        except Exception as e:
            # Don't crash the whole training run if the evaluator hiccups —
            # just log it and move on. sklearn metrics are the primary source.
            print(f'warning: MedMNIST evaluator failed: {e}')
            metrics['medmnist_auc'] = None
            metrics['medmnist_acc'] = None
    else:
        metrics['medmnist_auc'] = None
        metrics['medmnist_acc'] = None

    return metrics


def format_metrics(metrics, prefix=''):
    """
    One-line human-readable summary of the most important metrics.
    Used for terminal logging during training.
    """
    parts = [
        f'acc={metrics["accuracy"]:.4f}',
        f'bal_acc={metrics["balanced_accuracy"]:.4f}',
        f'macro_f1={metrics["macro_f1"]:.4f}',
    ]
    if metrics.get('medmnist_auc') is not None:
        parts.append(f'mm_auc={metrics["medmnist_auc"]:.4f}')
        parts.append(f'mm_acc={metrics["medmnist_acc"]:.4f}')
    return prefix + ' | '.join(parts)