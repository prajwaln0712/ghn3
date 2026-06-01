"""
Analyze GHN-3 vs Random Init experiments on MedMNIST datasets.

Parses two log files (one per init strategy) and produces:
  1. headline.png            — bar chart of accuracy / balanced_acc / macro_f1 / AUC
  2. per_class_precision.png — grouped bars for per-class precision
  3. per_class_f1.png        — grouped bars for per-class F1
  4. convergence.png         — balanced accuracy vs epoch for both inits

Usage:
    python analyze_results.py \
        --rand-log logs/derma_rand.out \
        --ghn3-log logs/derma_ghn3.out \
        --dataset dermamnist \
        --output-dir results

Run once per dataset (dermamnist, bloodmnist, pathmnist).
"""

import re
import ast
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# Class names for each dataset. Used as x-axis labels in per-class plots.
# Names are abbreviated for plot readability.
DATASET_CLASSES = {
    'dermamnist': ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
    'bloodmnist': ['baso', 'eosino', 'erythro', 'imm.gran',
                   'lympho', 'mono', 'neutro', 'platelet'],
    'pathmnist': ['adipose', 'backgr.', 'debris', 'lympho', 'mucus',
                  'sm.musc.', 'normal', 'stroma', 'tumor'],
}


# ---------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------

def parse_log(path):
    """
    Parse a training log file.

    Returns:
        epochs: list of tuples (epoch_num, acc, bal_acc, macro_f1, auc_or_None)
        final: dict with keys 'accuracy', 'balanced_accuracy', 'macro_f1',
               'auc', 'per_class_f1', 'confusion_matrix'
    """
    text = Path(path).read_text()

    # ----- per-epoch eval lines -----
    # Format:
    #   [eval epoch 001] acc=0.5900 | bal_acc=0.2226 | macro_f1=0.1776 [| auc=0.7821]
    eval_pattern = re.compile(
        r'\[eval epoch (\d+)\]\s+'
        r'acc=([\d.]+)\s*\|\s*'
        r'bal_acc=([\d.]+)\s*\|\s*'
        r'macro_f1=([\d.]+)'
        r'(?:\s*\|\s*auc=([\d.]+))?'  # AUC is optional (older runs don't have it)
    )
    epochs = []
    for m in eval_pattern.finditer(text):
        epoch = int(m.group(1))
        acc = float(m.group(2))
        bal_acc = float(m.group(3))
        macro_f1 = float(m.group(4))
        auc = float(m.group(5)) if m.group(5) else None
        epochs.append((epoch, acc, bal_acc, macro_f1, auc))

    # ----- final eval block -----
    final = {}

    final_pattern = re.compile(
        r'\[final\]\s+'
        r'acc=([\d.]+)\s*\|\s*'
        r'bal_acc=([\d.]+)\s*\|\s*'
        r'macro_f1=([\d.]+)'
        r'(?:\s*\|\s*auc=([\d.]+))?'
    )
    m = final_pattern.search(text)
    if m:
        final['accuracy'] = float(m.group(1))
        final['balanced_accuracy'] = float(m.group(2))
        final['macro_f1'] = float(m.group(3))
        final['auc'] = float(m.group(4)) if m.group(4) else None
    else:
        raise ValueError(f'Could not find [final] line in {path}')

    # per-class F1: [0.234, 0.456, ...]
    f1_pattern = re.compile(r'per-class F1:\s*(\[[^\]]+\])')
    m = f1_pattern.search(text)
    if m:
        final['per_class_f1'] = ast.literal_eval(m.group(1))
    else:
        raise ValueError(f'Could not find per-class F1 line in {path}')

    # confusion matrix: [[...], [...], ...]
    # Non-greedy match for everything between first `[[` and first `]]`.
    cm_pattern = re.compile(r'confusion matrix:\s*(\[\[.*?\]\])', re.DOTALL)
    m = cm_pattern.search(text)
    if m:
        final['confusion_matrix'] = np.array(ast.literal_eval(m.group(1)))
    else:
        raise ValueError(f'Could not find confusion matrix in {path}')

    return epochs, final


# ---------------------------------------------------------------------
# Metric derivation from confusion matrix
# ---------------------------------------------------------------------

def compute_per_class_precision(cm):
    """Per-class precision = diagonal[i] / column_sum[i]."""
    cm = np.asarray(cm)
    n = cm.shape[0]
    precision = np.zeros(n)
    for i in range(n):
        col_sum = cm[:, i].sum()
        precision[i] = cm[i, i] / col_sum if col_sum > 0 else 0.0
    return precision


def compute_per_class_recall(cm):
    """Per-class recall = diagonal[i] / row_sum[i]."""
    cm = np.asarray(cm)
    n = cm.shape[0]
    recall = np.zeros(n)
    for i in range(n):
        row_sum = cm[i, :].sum()
        recall[i] = cm[i, i] / row_sum if row_sum > 0 else 0.0
    return recall


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

# Consistent colors across all plots: grey for baseline, blue for GHN-3.
COLOR_RAND = '#888888'
COLOR_GHN3 = '#1f77b4'


def plot_headline_metrics(rand_final, ghn3_final, output_path, dataset):
    """Grouped bar chart of acc / bal_acc / macro_f1 / auc for both inits."""
    metrics = ['accuracy', 'balanced_accuracy', 'macro_f1']
    if rand_final.get('auc') is not None and ghn3_final.get('auc') is not None:
        metrics.append('auc')

    label_map = {
        'accuracy': 'Accuracy',
        'balanced_accuracy': 'Balanced Acc',
        'macro_f1': 'Macro F1',
        'auc': 'AUC',
    }

    rand_values = [rand_final[m] for m in metrics]
    ghn3_values = [ghn3_final[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, rand_values, width, label='Random Init', color=COLOR_RAND)
    ax.bar(x + width / 2, ghn3_values, width, label='GHN-3 Init', color=COLOR_GHN3)

    ax.set_ylabel('Score')
    ax.set_title(f'{dataset.upper()} — Random vs GHN-3 Init Headline Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[m] for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Numeric labels above each bar
    for i, v in enumerate(rand_values):
        ax.text(i - width / 2, v + 0.015, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(ghn3_values):
        ax.text(i + width / 2, v + 0.015, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_per_class_metric(rand_values, ghn3_values, class_names,
                          metric_name, output_path, dataset):
    """Grouped bars for any per-class metric."""
    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 1.2), 5))
    ax.bar(x - width / 2, rand_values, width, label='Random Init', color=COLOR_RAND)
    ax.bar(x + width / 2, ghn3_values, width, label='GHN-3 Init', color=COLOR_GHN3)

    ax.set_ylabel(metric_name)
    ax.set_xlabel('Class')
    ax.set_title(f'{dataset.upper()} — Per-Class {metric_name} Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_convergence(rand_epochs, ghn3_epochs, output_path, dataset):
    """Balanced accuracy vs epoch for both inits."""
    rand_e = [e[0] for e in rand_epochs]
    rand_bal = [e[2] for e in rand_epochs]
    ghn3_e = [e[0] for e in ghn3_epochs]
    ghn3_bal = [e[2] for e in ghn3_epochs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rand_e, rand_bal, marker='o', markersize=4, linewidth=1.5,
            label='Random Init', color=COLOR_RAND)
    ax.plot(ghn3_e, ghn3_bal, marker='s', markersize=4, linewidth=1.5,
            label='GHN-3 Init', color=COLOR_GHN3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title(f'{dataset.upper()} — Balanced Accuracy Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rand-log', required=True, help='Path to random-init log')
    parser.add_argument('--ghn3-log', required=True, help='Path to GHN-3-init log')
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CLASSES.keys()),
                        help='Which MedMNIST dataset')
    parser.add_argument('--output-dir', default='results',
                        help='Base directory for output PNGs (default: results)')
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Parsing {args.rand_log}...')
    rand_epochs, rand_final = parse_log(args.rand_log)
    print(f'  {len(rand_epochs)} eval epochs, '
          f'final acc={rand_final["accuracy"]:.4f}, '
          f'bal_acc={rand_final["balanced_accuracy"]:.4f}')

    print(f'Parsing {args.ghn3_log}...')
    ghn3_epochs, ghn3_final = parse_log(args.ghn3_log)
    print(f'  {len(ghn3_epochs)} eval epochs, '
          f'final acc={ghn3_final["accuracy"]:.4f}, '
          f'bal_acc={ghn3_final["balanced_accuracy"]:.4f}')

    class_names = DATASET_CLASSES[args.dataset]

    # Compute per-class precision from confusion matrices
    rand_precision = compute_per_class_precision(rand_final['confusion_matrix'])
    ghn3_precision = compute_per_class_precision(ghn3_final['confusion_matrix'])

    print()
    plot_headline_metrics(rand_final, ghn3_final,
                          out_dir / 'headline.png', args.dataset)
    print(f'  saved {out_dir / "headline.png"}')

    plot_per_class_metric(rand_precision, ghn3_precision, class_names,
                          'Precision', out_dir / 'per_class_precision.png',
                          args.dataset)
    print(f'  saved {out_dir / "per_class_precision.png"}')

    plot_per_class_metric(rand_final['per_class_f1'], ghn3_final['per_class_f1'],
                          class_names, 'F1',
                          out_dir / 'per_class_f1.png', args.dataset)
    print(f'  saved {out_dir / "per_class_f1.png"}')

    plot_convergence(rand_epochs, ghn3_epochs,
                     out_dir / 'convergence.png', args.dataset)
    print(f'  saved {out_dir / "convergence.png"}')

    # Print a summary table
    print(f'\n=== {args.dataset.upper()} SUMMARY ===')
    print(f'{"Metric":<22} {"Random":<12} {"GHN-3":<12} {"Delta":<10}')
    print('-' * 56)
    for key, label in [('accuracy', 'Accuracy'),
                       ('balanced_accuracy', 'Balanced Accuracy'),
                       ('macro_f1', 'Macro F1')]:
        r, g = rand_final[key], ghn3_final[key]
        print(f'{label:<22} {r:<12.4f} {g:<12.4f} {g - r:+.4f}')
    if rand_final.get('auc') is not None and ghn3_final.get('auc') is not None:
        r, g = rand_final['auc'], ghn3_final['auc']
        print(f'{"AUC":<22} {r:<12.4f} {g:<12.4f} {g - r:+.4f}')


if __name__ == '__main__':
    main()