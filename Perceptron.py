"""
Laboratory work 7 — Simplified perceptron for loan approval.
Student: George Mieabiye David  (variant #11)
Dataset: dataset_11.csv

Learning-rate schedule (fully automated):
    epochs     1 –  500    -> lr = 5.0
    epochs   501 – 1000     -> lr = 1.0
    epochs  1001 – 5000     -> lr = 0.1
    epochs  5001 – 10000    -> lr = 0.01

Model weights are snapshotted at the end of every phase, so per-phase
train/test accuracy and loss are reported alongside the final values.

Dependencies: numpy, matplotlib  (no pandas, no sklearn)
"""

import csv
import hashlib
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Student config

STUDENT_NAME = "George Mieabiye David"
STUDENT_ID   = 11


def seed_from_name(name: str) -> int:
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


SEED = seed_from_name(STUDENT_NAME)
np.random.seed(SEED)

OUT_DIR = "results_george"
os.makedirs(OUT_DIR, exist_ok=True)


# LEARNING-RATE SCHEDULE
# Each tuple: (start_epoch_1_based, end_epoch_1_based_inclusive, lr)

LR_SCHEDULE = [
    (1,    500,  5.0),
    (501,  1000, 1.0),
    (1001, 5000, 0.1),
    (5001, 10000, 0.01),
]

TOTAL_EPOCHS = LR_SCHEDULE[-1][1]


# Math helpers

def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# Data loading — pure numpy + stdlib csv

def load_data(path="dataset_11.csv"):
    features, labels = [], []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)
        for row in reader:
            if not row:
                continue
            vals = [float(v) for v in row]
            if len(vals) != n_cols:
                continue
            features.append(vals[:-1])
            labels.append(vals[-1])
    return (np.asarray(features, dtype=np.float64),
            np.asarray(labels,   dtype=np.float64))


# Train / test split

def train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True):
    rng = np.random.default_rng(random_state)
    n = len(y)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# Training with scheduled learning rate + phase snapshots

def train_perceptron_scheduled(X_train, y_train,
                               lr_schedule, total_epochs):
    """
    Batch-GD logistic-perceptron training with a piecewise-constant
    learning-rate schedule.

    Returns
    -------
    weights, bias
        Final model parameters.
    loss_history : np.ndarray of length total_epochs
        Per-epoch BCE loss on the training set.
    lr_history : np.ndarray of length total_epochs
        Per-epoch learning rate actually used.
    snapshots : dict {epoch_1based: (weights_copy, bias, loss)}
        Model parameters AND loss at the END of each phase.
    """
    samples, features = X_train.shape
    weights = np.random.rand(features) * 0.01
    bias = 0.0
    eps = 1e-15

    loss_history = np.empty(total_epochs)
    lr_history   = np.empty(total_epochs)

    # end-of-phase epochs where we snapshot the model
    phase_end_epochs = {end for _, end, _ in lr_schedule
                        if 1 <= end <= total_epochs}

    snapshots = {}

    for epoch in range(1, total_epochs + 1):
        # ---- pick lr for this epoch ----
        lr = None
        for start, end, rate in lr_schedule:
            if start <= epoch <= end:
                lr = rate
                break
        if lr is None:
            raise RuntimeError(f"No learning rate for epoch {epoch}")
        lr_history[epoch - 1] = lr

        # ---- forward + backward ----
        a = X_train @ weights + bias
        r = sigmoid(a)
        error = r - y_train
        dw = X_train.T @ error / samples
        db = error.mean()

        # ---- update ----
        weights -= lr * dw
        bias    -= lr * db

        # ---- loss for this epoch ----
        loss = -np.mean(
            y_train * np.log(r + eps) +
            (1.0 - y_train) * np.log(1.0 - r + eps)
        )
        loss_history[epoch - 1] = loss

        # ---- snapshot if this epoch ends a phase ----
        if epoch in phase_end_epochs:
            snapshots[epoch] = (weights.copy(), float(bias), float(loss))

    return weights, bias, loss_history, lr_history, snapshots


def evaluate(X, y, weights, bias):
    r = sigmoid(X @ weights + bias)
    return float(np.mean((r >= 0.5).astype(int) == y))


def bce_loss(X, y, weights, bias):
    eps = 1e-15
    r = sigmoid(X @ weights + bias)
    return float(-np.mean(y * np.log(r + eps) +
                          (1 - y) * np.log(1 - r + eps)))


# Main

def main():
    # ---- Load ----
    X, y = load_data("dataset_11.csv")
    print(f"Student  : {STUDENT_NAME} (#{STUDENT_ID})")
    print(f"Seed     : {SEED}")
    print(f"Dataset  : X={X.shape}, y={y.shape}, "
          f"approved fraction={y.mean():.4f}")
    print("LR schedule:")
    for s, e, lr in LR_SCHEDULE:
        print(f"   epochs {s:>5} – {e:<5}  lr = {lr}")
    print(f"Total epochs: {TOTAL_EPOCHS}\n")

    # ---- Split ----
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, shuffle=True
    )
    print(f"Train split: {X_tr.shape}   Test split: {X_te.shape}\n")

    # ---- Train ----
    np.random.seed(SEED)
    w, b, hist, lr_hist, snapshots = train_perceptron_scheduled(
        X_tr, y_tr, LR_SCHEDULE, TOTAL_EPOCHS
    )

    # ---- Final metrics ----
    acc_tr  = evaluate(X_tr, y_tr, w, b)
    acc_te  = evaluate(X_te, y_te, w, b)
    loss_tr = bce_loss(X_tr, y_tr, w, b)
    loss_te = bce_loss(X_te, y_te, w, b)

    print("Final results after scheduled training:")
    print(f"  Train accuracy : {acc_tr:.4f}")
    print(f"  Test  accuracy : {acc_te:.4f}")
    print(f"  Train loss     : {loss_tr:.6f}")
    print(f"  Test  loss     : {loss_te:.6f}")

    # ---- Per-phase diagnostics (fully populated) ----
    print("\nLoss / accuracy at the end of each phase:")
    print(f"  {'phase':<6}{'epochs':<16}{'lr':<8}"
          f"{'end_loss':<12}{'train_acc':<12}{'test_acc':<12}")
    phase_rows = []
    for i, (s, e, lr) in enumerate(LR_SCHEDULE, start=1):
        w_e, b_e, l_e = snapshots[e]
        a_tr = evaluate(X_tr, y_tr, w_e, b_e)
        a_te = evaluate(X_te, y_te, w_e, b_e)
        phase_rows.append((i, f"{s}-{e}", lr, l_e, a_tr, a_te))
        print(f"  {i:<6}{f'{s}-{e}':<16}{lr:<8}"
              f"{l_e:<12.6f}{a_tr:<12.4f}{a_te:<12.4f}")

    # ---- Plot 1: full 10k-epoch loss with phase boundaries ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    epochs_axis = np.arange(1, TOTAL_EPOCHS + 1)
    ax.plot(epochs_axis, hist, color="steelblue", label="training loss")

    phase_colors = ["crimson", "darkorange", "green", "navy"]
    for (s, e, lr), col in zip(LR_SCHEDULE, phase_colors):
        ax.axvline(s, color=col, linestyle="--",
                   linewidth=0.9, alpha=0.7)
        mid = (s + e) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.98, f"lr = {lr}",
                ha="center", va="top", fontsize=9, color=col)

    ax.set_title(f"Training loss with scheduled learning rate — "
                 f"{STUDENT_NAME}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average BCE loss")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR,
                             f"{STUDENT_ID}_loss_scheduled.png"),
                dpi=150)
    plt.close(fig)

    # ---- Plot 2: zoom on first 1000 epochs (linear x) ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs_axis[:1000], hist[:1000], color="steelblue")
    ax.axvline(500, color="crimson", linestyle="--",
               label="lr: 5.0 → 1.0")
    ax.set_title(f"First 1000 epochs (zoom) — {STUDENT_NAME}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average BCE loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR,
                             f"{STUDENT_ID}_loss_scheduled_zoom.png"),
                dpi=150)
    plt.close(fig)

    # ---- Save per-epoch loss (CSV) ----
    with open(os.path.join(OUT_DIR,
                           f"{STUDENT_ID}_scheduled_loss.csv"),
              "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "learning_rate", "loss"])
        for ep, lr, lo in zip(epochs_axis, lr_hist, hist):
            wr.writerow([int(ep), lr, f"{lo:.8f}"])

    # ---- Save per-phase table (CSV) ----
    with open(os.path.join(OUT_DIR,
                           f"{STUDENT_ID}_per_phase.csv"),
              "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["phase", "epochs", "learning_rate",
                     "end_loss", "train_acc", "test_acc"])
        for i, ep_range, lr, lo, a_tr, a_te in phase_rows:
            wr.writerow([i, ep_range, lr,
                         f"{lo:.6f}", f"{a_tr:.4f}", f"{a_te:.4f}"])

    # ---- Summary CSV ----
    with open(os.path.join(OUT_DIR, f"{STUDENT_ID}_summary.csv"),
              "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["student", "student_id", "seed", "total_epochs",
                     "train_acc", "test_acc",
                     "train_loss", "test_loss", "schedule"])
        sched_str = ";".join(f"{s}-{e}:{lr}"
                             for s, e, lr in LR_SCHEDULE)
        wr.writerow([STUDENT_NAME, STUDENT_ID, SEED, TOTAL_EPOCHS,
                     f"{acc_tr:.4f}", f"{acc_te:.4f}",
                     f"{loss_tr:.6f}", f"{loss_te:.6f}",
                     sched_str])

    print(f"\nAll outputs written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()