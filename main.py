"""
==============================================================================
LandCover.ai Classification using VGG16 Transfer Learning
Full Pipeline: Data Collection → Preprocessing → Augmentation →
               Feature Extraction → Transfer Learning → Evaluation
==============================================================================
Classes: Buildings, Vegetation, Roads, Water Bodies
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────
# 0.  REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS_FROZEN= 10          # train only top layers
EPOCHS_FINE  = 10          # fine-tune last conv block
NUM_CLASSES  = 4
CLASS_NAMES  = ["Buildings", "Vegetation", "Roads", "Water Bodies"]
DATA_DIR     = "dataset"   # change to your dataset root
PLOT_DIR     = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2.  DATA PIPELINE
# ─────────────────────────────────────────────

def build_generators(data_dir, img_size, batch_size):
    """
    Image resizing, Normalization, Label encoding,
    Train-validation-test split + Augmentation.
    """
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        # ── Color adjustments ──────────────────
        brightness_range=[0.8, 1.2],
        channel_shift_range=30,
        # ── Geometric transformations ──────────
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        validation_split=0.20,
    )
    test_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        seed=SEED,
    )
    val_gen = train_aug.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
    )
    test_gen = test_aug.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────
# 3.  MODEL: VGG16 TRANSFER LEARNING
# ─────────────────────────────────────────────

def build_model(num_classes, freeze_conv=True):
    """
    Feature Extraction Using VGG16:
      • Low-level layers  → frozen
      • High-level layers → optionally frozen
      • Global Average Pooling
      • Sigmoid / Softmax output layer
    """
    base = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )

    if freeze_conv:
        # Freeze first 5 convolution modules (all conv blocks)
        for layer in base.layers:
            layer.trainable = False
    else:
        # Fine-tune: unfreeze last conv block (block5)
        for layer in base.layers:
            layer.trainable = layer.name.startswith("block5")

    x = base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)       # Global Average Pooling
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Sigmoid for binary / Softmax for multiclass
    activation = "sigmoid" if num_classes == 1 else "softmax"
    out = layers.Dense(num_classes, activation=activation, name="predictions")(x)

    model = Model(inputs=base.input, outputs=out)
    return model


def compile_and_train(model, train_gen, val_gen, epochs, lr=1e-4, tag="frozen"):
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=0),
    ]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=cb,
        verbose=1,
    )
    return history



# ─────────────────────────────────────────────
# 5.  PLOTTING FUNCTIONS
# ─────────────────────────────────────────────

PALETTE = {
    "Buildings":    "#E63946",
    "Vegetation":   "#2A9D8F",
    "Roads":        "#E9C46A",
    "Water Bodies": "#457B9D",
    "macro":        "#6A0572",
}

def save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✔  Saved → {path}")
    plt.show()
    plt.close(fig)


# ── 5a. Accuracy Plot ──────────────────────────────────────────────────────
def plot_accuracy(hist):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_facecolor("#0F0F1A")

    ep = hist["epochs"]
    ax.plot(ep, hist["accuracy"],     color="#00D4FF", lw=2.2, label="Train Accuracy",      marker="o", markersize=4)
    ax.plot(ep, hist["val_accuracy"], color="#FF6B6B", lw=2.2, label="Validation Accuracy", marker="s", markersize=4, linestyle="--")
    ax.fill_between(ep, hist["accuracy"], hist["val_accuracy"], alpha=0.12, color="#A29BFE")

    ax.set_title("Model Accuracy — VGG16 Transfer Learning",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", color="#AAAAAA"); ax.set_ylabel("Accuracy", color="#AAAAAA")
    ax.tick_params(colors="#AAAAAA"); ax.set_ylim(0, 1.05)
    for sp in ax.spines.values(): sp.set_color("#333355")
    ax.legend(facecolor="#1A1A2E", labelcolor="white", framealpha=0.8)
    ax.grid(alpha=0.15, color="white")
    fig.tight_layout()
    save(fig, "01_model_accuracy.png")


# ── 5b. Loss Plot ──────────────────────────────────────────────────────────
def plot_loss(hist):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0F0F1A"); ax.set_facecolor("#0F0F1A")

    ep = hist["epochs"]
    ax.plot(ep, hist["loss"],     color="#FDB44B", lw=2.2, label="Train Loss",      marker="o", markersize=4)
    ax.plot(ep, hist["val_loss"], color="#FF6B6B", lw=2.2, label="Validation Loss", marker="s", markersize=4, linestyle="--")
    ax.fill_between(ep, hist["loss"], hist["val_loss"], alpha=0.12, color="#FDCB6E")

    ax.set_title("Model Loss — VGG16 Transfer Learning",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", color="#AAAAAA"); ax.set_ylabel("Loss", color="#AAAAAA")
    ax.tick_params(colors="#AAAAAA")
    for sp in ax.spines.values(): sp.set_color("#333355")
    ax.legend(facecolor="#1A1A2E", labelcolor="white", framealpha=0.8)
    ax.grid(alpha=0.15, color="white")
    fig.tight_layout()
    save(fig, "02_model_loss.png")


# ── 5c. ROC Curve ─────────────────────────────────────────────────────────
def plot_roc(y_true_idx, y_prob):
    y_bin = label_binarize(y_true_idx, classes=range(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0F0F1A"); ax.set_facecolor("#0F0F1A")

    roc_aucs = []
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=2, label=f"{cls}  (AUC={roc_auc:.3f})",
                color=list(PALETTE.values())[i])

    # Macro average
    all_fpr = np.unique(np.concatenate([roc_curve(y_bin[:, i], y_prob[:, i])[0] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= NUM_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, lw=2.5, linestyle=":", color=PALETTE["macro"],
            label=f"Macro Avg  (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5, label="Random")
    ax.set_title("ROC Curves — One-vs-Rest (OvR)", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate", color="#AAAAAA")
    ax.set_ylabel("True Positive Rate", color="#AAAAAA")
    ax.tick_params(colors="#AAAAAA"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    for sp in ax.spines.values(): sp.set_color("#333355")
    ax.legend(facecolor="#1A1A2E", labelcolor="white", framealpha=0.8, fontsize=9)
    ax.grid(alpha=0.15, color="white")
    fig.tight_layout()
    save(fig, "03_roc_curve.png")


# ── 5d. Precision-Recall Curve ────────────────────────────────────────────
def plot_precision_recall(y_true_idx, y_prob):
    y_bin = label_binarize(y_true_idx, classes=range(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0F0F1A"); ax.set_facecolor("#0F0F1A")

    for i, cls in enumerate(CLASS_NAMES):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, lw=2, label=f"{cls}  (AUC={pr_auc:.3f})",
                color=list(PALETTE.values())[i])

    ax.set_title("Precision-Recall Curves", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall", color="#AAAAAA"); ax.set_ylabel("Precision", color="#AAAAAA")
    ax.tick_params(colors="#AAAAAA"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    for sp in ax.spines.values(): sp.set_color("#333355")
    ax.legend(facecolor="#1A1A2E", labelcolor="white", framealpha=0.8, fontsize=9)
    ax.grid(alpha=0.15, color="white")
    fig.tight_layout()
    save(fig, "04_precision_recall.png")


# ── 5e. Calibration Plot ──────────────────────────────────────────────────
def plot_calibration(y_true_idx, y_prob):
    y_bin = label_binarize(y_true_idx, classes=range(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0F0F1A"); ax.set_facecolor("#0F0F1A")

    ax.plot([0, 1], [0, 1], "w--", lw=1.5, label="Perfect Calibration")
    for i, cls in enumerate(CLASS_NAMES):
        prob_true, prob_pred = calibration_curve(y_bin[:, i], y_prob[:, i], n_bins=10)
        bs = brier_score_loss(y_bin[:, i], y_prob[:, i])
        ax.plot(prob_pred, prob_true, marker="o", lw=2,
                label=f"{cls}  (Brier={bs:.3f})",
                color=list(PALETTE.values())[i])

    ax.set_title("Calibration Plot (Reliability Diagram)", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", color="#AAAAAA")
    ax.set_ylabel("Fraction of Positives", color="#AAAAAA")
    ax.tick_params(colors="#AAAAAA"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    for sp in ax.spines.values(): sp.set_color("#333355")
    ax.legend(facecolor="#1A1A2E", labelcolor="white", framealpha=0.8, fontsize=9)
    ax.grid(alpha=0.15, color="white")
    fig.tight_layout()
    save(fig, "05_calibration.png")


# ── 5f. FNR & FPR Plot ────────────────────────────────────────────────────
def plot_fnr_fpr(y_true_idx, y_prob):
    y_bin = label_binarize(y_true_idx, classes=range(NUM_CLASSES))
    thresholds_list = np.linspace(0.01, 0.99, 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.patch.set_facecolor("#0F0F1A")
    for ax in axes:
        ax.set_facecolor("#0F0F1A")
        ax.tick_params(colors="#AAAAAA")
        for sp in ax.spines.values(): sp.set_color("#333355")
        ax.grid(alpha=0.15, color="white")

    for i, cls in enumerate(CLASS_NAMES):
        fnrs, fprs = [], []
        for thr in thresholds_list:
            pred = (y_prob[:, i] >= thr).astype(int)
            tn = ((pred == 0) & (y_bin[:, i] == 0)).sum()
            fp = ((pred == 1) & (y_bin[:, i] == 0)).sum()
            fn = ((pred == 0) & (y_bin[:, i] == 1)).sum()
            tp = ((pred == 1) & (y_bin[:, i] == 1)).sum()
            fnrs.append(fn / (fn + tp + 1e-9))
            fprs.append(fp / (fp + tn + 1e-9))
        col = list(PALETTE.values())[i]
        axes[0].plot(thresholds_list, fnrs, lw=2, label=cls, color=col)
        axes[1].plot(thresholds_list, fprs, lw=2, label=cls, color=col)

    for ax, title in zip(axes, ["False Negative Rate (FNR)", "False Positive Rate (FPR)"]):
        ax.set_title(title, color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("Threshold", color="#AAAAAA")
        ax.set_ylabel("Rate", color="#AAAAAA"); ax.set_ylim(0, 1)
        ax.legend(facecolor="#1A1A2E", labelcolor="white", framealpha=0.8, fontsize=9)

    fig.suptitle("FNR & FPR vs Threshold", color="white", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "06_fnr_fpr.png")


# ── 5g. Performance Metrics Bar Plot ─────────────────────────────────────
def plot_performance_metrics(y_true_idx, y_pred_idx, y_prob):
    labels = CLASS_NAMES + ["Macro Avg"]
    metrics = {
        "Accuracy":  [],
        "Precision": [],
        "Recall":    [],
        "F1-Score":  [],
    }
    y_bin = label_binarize(y_true_idx, classes=range(NUM_CLASSES))

    for i in range(NUM_CLASSES):
        yt = (y_true_idx == i).astype(int)
        yp = (y_pred_idx == i).astype(int)
        metrics["Accuracy"].append(accuracy_score(yt, yp))
        metrics["Precision"].append(precision_score(yt, yp, zero_division=0))
        metrics["Recall"].append(recall_score(yt, yp, zero_division=0))
        metrics["F1-Score"].append(f1_score(yt, yp, zero_division=0))

    metrics["Accuracy"].append(accuracy_score(y_true_idx, y_pred_idx))
    metrics["Precision"].append(precision_score(y_true_idx, y_pred_idx, average="macro", zero_division=0))
    metrics["Recall"].append(recall_score(y_true_idx, y_pred_idx, average="macro", zero_division=0))
    metrics["F1-Score"].append(f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0))

    x = np.arange(len(labels))
    width = 0.20
    colors = ["#00D4FF", "#FF6B6B", "#2ECC71", "#FDB44B"]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0F0F1A"); ax.set_facecolor("#0F0F1A")

    for j, (metric, col) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + j * width, metrics[metric], width, label=metric, color=col, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7, color="white")

    ax.set_title("Performance Metrics per Class", color="white", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5); ax.set_xticklabels(labels, color="#AAAAAA", fontsize=10)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score", color="#AAAAAA")
    ax.tick_params(colors="#AAAAAA")
    for sp in ax.spines.values(): sp.set_color("#333355")
    ax.legend(facecolor="#1A1A2E", labelcolor="white", framealpha=0.8)
    ax.grid(axis="y", alpha=0.15, color="white")
    fig.tight_layout()
    save(fig, "07_performance_metrics.png")


# ── 5h. Confusion Matrix ──────────────────────────────────────────────────
def plot_confusion_matrix(y_true_idx, y_pred_idx):
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0F0F1A"); ax.set_facecolor("#0F0F1A")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})

    ax.set_title("Confusion Matrix", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", color="#AAAAAA"); ax.set_ylabel("Actual", color="#AAAAAA")
    ax.tick_params(colors="#CCCCCC", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", color="#CCCCCC")
    plt.setp(ax.get_yticklabels(), rotation=0, color="#CCCCCC")
    fig.tight_layout()
    save(fig, "08_confusion_matrix.png")


# ─────────────────────────────────────────────
# 6.  MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    USE_REAL_DATA = os.path.isdir(DATA_DIR)

    if USE_REAL_DATA:
        # ── Real dataset path ─────────────────
        print("=" * 60)
        print("  REAL DATASET DETECTED — running full pipeline")
        print("=" * 60)
        train_gen, val_gen, test_gen = build_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)

        # Phase 1: Feature Extraction (all conv layers frozen)
        print("\n[Phase 1] Feature Extraction — frozen VGG16 conv layers")
        model = build_model(NUM_CLASSES, freeze_conv=True)
        hist1 = compile_and_train(model, train_gen, val_gen, EPOCHS_FROZEN, lr=1e-3, tag="frozen")

        # Phase 2: Fine-tuning (unfreeze block5)
        print("\n[Phase 2] Fine-tuning — unfreeze VGG16 block5")
        for layer in model.layers:
            if hasattr(layer, "name") and layer.name.startswith("block5"):
                layer.trainable = True
        hist2 = compile_and_train(model, train_gen, val_gen, EPOCHS_FINE, lr=1e-5, tag="finetune")

        # Merge histories
        merged_hist = {}
        for k in ["accuracy", "val_accuracy", "loss", "val_loss"]:
            merged_hist[k] = hist1.history[k] + hist2.history[k]
        total_ep = EPOCHS_FROZEN + EPOCHS_FINE
        merged_hist["epochs"] = list(range(1, total_ep + 1))

        # Evaluate
        print("\n[Evaluation]")
        test_gen.reset()
        y_prob    = model.predict(test_gen, verbose=1)
        y_true_idx= test_gen.classes
        y_pred_idx= y_prob.argmax(axis=1)

    else:

        print("=" * 60)
        print("  No dataset found at './dataset'")

        print("  To use real data, place images at:")
        print("    dataset/train/<class>/<images>")
        print("    dataset/test/<class>/<images>")
        print("=" * 60)


    # ── Print classification report ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true_idx, y_pred_idx, target_names=CLASS_NAMES))

    # ── Generate all plots ────────────────────────────────────────────────
    print("\n[Plotting]")
    plot_accuracy(merged_hist)
    plot_loss(merged_hist)
    plot_roc(y_true_idx, y_prob)
    plot_precision_recall(y_true_idx, y_prob)
    plot_calibration(y_true_idx, y_prob)
    plot_fnr_fpr(y_true_idx, y_prob)
    plot_performance_metrics(y_true_idx, y_pred_idx, y_prob)
    plot_confusion_matrix(y_true_idx, y_pred_idx)

    print(f"\n✅  All plots saved to '{PLOT_DIR}/' folder")


if __name__ == "__main__":
    main()