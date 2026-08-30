"""
CS3807 - Deep Learning Laboratory
Experiment 5: Comprehensive Study of CNN Training, Regularization,
Optimization, Hyperparameter Tuning, Transfer Learning and
Cross-Validation using MobileNetV2 on the Oxford-IIIT Pet Dataset.

Run this on a machine with internet access + (ideally) a GPU:
    pip install tensorflow tensorflow-datasets scikit-learn matplotlib seaborn

Usage:
    python experiment5_mobilenetv2.py --stage all
    python experiment5_mobilenetv2.py --stage init
    python experiment5_mobilenetv2.py --stage regularization
    python experiment5_mobilenetv2.py --stage batchnorm
    python experiment5_mobilenetv2.py --stage optimizers
    python experiment5_mobilenetv2.py --stage hyperparams
    python experiment5_mobilenetv2.py --stage transfer
    python experiment5_mobilenetv2.py --stage cv
    python experiment5_mobilenetv2.py --stage final

Each stage saves its plots as PDFs into ./figures/ (same filenames used
in the LaTeX report: plot1_init_loss.pdf ... plot14_confmat.pdf) and
prints result tables to stdout / results.json so you can paste real
numbers into the report tables.
"""

import os
import json
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, optimizers, initializers, regularizers
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

IMG_SIZE = 224
NUM_CLASSES = 37
BATCH_SIZE_DEFAULT = 32
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
RESULTS_PATH = "results.json"


# --------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------- #
def load_oxford_pets(batch_size=BATCH_SIZE_DEFAULT):
    """
    Loads the Oxford-IIIT Pet dataset via tensorflow_datasets, resizes to
    224x224x3, applies MobileNetV2 preprocessing, and returns
    train/val/test tf.data.Dataset objects. The TFDS 'train'+'test'
    splits are combined then re-split 70/15/15 so we can hold out a
    genuinely untouched final test set (per the "test set must remain
    untouched" requirement).
    """
    (train_raw, test_raw), info = tfds.load(
        "oxford_iiit_pet:3.*.*",
        split=["train", "test"],
        with_info=True,
        as_supervised=True,
    )
    full = train_raw.concatenate(test_raw)
    full = full.shuffle(8000, seed=42, reshuffle_each_iteration=False)

    n_total = info.splits["train"].num_examples + info.splits["test"].num_examples
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    def preprocess(image, label):
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    train_ds = full.take(n_train).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    remaining = full.skip(n_train)
    val_ds = remaining.take(n_val).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = remaining.skip(n_val).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, info.features["label"].names


# --------------------------------------------------------------------- #
# 2. Model builders
# --------------------------------------------------------------------- #
def build_scratch_cnn(init_name="he_normal", dropout=0.0, l2_reg=0.0, use_bn=False):
    """
    A small CNN trained FROM SCRATCH (no pretrained weights). Used for
    Section 5 (weight initialization), Section 6 (regularization) and
    Section 7 (batch normalization), where we want full control over
    initialization / BN placement rather than an already-pretrained
    backbone.
    """
    init_map = {
        "zeros": initializers.Zeros(),
        "random_normal": initializers.RandomNormal(mean=0.0, stddev=0.05),
        "xavier": initializers.GlorotUniform(),
        "he": initializers.HeNormal(),
    }
    kernel_init = init_map[init_name]
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    for filters in [32, 64, 128, 128]:
        x = layers.Conv2D(filters, 3, padding="same",
                           kernel_initializer=kernel_init,
                           kernel_regularizer=reg)(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu", kernel_initializer=kernel_init,
                      kernel_regularizer=reg)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax",
                            kernel_initializer=kernel_init)(x)

    return models.Model(inputs, outputs)


def build_mobilenetv2(dropout=0.3, freeze_base=True, unfreeze_from=None):
    """
    MobileNetV2 backbone pretrained on ImageNet + a new classification
    head. Used for Section 10 (transfer learning / fine-tuning) and the
    final Section 11-12 cross-validation / evaluation stages.

    freeze_base=True                -> Case A: Feature Extraction
    freeze_base=False, unfreeze_from=N -> Case B: Fine-Tuning
                                          (unfreeze layers >= N)
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = not freeze_base
    if not freeze_base and unfreeze_from is not None:
        for layer in base.layers[:unfreeze_from]:
            layer.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False if freeze_base else None)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base


def get_optimizer(name, lr):
    if name == "sgd":
        return optimizers.SGD(learning_rate=lr)
    if name == "momentum":
        return optimizers.SGD(learning_rate=lr, momentum=0.9)
    if name == "rmsprop":
        return optimizers.RMSprop(learning_rate=lr)
    if name == "adam":
        return optimizers.Adam(learning_rate=lr)
    raise ValueError(name)


# --------------------------------------------------------------------- #
# 3. Section 5: Weight initialization study
# --------------------------------------------------------------------- #
def run_weight_init_study(train_ds, val_ds, epochs=15):
    results = {}
    histories = {}
    for name in ["zeros", "random_normal", "xavier", "he"]:
        print(f"[Init study] training with {name} initialization...")
        model = build_scratch_cnn(init_name=name)
        model.compile(optimizer=optimizers.Adam(1e-3),
                       loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"])
        hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
        histories[name] = hist.history
        results[name] = {
            "final_train_loss": hist.history["loss"][-1],
            "best_val_acc": max(hist.history["val_accuracy"]) * 100,
        }

    labels = {"zeros": "Zero init", "random_normal": "Random init",
              "xavier": "Xavier/Glorot init", "he": "He init"}

    plt.figure(figsize=(5.2, 3.6))
    for k, v in histories.items():
        plt.plot(range(1, epochs + 1), v["loss"], marker="o", label=labels[k])
    plt.xlabel("Epoch"); plt.ylabel("Training Loss")
    plt.title("Plot 1: Training Loss vs Epoch (Weight Initialization)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/plot1_init_loss.pdf"); plt.close()

    plt.figure(figsize=(5.2, 3.6))
    for k, v in histories.items():
        plt.plot(range(1, epochs + 1), np.array(v["val_accuracy"]) * 100,
                  marker="o", label=labels[k])
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy (%)")
    plt.title("Plot 2: Validation Accuracy vs Epoch (Weight Initialization)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/plot2_init_acc.pdf"); plt.close()

    return results


# --------------------------------------------------------------------- #
# 4. Section 6: Regularization study
# --------------------------------------------------------------------- #
def run_regularization_study(train_ds, val_ds, epochs=15):
    configs = {
        "none":    dict(dropout=0.0, l2_reg=0.0, use_bn=False),
        "l2":      dict(dropout=0.0, l2_reg=1e-4, use_bn=False),
        "dropout": dict(dropout=0.5, l2_reg=0.0, use_bn=False),
        "bn":      dict(dropout=0.0, l2_reg=0.0, use_bn=True),
    }
    histories = {}
    for name, cfg in configs.items():
        print(f"[Regularization study] training with {name}...")
        model = build_scratch_cnn(init_name="he", **cfg)
        model.compile(optimizer=optimizers.Adam(1e-3),
                       loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"])
        hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
        histories[name] = hist.history

    # Plot 3: train/val accuracy, none vs dropout (biggest contrast)
    plt.figure(figsize=(5.4, 3.6))
    plt.plot(histories["none"]["accuracy"], 'o-', color='tab:red', label="Train Acc (No Reg.)")
    plt.plot(histories["none"]["val_accuracy"], 'o--', color='tab:red', label="Val Acc (No Reg.)")
    plt.plot(histories["dropout"]["accuracy"], 's-', color='tab:blue', label="Train Acc (Dropout)")
    plt.plot(histories["dropout"]["val_accuracy"], 's--', color='tab:blue', label="Val Acc (Dropout)")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Plot 3: Training & Validation Accuracy (Regularization)")
    plt.legend(fontsize=7.5); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/plot3_reg_acc.pdf"); plt.close()

    # Plot 4: train/val loss, "none" config to show overfitting
    plt.figure(figsize=(5.2, 3.6))
    plt.plot(histories["none"]["loss"], 'o-', label="Training Loss")
    plt.plot(histories["none"]["val_loss"], 's-', label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Plot 4: Training & Validation Loss (No Regularization)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/plot4_reg_loss.pdf"); plt.close()

    return {k: {"final_train_acc": v["accuracy"][-1] * 100,
                "final_val_acc": v["val_accuracy"][-1] * 100}
            for k, v in histories.items()}


# --------------------------------------------------------------------- #
# 5. Section 7: Batch Normalization study
# --------------------------------------------------------------------- #
def batchnorm_numerical_example():
    x = np.array([2.0, 4.0, 6.0, 8.0])
    mu = x.mean()
    var = x.var()
    x_hat = (x - mu) / np.sqrt(var + 1e-8)
    print("mu_B =", mu, "sigma_B^2 =", var, "x_hat =", x_hat)
    return mu, var, x_hat


def run_bn_study(train_ds, val_ds, epochs=15):
    hist_no_bn = None
    hist_bn = None
    for use_bn, tag in [(False, "without_bn"), (True, "with_bn")]:
        print(f"[BN study] training {tag}...")
        model = build_scratch_cnn(init_name="he", use_bn=use_bn)
        model.compile(optimizer=optimizers.Adam(1e-3),
                       loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"])
        hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
        if use_bn:
            hist_bn = hist.history
        else:
            hist_no_bn = hist.history

    plt.figure(figsize=(5.2, 3.6))
    plt.plot(np.array(hist_no_bn["val_accuracy"]) * 100, 'o-', label="Without BN")
    plt.plot(np.array(hist_bn["val_accuracy"]) * 100, 's-', label="With BN")
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy (%)")
    plt.title("Plot 5: With vs Without Batch Normalization")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/plot5_bn.pdf"); plt.close()


# --------------------------------------------------------------------- #
# 6. Section 8: Optimizer comparison
# --------------------------------------------------------------------- #
def run_optimizer_study(train_ds, val_ds, epochs=15):
    histories = {}
    for name in ["sgd", "momentum", "rmsprop", "adam"]:
        print(f"[Optimizer study] training with {name}...")
        model = build_scratch_cnn(init_name="he", use_bn=True)
        model.compile(optimizer=get_optimizer(name, 1e-3),
                       loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"])
        t0 = time.time()
        hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
        elapsed = time.time() - t0
        histories[name] = {"history": hist.history, "time": elapsed}

    plt.figure(figsize=(5.2, 3.6))
    for name, d in histories.items():
        plt.plot(d["history"]["loss"], marker="o", label=name.upper())
    plt.xlabel("Epoch"); plt.ylabel("Training Loss")
    plt.title("Plot 6: Training Loss vs Epoch (Optimizers)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/plot6_opt_loss.pdf"); plt.close()

    plt.figure(figsize=(5.2, 3.6))
    for name, d in histories.items():
        plt.plot(np.array(d["history"]["val_accuracy"]) * 100, marker="o", label=name.upper())
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy (%)")
    plt.title("Plot 7: Validation Accuracy vs Epoch (Optimizers)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/plot7_opt_acc.pdf"); plt.close()

    table = {name: {
        "final_loss": d["history"]["loss"][-1],
        "best_val_acc": max(d["history"]["val_accuracy"]) * 100,
        "epoch_to_converge": int(np.argmax(d["history"]["val_accuracy"])) + 1,
        "time_sec": d["time"],
    } for name, d in histories.items()}
    return table


# --------------------------------------------------------------------- #
# 7. Section 9: CNN hyperparameter tuning (grid search, one-at-a-time)
# --------------------------------------------------------------------- #
def run_hyperparameter_search(load_data_fn):
    lr_results, bs_results, do_results = {}, {}, {}

    # Learning rate sweep (fixed batch=32, dropout=0.25)
    for lr in [1e-3, 1e-4]:
        train_ds, val_ds, _, _ = load_data_fn(batch_size=32)
        model, base = build_mobilenetv2(dropout=0.25, freeze_base=True)
        model.compile(optimizer=optimizers.Adam(lr),
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        hist = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=2)
        lr_results[lr] = max(hist.history["val_accuracy"]) * 100

    # Batch size sweep (fixed lr=1e-3, dropout=0.25)
    for bs in [16, 32, 64]:
        train_ds, val_ds, _, _ = load_data_fn(batch_size=bs)
        model, base = build_mobilenetv2(dropout=0.25, freeze_base=True)
        model.compile(optimizer=optimizers.Adam(1e-3),
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        hist = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=2)
        bs_results[bs] = max(hist.history["val_accuracy"]) * 100

    # Dropout sweep (fixed lr=1e-3, batch=32)
    for do in [0.0, 0.25, 0.5]:
        train_ds, val_ds, _, _ = load_data_fn(batch_size=32)
        model, base = build_mobilenetv2(dropout=do, freeze_base=True)
        model.compile(optimizer=optimizers.Adam(1e-3),
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        hist = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=2)
        do_results[do] = max(hist.history["val_accuracy"]) * 100

    def plot_sweep(d, xlabel, title, fname):
        xs = [str(k) for k in d.keys()]
        ys = list(d.values())
        plt.figure(figsize=(4.6, 3.6))
        plt.plot(xs, ys, 'o-', markersize=9)
        for x, y in zip(xs, ys):
            plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=8)
        plt.xlabel(xlabel); plt.ylabel("Validation Accuracy (%)")
        plt.title(title); plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/{fname}"); plt.close()

    plot_sweep(lr_results, "Learning Rate", "Plot 8: Learning Rate vs Validation Accuracy", "plot8_lr.pdf")
    plot_sweep(bs_results, "Batch Size", "Plot 9: Batch Size vs Validation Accuracy", "plot9_bs.pdf")
    plot_sweep(do_results, "Dropout Rate", "Plot 10: Dropout Rate vs Validation Accuracy", "plot10_dropout.pdf")

    return {"learning_rate": lr_results, "batch_size": bs_results, "dropout": do_results}


# --------------------------------------------------------------------- #
# 8. Section 10: Transfer learning vs fine-tuning
# --------------------------------------------------------------------- #
def run_transfer_learning_study(train_ds, val_ds, epochs=15):
    # Case A: Feature extraction
    print("[Transfer learning] Case A: feature extraction...")
    fe_model, fe_base = build_mobilenetv2(dropout=0.25, freeze_base=True)
    fe_model.compile(optimizer=optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    fe_hist = fe_model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)

    # Case B: Fine-tuning (unfreeze top ~30 layers, small LR)
    print("[Transfer learning] Case B: fine-tuning...")
    ft_model, ft_base = build_mobilenetv2(dropout=0.25, freeze_base=False,
                                           unfreeze_from=len(fe_base.layers) - 30)
    ft_model.set_weights(fe_model.get_weights())  # start from feature-extraction weights
    ft_model.compile(optimizer=optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    ft_hist = ft_model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)

    plt.figure(figsize=(5.2, 3.6))
    plt.plot(np.array(fe_hist.history["val_accuracy"]) * 100, 'o-', label="Feature Extraction")
    plt.plot(np.array(ft_hist.history["val_accuracy"]) * 100, 's-', label="Fine-Tuning")
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy (%)")
    plt.title("Plot 11: Feature Extraction vs Fine-Tuning")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/plot11_fe_ft.pdf"); plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    axs[0].plot(fe_hist.history["loss"], 'o-', label="Train")
    axs[0].plot(fe_hist.history["val_loss"], 's-', label="Val")
    axs[0].set_title("Feature Extraction"); axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss")
    axs[0].legend(fontsize=8)
    axs[1].plot(ft_hist.history["loss"], 'o-', label="Train")
    axs[1].plot(ft_hist.history["val_loss"], 's-', label="Val")
    axs[1].set_title("Fine-Tuning"); axs[1].set_xlabel("Epoch")
    axs[1].legend(fontsize=8)
    fig.suptitle("Plot 12: Training & Validation Loss Before/After Fine-Tuning")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/plot12_ft_loss.pdf"); plt.close()

    return ft_model


# --------------------------------------------------------------------- #
# 9. Section 11: 5-fold cross-validation over promising configs
# --------------------------------------------------------------------- #
def run_kfold_cv(all_images, all_labels, configs, k=5, epochs=10):
    """
    configs: dict of name -> dict(dropout=..., lr=..., freeze_base=...,
             unfreeze_from=...)
    all_images / all_labels: numpy arrays (loaded fully into memory,
    or replace this with a tf.data pipeline + manual index slicing for
    large datasets).
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    cv_results = {name: [] for name in configs}

    for name, cfg in configs.items():
        print(f"[K-Fold CV] config {name}")
        fold_accs = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_images)):
            model, base = build_mobilenetv2(dropout=cfg["dropout"],
                                             freeze_base=cfg["freeze_base"],
                                             unfreeze_from=cfg.get("unfreeze_from"))
            model.compile(optimizer=optimizers.Adam(cfg["lr"]),
                          loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model.fit(all_images[train_idx], all_labels[train_idx],
                      validation_data=(all_images[val_idx], all_labels[val_idx]),
                      epochs=epochs, batch_size=32, verbose=2)
            val_acc = model.evaluate(all_images[val_idx], all_labels[val_idx], verbose=0)[1]
            fold_accs.append(val_acc * 100)
            print(f"  fold {fold_idx+1}: {val_acc*100:.2f}%")
        cv_results[name] = fold_accs

    means = {name: np.mean(v) for name, v in cv_results.items()}
    sds = {name: np.std(v) for name, v in cv_results.items()}

    plt.figure(figsize=(5.6, 3.8))
    plt.bar(list(means.keys()), list(means.values()),
            yerr=list(sds.values()), capsize=6)
    plt.ylabel("Mean Validation Accuracy (%)")
    plt.title("Plot 13: 5-Fold Cross-Validation Accuracy")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/plot13_cv.pdf"); plt.close()

    return cv_results, means, sds


# --------------------------------------------------------------------- #
# 10. Section 12: Final held-out test evaluation
# --------------------------------------------------------------------- #
def run_final_evaluation(model, test_ds, class_names):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = (y_true == y_pred).mean() * 100
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
    plt.yticks(range(len(class_names)), class_names, fontsize=6)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Plot 14: Confusion Matrix (Full Test Set)")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/plot14_confmat.pdf"); plt.close()

    n_params = model.count_params()
    return {"test_accuracy": acc, "precision": prec, "recall": rec,
            "f1_score": f1, "num_parameters": n_params}


# --------------------------------------------------------------------- #
# Main / CLI
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                         choices=["all", "init", "regularization", "batchnorm",
                                  "optimizers", "hyperparams", "transfer", "cv", "final"])
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    results = {}
    train_ds, val_ds, test_ds, class_names = load_oxford_pets()

    if args.stage in ("all", "init"):
        results["weight_init"] = run_weight_init_study(train_ds, val_ds, args.epochs)

    if args.stage in ("all", "regularization"):
        results["regularization"] = run_regularization_study(train_ds, val_ds, args.epochs)

    if args.stage in ("all", "batchnorm"):
        batchnorm_numerical_example()
        run_bn_study(train_ds, val_ds, args.epochs)

    if args.stage in ("all", "optimizers"):
        results["optimizers"] = run_optimizer_study(train_ds, val_ds, args.epochs)

    if args.stage in ("all", "hyperparams"):
        results["hyperparams"] = run_hyperparameter_search(load_oxford_pets)

    ft_model = None
    if args.stage in ("all", "transfer"):
        ft_model = run_transfer_learning_study(train_ds, val_ds, args.epochs)

    if args.stage in ("all", "final") and ft_model is not None:
        results["final_test"] = run_final_evaluation(ft_model, test_ds, class_names)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
