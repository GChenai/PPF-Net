#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_thz_csv_module():
    module_path = REPO_ROOT / "ppfnet" / "thz_csv.py"
    module_name = "ppfnet_thz_csv_standalone"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load THz CSV module from {0}".format(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_THZ_CSV_MODULE = _load_thz_csv_module()
THzCube = _THZ_CSV_MODULE.THzCube
extract_valid_spectra = _THZ_CSV_MODULE.extract_valid_spectra
load_thz_csv = _THZ_CSV_MODULE.load_thz_csv


@dataclass
class SampleRecord:
    sample_id: str
    class_name: str
    class_index: int
    csv_path: str
    valid_pixel_count: int
    feature_vector: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a five-class THz watermelon-seed classifier on FS CSV cubes."
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/thz_seed_only/FS"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fs_seed_classifier"))
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["A", "B", "C", "D", "E"],
        help="Class folder names under data-root.",
    )
    parser.add_argument(
        "--model",
        choices=["random_forest", "svm", "logreg", "mlp"],
        default="random_forest",
    )
    parser.add_argument(
        "--feature-set",
        choices=["mean", "mean_std", "mean_std_quantile"],
        default="mean_std_quantile",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=0)
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--svm-c", type=float, default=2.0)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def summarize_spectrum_set(spectra: np.ndarray, feature_set: str) -> np.ndarray:
    if spectra.size == 0:
        raise ValueError("No valid spectra found in sample.")

    summaries: List[np.ndarray] = [
        np.nanmean(spectra, axis=0, dtype=np.float64).astype(np.float32),
    ]
    if feature_set in {"mean_std", "mean_std_quantile"}:
        summaries.append(np.nanstd(spectra, axis=0, dtype=np.float64).astype(np.float32))
    if feature_set == "mean_std_quantile":
        summaries.append(np.nanmedian(spectra, axis=0).astype(np.float32))
        summaries.append(np.nanpercentile(spectra, 25, axis=0).astype(np.float32))
        summaries.append(np.nanpercentile(spectra, 75, axis=0).astype(np.float32))
    return np.concatenate(summaries, axis=0).astype(np.float32, copy=False)


def extract_global_features(cube_data: THzCube, spectra: np.ndarray) -> np.ndarray:
    valid_pixel_count = int(cube_data.valid_mask.sum())
    total_pixel_count = int(cube_data.valid_mask.size)
    valid_ratio = float(valid_pixel_count / max(total_pixel_count, 1))
    spectrum_mean = float(np.nanmean(spectra))
    spectrum_std = float(np.nanstd(spectra))
    spectrum_min = float(np.nanmin(spectra))
    spectrum_max = float(np.nanmax(spectra))
    return np.asarray(
        [
            valid_pixel_count,
            total_pixel_count,
            valid_ratio,
            spectrum_mean,
            spectrum_std,
            spectrum_min,
            spectrum_max,
        ],
        dtype=np.float32,
    )


def extract_sample_feature(cube_data: THzCube, feature_set: str) -> Tuple[np.ndarray, int]:
    spectra = extract_valid_spectra(cube_data)
    spectral_features = summarize_spectrum_set(spectra, feature_set=feature_set)
    global_features = extract_global_features(cube_data, spectra)
    return np.concatenate([spectral_features, global_features], axis=0).astype(np.float32), int(spectra.shape[0])


def build_records(data_root: Path, classes: Sequence[str], feature_set: str) -> Tuple[List[SampleRecord], np.ndarray]:
    records: List[SampleRecord] = []
    axis_values_ref: np.ndarray | None = None

    for class_index, class_name in enumerate(classes):
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError("Class directory not found: {0}".format(class_dir))

        for csv_path in sorted(class_dir.glob("*.csv")):
            cube_data = load_thz_csv(csv_path)
            if axis_values_ref is None:
                axis_values_ref = cube_data.axis_values.astype(np.float32, copy=True)
            elif not np.allclose(axis_values_ref, cube_data.axis_values, atol=1e-6):
                raise ValueError("Axis values are inconsistent across samples.")

            feature_vector, valid_pixel_count = extract_sample_feature(cube_data, feature_set=feature_set)
            records.append(
                SampleRecord(
                    sample_id=csv_path.stem,
                    class_name=class_name,
                    class_index=class_index,
                    csv_path=str(csv_path.resolve()),
                    valid_pixel_count=valid_pixel_count,
                    feature_vector=feature_vector,
                )
            )

    if not records:
        raise ValueError("No CSV samples were found under {0}".format(data_root))
    if axis_values_ref is None:
        raise ValueError("Failed to infer THz axis values.")
    return records, axis_values_ref


def split_indices(y: np.ndarray, test_size: float, val_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(outer_split.split(np.zeros_like(y), y))

    y_train_val = y[train_val_idx]
    val_ratio_within_train_val = val_size / max(1e-8, 1.0 - test_size)
    inner_split = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_within_train_val, random_state=seed + 1)
    train_inner, val_inner = next(inner_split.split(np.zeros_like(y_train_val), y_train_val))

    train_idx = train_val_idx[train_inner]
    val_idx = train_val_idx[val_inner]
    return train_idx, val_idx, test_idx


def build_classifier(args: argparse.Namespace) -> Pipeline:
    steps: List[Tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]

    needs_scaling = args.model in {"svm", "logreg", "mlp"}
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))

    if args.pca_components and args.pca_components > 0:
        steps.append(("pca", PCA(n_components=args.pca_components, random_state=args.seed)))

    estimator: ClassifierMixin
    if args.model == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=args.rf_trees,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
    elif args.model == "svm":
        estimator = SVC(
            C=args.svm_c,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=args.seed,
        )
    elif args.model == "logreg":
        estimator = LogisticRegression(
            C=args.logreg_c,
            max_iter=5000,
            class_weight="balanced",
            multi_class="auto",
            random_state=args.seed,
        )
    else:
        estimator = MLPClassifier(
            hidden_layer_sizes=(args.mlp_hidden, max(32, args.mlp_hidden // 2)),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=1500,
            early_stopping=True,
            random_state=args.seed,
        )

    steps.append(("classifier", estimator))
    return Pipeline(steps)


def ensure_probability_estimates(model: Pipeline, x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return model

    calibrator = CalibratedClassifierCV(classifier, method="sigmoid", cv=3)
    steps = [(name, step) for name, step in model.steps if name != "classifier"]
    steps.append(("classifier", calibrator))
    calibrated = Pipeline(steps)
    calibrated.fit(x_train, y_train)
    return calibrated


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    labels: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    metrics["per_class"] = {
        class_name: {
            "precision": float(precision_per_class[idx]),
            "recall": float(recall_per_class[idx]),
            "f1": float(f1_per_class[idx]),
            "support": int(support_per_class[idx]),
        }
        for idx, class_name in enumerate(class_names)
    }
    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_normalized"] = (
        cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    ).tolist()

    if y_proba is not None:
        metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=list(labels)))
        topk = min(2, len(labels))
        metrics["top_{0}_accuracy".format(topk)] = float(top_k_accuracy_score(y_true, y_proba, k=topk, labels=list(labels)))

        y_true_bin = label_binarize(y_true, classes=list(labels))
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
            )
            metrics["roc_auc_ovo_macro"] = float(
                roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovo")
            )
        except ValueError:
            metrics["roc_auc_ovr_macro"] = None
            metrics["roc_auc_ovo_macro"] = None

    return metrics


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_predictions_csv(
    path: Path,
    records: Sequence[SampleRecord],
    indices: np.ndarray,
    class_names: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "class_name", "predicted_class", "csv_path", "valid_pixel_count"]
    if y_proba is not None:
        fieldnames.extend(["prob_{0}".format(class_name) for class_name in class_names])

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, sample_index in enumerate(indices.tolist()):
            record = records[sample_index]
            row = {
                "sample_id": record.sample_id,
                "class_name": record.class_name,
                "predicted_class": class_names[int(y_pred[row_idx])],
                "csv_path": record.csv_path,
                "valid_pixel_count": record.valid_pixel_count,
            }
            if y_proba is not None:
                for class_offset, class_name in enumerate(class_names):
                    row["prob_{0}".format(class_name)] = float(y_proba[row_idx, class_offset])
            writer.writerow(row)


def save_feature_importance(
    path: Path,
    model: Pipeline,
    feature_names: Sequence[str],
) -> None:
    classifier = model.named_steps["classifier"]
    importances: np.ndarray | None = None

    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_, dtype=np.float64)
    elif hasattr(classifier, "coef_"):
        coef = np.asarray(classifier.coef_, dtype=np.float64)
        importances = np.mean(np.abs(coef), axis=0)

    if importances is None:
        return

    order = np.argsort(importances)[::-1]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "feature_name", "importance"])
        writer.writeheader()
        for rank, feature_index in enumerate(order.tolist(), start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "feature_name": feature_names[feature_index],
                    "importance": float(importances[feature_index]),
                }
            )


def plot_confusion_matrix(path: Path, matrix: np.ndarray, class_names: Sequence[str], title: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = float(matrix.max() * 0.5) if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "{0:.2f}".format(value) if matrix.dtype.kind == "f" else str(int(value))
            ax.text(j, i, text, ha="center", va="center", color="white" if value > threshold else "black")

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_feature_names(axis_values: np.ndarray, feature_set: str) -> List[str]:
    prefixes = ["mean"]
    if feature_set in {"mean_std", "mean_std_quantile"}:
        prefixes.append("std")
    if feature_set == "mean_std_quantile":
        prefixes.extend(["median", "q25", "q75"])

    names: List[str] = []
    for prefix in prefixes:
        for axis_value in axis_values.tolist():
            names.append("{0}_{1:.6f}THz".format(prefix, float(axis_value)))
    names.extend(
        [
            "valid_pixel_count",
            "total_pixel_count",
            "valid_ratio",
            "global_mean",
            "global_std",
            "global_min",
            "global_max",
        ]
    )
    return names


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, axis_values = build_records(args.data_root, args.classes, feature_set=args.feature_set)
    x = np.stack([record.feature_vector for record in records]).astype(np.float32)
    y = np.asarray([record.class_index for record in records], dtype=np.int64)

    label_encoder = LabelEncoder()
    label_encoder.fit(args.classes)
    class_names = list(label_encoder.classes_)
    labels = list(range(len(class_names)))

    train_idx, val_idx, test_idx = split_indices(y, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    model = build_classifier(args)
    model.fit(x_train, y_train)
    model = ensure_probability_estimates(model, x_train, y_train)

    y_pred_val = model.predict(x_val)
    y_pred_test = model.predict(x_test)
    y_proba_val = model.predict_proba(x_val) if hasattr(model, "predict_proba") else None
    y_proba_test = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None

    val_metrics = compute_metrics(y_val, y_pred_val, y_proba_val, labels=labels, class_names=class_names)
    test_metrics = compute_metrics(y_test, y_pred_test, y_proba_test, labels=labels, class_names=class_names)

    split_summary = {
        "total_samples": int(len(records)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "class_distribution": {
            class_name: int(sum(record.class_name == class_name for record in records))
            for class_name in class_names
        },
    }
    config = {
        "data_root": str(args.data_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "classes": list(args.classes),
        "model": args.model,
        "feature_set": args.feature_set,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "seed": args.seed,
        "pca_components": args.pca_components,
        "rf_trees": args.rf_trees,
        "n_jobs": args.n_jobs,
        "svm_c": args.svm_c,
        "logreg_c": args.logreg_c,
        "mlp_hidden": args.mlp_hidden,
    }
    payload = {
        "config": config,
        "split_summary": split_summary,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(args.output_dir / "metrics_summary.json", payload)

    save_predictions_csv(
        args.output_dir / "validation_predictions.csv",
        records=records,
        indices=val_idx,
        class_names=class_names,
        y_true=y_val,
        y_pred=y_pred_val,
        y_proba=y_proba_val,
    )
    save_predictions_csv(
        args.output_dir / "test_predictions.csv",
        records=records,
        indices=test_idx,
        class_names=class_names,
        y_true=y_test,
        y_pred=y_pred_test,
        y_proba=y_proba_test,
    )

    feature_names = build_feature_names(axis_values, feature_set=args.feature_set)
    save_feature_importance(args.output_dir / "feature_importance.csv", model=model, feature_names=feature_names)

    plot_confusion_matrix(
        args.output_dir / "validation_confusion_matrix.png",
        np.asarray(val_metrics["confusion_matrix"], dtype=np.int64),
        class_names=class_names,
        title="Validation Confusion Matrix",
        dpi=args.dpi,
    )
    plot_confusion_matrix(
        args.output_dir / "validation_confusion_matrix_normalized.png",
        np.asarray(val_metrics["confusion_matrix_normalized"], dtype=np.float64),
        class_names=class_names,
        title="Validation Confusion Matrix (Normalized)",
        dpi=args.dpi,
    )
    plot_confusion_matrix(
        args.output_dir / "test_confusion_matrix.png",
        np.asarray(test_metrics["confusion_matrix"], dtype=np.int64),
        class_names=class_names,
        title="Test Confusion Matrix",
        dpi=args.dpi,
    )
    plot_confusion_matrix(
        args.output_dir / "test_confusion_matrix_normalized.png",
        np.asarray(test_metrics["confusion_matrix_normalized"], dtype=np.float64),
        class_names=class_names,
        title="Test Confusion Matrix (Normalized)",
        dpi=args.dpi,
    )

    print("Finished training FS seed classifier.")
    print("Validation accuracy: {0:.4f}".format(val_metrics["accuracy"]))
    print("Test accuracy: {0:.4f}".format(test_metrics["accuracy"]))
    print("Metrics saved to: {0}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
