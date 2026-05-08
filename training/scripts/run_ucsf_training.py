from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb

from scipy import stats

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from common import (
    IMAGING_FEATURES,
    LABEL_MAP,
    LABEL_NAMES,
    MODELS_ROOT,
    TARGET,
    TRAINING_ROOT,
    detect_xgboost_device,
    ensure_directories,
    load_config,
    load_dataset,
    save_json,
    save_markdown,
    xgb_base_params,
)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

xgb.set_config(verbosity=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UCSF risk classification training workflow.")
    parser.add_argument(
        "--config",
        type=Path,
        default=TRAINING_ROOT / "configs" / "training_config.json",
    )
    return parser.parse_args()


def plot_class_distribution(df: pd.DataFrame, figure_path: Path) -> None:
    ax = df[TARGET].value_counts().reindex(LABEL_NAMES).plot(
        kind="bar",
        color=["green", "orange", "red"],
        figsize=(7, 5),
    )
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()


def plot_feature_distributions(X: pd.DataFrame, labels: pd.Series, figure_path: Path) -> None:
    fig, axes = plt.subplots(5, 4, figsize=(18, 16))
    axes = axes.flatten()
    label_strings = labels.map({v: k for k, v in LABEL_MAP.items()})

    for index, feature in enumerate(IMAGING_FEATURES):
        for label in LABEL_NAMES:
            axes[index].hist(
                X.loc[label_strings == label, feature],
                bins=20,
                alpha=0.5,
                label=label,
            )
        axes[index].set_title(feature, fontsize=8)
        axes[index].legend(fontsize=6)

    for index in range(len(IMAGING_FEATURES), len(axes)):
        axes[index].set_visible(False)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(X: pd.DataFrame, figure_path: Path) -> pd.DataFrame:
    corr = X.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
    return corr


def high_correlation_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    upper = corr_matrix.abs().where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    rows: list[dict[str, object]] = []
    for col in upper.columns:
        for row in upper.index:
            value = upper.loc[row, col]
            if pd.notna(value) and value > threshold:
                rows.append({"feature_a": row, "feature_b": col, "absolute_correlation": float(value)})
    if not rows:
        return pd.DataFrame(columns=["feature_a", "feature_b", "absolute_correlation"])
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False)


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    test_size: float,
    validation_fraction_of_trainval: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=validation_fraction_of_trainval,
        random_state=random_state,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_split_assignments(df: pd.DataFrame, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, output_path: Path) -> None:
    split_series = pd.Series(index=df.index, dtype="string")
    split_series.loc[X_train.index] = "train"
    split_series.loc[X_val.index] = "validation"
    split_series.loc[X_test.index] = "test"
    output = df.loc[split_series.dropna().index, ["case_id", "patient_id", TARGET]].copy()
    output["split"] = split_series.dropna().astype(str)
    output.to_csv(output_path, index=False)


def run_baselines(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    output_path: Path,
) -> pd.DataFrame:

    scaled_models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }

    tree_models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            random_state=42,
        ),
    }

    vt = VarianceThreshold()
    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(vt.fit_transform(X_train))
    X_val_scaled = scaler.transform(vt.transform(X_val))

    rows = []

    for name, model in scaled_models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)

        rows.append({
            "model": name,
            "balanced_accuracy": balanced_accuracy_score(y_val, preds),
            "macro_f1": f1_score(y_val, preds, average="macro"),
        })

    for name, model in tree_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rows.append({
            "model": name,
            "balanced_accuracy": balanced_accuracy_score(y_val, preds),
            "macro_f1": f1_score(y_val, preds, average="macro"),
        })

    baseline_df = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "macro_f1"],
        ascending=False,
    )

    baseline_df.to_csv(output_path, index=False)

    return baseline_df

def fit_validation_xgb(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    device: str,
    random_state: int,
    early_stopping_rounds: int,
) -> xgb.XGBClassifier:
    params = xgb_base_params(
        device=device,
        random_state=random_state,
        n_jobs=-1 if device == "cpu" else 1,
        early_stopping_rounds=early_stopping_rounds,
    )
    params.update(
        {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 4,
            "gamma": 0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
    )
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def tune_xgboost(
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    random_state: int,
    n_trials: int,
    n_splits: int,
    device: str,
    early_stopping_rounds: int,
    trials_output_path: Path,
) -> tuple[optuna.Study, dict[str, object]]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = xgb_base_params(
            device=device,
            random_state=random_state,
            n_jobs=-1 if device == "cpu" else 1,
            early_stopping_rounds=early_stopping_rounds,
        )
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 700),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
        )

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_trainval, y_trainval), start=1):
            X_tr = X_trainval.iloc[train_idx]
            X_va = X_trainval.iloc[val_idx]
            y_tr = y_trainval.iloc[train_idx]
            y_va = y_trainval.iloc[val_idx]
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr)
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr,
                y_tr,
                sample_weight=sample_weights,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
            preds = model.predict(X_va)

            score = balanced_accuracy_score(y_va, preds)
            scores.append(score)
            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_output_path, index=False)
    best_params = study.best_params.copy()
    return study, best_params


def fit_best_model(
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    best_params: dict[str, object],
    device: str,
    random_state: int,
) -> xgb.XGBClassifier:
    params = xgb_base_params(
        device=device,
        random_state=random_state,
        n_jobs=-1 if device == "cpu" else 1,
    )
    params.update(best_params)
    model = xgb.XGBClassifier(**params)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_trainval)
    model.fit(X_trainval, y_trainval, sample_weight=sample_weights, verbose=False)
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: Path,
    metrics_dir: Path,
) -> dict[str, object]:
    y_pred = model.predict(X_test)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    y_proba = model.predict_proba(X_test)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report_dict).transpose().to_csv(metrics_dir / "test_classification_report.csv")

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "quadratic_kappa": float(cohen_kappa_score(y_test, y_pred, weights="quadratic")),
        "macro_auc_ovr": float(roc_auc_score(y_test, y_proba[:, 1])),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    save_json(metrics, metrics_dir / "test_metrics.json")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=LABEL_NAMES,
        cmap="Blues",
        ax=ax,
    )
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return metrics


def save_feature_importance(model: xgb.XGBClassifier, figures_dir: Path, tables_dir: Path) -> None:
    importance_df = pd.DataFrame(
        {
            "feature": IMAGING_FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=True)
    importance_df.to_csv(tables_dir / "feature_importance.csv", index=False)

    plt.figure(figsize=(8, 7))
    plt.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=150)
    plt.close()


def save_shap_outputs(model: xgb.XGBClassifier, X_test: pd.DataFrame, figures_dir: Path, logs_dir: Path) -> None:
    error_path = logs_dir / "shap_error.txt"
    try:
        contribs = model.get_booster().predict(xgb.DMatrix(X_test), pred_contribs=True)
        if contribs.ndim == 3:
            high_risk_values = contribs[:, 1, :-1]
        elif contribs.ndim == 2:
            n_features_plus_bias = len(IMAGING_FEATURES) + 1
            n_classes = 2
            reshaped = contribs.reshape(X_test.shape[0], n_classes, n_features_plus_bias)
            high_risk_values = reshaped[:, 1, :-1]
        else:
            raise ValueError(f"Unexpected pred_contribs shape: {contribs.shape}")
        
        # ---- Compute mean absolute SHAP importance ----
        shap_importance = np.abs(high_risk_values).mean(axis=0)

        shap_df = pd.DataFrame({
            "feature": IMAGING_FEATURES,
            "shap_importance": shap_importance
        }).sort_values("shap_importance", ascending=False)

        top3_shap = shap_df.head(3)

        print("\n=== Top 3 SHAP Features (High Risk Class) ===")
        for _, row in top3_shap.iterrows():
            print(f"{row['feature']} -> {row['shap_importance']:.6f}")

        shap.summary_plot(
            high_risk_values,
            X_test,
            feature_names=IMAGING_FEATURES,
            plot_type="bar",
            show=False,
        )
        plt.title("SHAP - high_risk class")
        plt.tight_layout()
        plt.savefig(figures_dir / "shap_high_risk_bar.png", dpi=150)
        plt.close()

        shap.summary_plot(
            high_risk_values,
            X_test,
            feature_names=IMAGING_FEATURES,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(figures_dir / "shap_high_risk_beeswarm.png", dpi=150)
        plt.close()
        if error_path.exists():
            error_path.unlink()
    except Exception as exc:
        error_path.write_text(str(exc), encoding="utf-8")


def final_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, object],
    random_state: int,
    n_splits: int,
    device: str,
    early_stopping_rounds: int,
    output_path: Path,
) -> dict[str, object]:
    params = xgb_base_params(
        device=device,
        random_state=random_state,
        n_jobs=-1 if device == "cpu" else 1,
    )
    params.update(best_params)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    printed_fold = False 
    rows: list[dict[str, float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_tr, y_tr)

        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[-30:]

        X_tr = X_tr.iloc[:, top_idx]
        X_te = X_te.iloc[:, top_idx]
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr)
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=sample_weights, verbose=False)
        proba = model.predict_proba(X_te)
        preds = np.argmax(proba, axis=1)


        if not printed_fold:
            printed_fold = True
            inv_map = {v: k for k, v in LABEL_MAP.items()}
            pred_labels = [inv_map[p] for p in preds[:20]]
            true_labels = [inv_map[t] for t in y_te.values[:20]]
            for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
                match = "✓" if t == p else "✗"

        bal_acc = float(balanced_accuracy_score(y_te, preds))
        auc = float(roc_auc_score(y_te, proba[:, 1]))

        print(f"Fold {fold_idx} -> AUC: {auc:.4f}")
        # print("🐸= ", np.bincount(y_te))

        rows.append(
            {
                "fold": fold_idx,
                "balanced_accuracy": bal_acc,
                "auc": auc,
            }
        )

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(output_path, index=False)
    bal_mean = float(cv_df["balanced_accuracy"].mean())
    auc_mean = float(cv_df["auc"].mean())

    bal_std = float(cv_df["balanced_accuracy"].std())
    auc_std = float(cv_df["auc"].std())
    bal_median = float(cv_df["balanced_accuracy"].median())

    n = len(cv_df)

    # ---- 95% Confidence Interval ----
    bal_ci_low, bal_ci_high = stats.t.interval(
        0.95, df=n-1,
        loc=bal_mean,
        scale=bal_std / np.sqrt(n)
    )

    auc_ci_low, auc_ci_high = stats.t.interval(
        0.95, df=n-1,
        loc=auc_mean,
        scale=auc_std / np.sqrt(n)
    )

    # ---- p-value (test > random baseline = 0.5) ----
    bal_t, bal_p = stats.ttest_1samp(cv_df["balanced_accuracy"], popmean=0.5)
    auc_t, auc_p = stats.ttest_1samp(cv_df["auc"], popmean=0.5)

    print(f"\nMean AUC               : {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"95% CI (AUC)           : [{auc_ci_low:.4f}, {auc_ci_high:.4f}]")
    print(f"p-value (vs 0.5)       : {auc_p:.2e}")

    return {
        "balanced_accuracy_mean": bal_mean,
        "balanced_accuracy_std": bal_std,
        "balanced_accuracy_median": bal_median,
        "balanced_accuracy_ci": [bal_ci_low, bal_ci_high],
        "balanced_accuracy_p": bal_p,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_ci": [auc_ci_low, auc_ci_high],
        "auc_p": auc_p,
    }


def get_feature_summary(X: pd.DataFrame, y: pd.Series) -> dict:
    return {
        "n_samples": int(len(X)),
        "n_features_total": int(X.shape[1]),
        "feature_names": list(X.columns),

        "class_distribution": y.value_counts().to_dict(),

        "feature_mean": X.mean().to_dict(),
        "feature_std": X.std().to_dict(),

        "missing_values": int(X.isnull().sum().sum()),

        "correlation_max": float(X.corr().abs().values[np.triu_indices(X.shape[1], 1)].max())
    }

def save_model_artifacts(model: xgb.XGBClassifier) -> None:
    joblib.dump(model, MODELS_ROOT / "best_xgb_risk_classifier.pkl")
    with (MODELS_ROOT / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump({"map": LABEL_MAP, "names": LABEL_NAMES}, handle, indent=2)
    with (MODELS_ROOT / "feature_list.json").open("w", encoding="utf-8") as handle:
        json.dump({"features": IMAGING_FEATURES}, handle, indent=2)


def build_markdown_report(
    dataset_summary: dict[str, object],
    device_summary: dict[str, object],
    baseline_df: pd.DataFrame,
    validation_metrics: dict[str, object],
    test_metrics: dict[str, object],
    cv_summary: dict[str, object],
    best_params: dict[str, object],
    optuna_best_value: float,
    config: dict[str, object],
) -> str:
    best_baseline = baseline_df.iloc[0].to_dict()
    return f"""# UCSF Training Summary

## Dataset

- Rows after label cleaning: `{dataset_summary['rows_after_null_label_drop']}`
- Imaging features used: `{dataset_summary['n_features']}`
- Class distribution: `{dataset_summary['class_distribution']}`
- Total feature nulls at training input: `{dataset_summary['total_feature_nulls']}`

## Runtime

- Requested accelerator: `cuda`
- Selected device: `{device_summary['selected_device']}`
- GPU available: `{device_summary['gpu_available']}`
- Device note: `{device_summary['reason']}`
- Optuna trials run: `{config['optuna_trials']}`

## Baselines

- Best validation baseline: `{best_baseline['model']}`
- Best baseline balanced accuracy: `{best_baseline['balanced_accuracy']:.4f}`
- Best baseline macro F1: `{best_baseline['macro_f1']:.4f}`

## XGBoost

- Validation balanced accuracy: `{validation_metrics['balanced_accuracy']:.4f}`
- Validation macro F1: `{validation_metrics['macro_f1']:.4f}`
- Best Optuna CV balanced accuracy: `{optuna_best_value:.4f}`
- Best params: `{best_params}`

## Final Test Metrics

- Balanced accuracy: `{test_metrics['balanced_accuracy']:.4f}`
- Macro F1: `{test_metrics['macro_f1']:.4f}`
- Quadratic kappa: `{test_metrics['quadratic_kappa']:.4f}`
- Macro AUC OvR: `{test_metrics['macro_auc_ovr']:.4f}`

## Final Cross-Validation

- Balanced accuracy: `{cv_summary['balanced_accuracy_mean']:.4f}`
- AUC: `{cv_summary['auc_mean']:.4f}`
"""


def main() -> None:
    args = parse_args()
    paths = ensure_directories()
    config = load_config(args.config)


    input_csv = Path(str(config["input_csv"]))
    if not input_csv.is_absolute():
        input_csv = Path.cwd() / input_csv

    random_state = int(config["random_state"])
    early_stopping_rounds = int(config["xgboost_early_stopping_rounds"])

    clean_df, X, y, dataset_summary = load_dataset(input_csv)

    feature_summary = get_feature_summary(X, y)
    save_json(feature_summary, paths["metrics"] / "feature_summary.json")

    if isinstance(y, pd.DataFrame):
        y = y.idxmax(axis=1)
    elif isinstance(y, np.ndarray) and len(y.shape) > 1:
        y = np.argmax(y, axis=1)

    y = pd.Series(y).astype(int)

    save_json(dataset_summary, paths["metrics"] / "dataset_summary.json")

    device_summary = detect_xgboost_device(random_state=random_state)
    save_json(device_summary, paths["logs"] / "gpu_status.json")

    plot_class_distribution(clean_df, paths["figures"] / "class_distribution.png")
    plot_feature_distributions(X, y, paths["figures"] / "feature_distributions.png")
    corr_matrix = plot_correlation_heatmap(X, paths["figures"] / "correlation_heatmap.png")
    high_corr_df = high_correlation_pairs(corr_matrix, threshold=0.9)
    high_corr_df.to_csv(paths["tables"] / "high_correlation_pairs.csv", index=False)
    X.describe().transpose().to_csv(paths["tables"] / "feature_summary.csv")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X=X,
        y=y,
        random_state=random_state,
        test_size=float(config["test_size"]),
        validation_fraction_of_trainval=float(config["validation_fraction_of_trainval"]),
    )
    X_trainval = pd.concat([X_train, X_val]).sort_index()
    y_trainval = pd.concat([y_train, y_val]).sort_index()

    save_split_assignments(clean_df, X_train, X_val, X_test, paths["tables"] / "split_assignments.csv")

    baseline_df = run_baselines(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        output_path=paths["metrics"] / "baseline_results.csv",
    )

    validation_model = fit_validation_xgb(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        device=str(device_summary["selected_device"]),
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )
    val_preds = validation_model.predict(X_val)
    if len(val_preds.shape) > 1:
        val_preds = np.argmax(val_preds, axis=1)
    validation_metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_val, val_preds)),
        "macro_f1": float(f1_score(y_val, val_preds, average="macro")),
    }
    save_json(validation_metrics, paths["metrics"] / "validation_xgboost_metrics.json")

    study, best_params = tune_xgboost(
        X_trainval=X_trainval,
        y_trainval=y_trainval,
        random_state=random_state,
        n_trials=int(config["optuna_trials"]),
        n_splits=int(config["optuna_cv_folds"]),
        device=str(device_summary["selected_device"]),
        early_stopping_rounds=early_stopping_rounds,
        trials_output_path=paths["logs"] / "optuna_trials.csv",
    )
    save_json(best_params, paths["models"] / "best_params.json")

    best_model = fit_best_model(
        X_trainval=X_trainval,
        y_trainval=y_trainval,
        best_params=best_params,
        device=str(device_summary["selected_device"]),
        random_state=random_state,
    )

    test_metrics = evaluate_model(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        figures_dir=paths["figures"],
        metrics_dir=paths["metrics"],
    )
    save_feature_importance(best_model, paths["figures"], paths["tables"])

    save_shap_outputs(best_model, X_test, paths["figures"], paths["logs"])

    cv_summary = final_cross_validation(
        X=X,
        y=y,
        best_params=best_params,
        random_state=random_state,
        n_splits=int(config["final_cv_folds"]),
        device=str(device_summary["selected_device"]),
        early_stopping_rounds=early_stopping_rounds,
        output_path=paths["metrics"] / "final_cv_fold_metrics.csv",
    )
    save_json(cv_summary, paths["metrics"] / "final_cv_summary.json")

    save_model_artifacts(best_model)

    report = build_markdown_report(
        dataset_summary=dataset_summary,
        device_summary=device_summary,
        baseline_df=baseline_df,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        cv_summary=cv_summary,
        best_params=best_params,
        optuna_best_value=float(study.best_value),
        config=config,
    )
    save_markdown(report, paths["reports"] / "training_summary.md")


if __name__ == "__main__":
    main()
