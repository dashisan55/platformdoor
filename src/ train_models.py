"""
モデル学習・比較スクリプト

実行方法:
  python src/train_models.py --features data/features/features_latest.csv
"""

import os
import json
import logging
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRAIN] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_config(path="config/settings.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


FEATURE_COLS = [
    "mean_current", "std_current", "max_current", "min_current", "range_current",
    "p25_current", "p50_current", "p75_current",
    "duration", "data_points",
    "mean_diff", "max_diff",
    "energy", "rms_current",
    "skewness", "kurtosis",
    "mean_current_left", "mean_current_right",
    "max_current_left", "max_current_right",
]

def get_available_features(df):
    return [c for c in FEATURE_COLS if c in df.columns]


def preprocess(df, feature_cols):
    X = df[feature_cols].copy()
    for col in X.columns:
        med = X[col].median()
        X[col] = X[col].fillna(med)
    for col in X.columns:
        mu, sigma = X[col].mean(), X[col].std()
        if sigma > 0:
            X[col] = X[col].clip(mu - 3*sigma, mu + 3*sigma)
    return X


def get_models(contamination=0.02):
    return {
        "isolation_forest": IsolationForest(
            n_estimators=200, contamination=contamination,
            random_state=42, n_jobs=-1
        ),
        "lof": LocalOutlierFactor(
            n_neighbors=20, contamination=contamination,
            novelty=True, n_jobs=-1
        ),
        "one_class_svm": OneClassSVM(
            kernel="rbf", nu=contamination, gamma="scale"
        ),
    }


def normalize_scores(raw_scores):
    mn, mx = raw_scores.min(), raw_scores.max()
    if mx == mn:
        return np.zeros_like(raw_scores)
    return 1.0 - (raw_scores - mn) / (mx - mn)


def train_individual_models(df, config, models_dir):
    logger.info("=== 個別モデル学習開始 ===")
    contamination = config["anomaly_detection"]["contamination"]
    feature_cols  = get_available_features(df)
    logger.info(f"使用特徴量: {len(feature_cols)}列")

    summary = {}
    group_keys = ["car", "door", "operation"]
    groups = df.groupby(group_keys, dropna=False)

    for key, group_df in groups:
        car, door, operation = key
        model_id = f"car{car}_door{door}_{operation}"

        if len(group_df) < 10:
            logger.warning(f"  スキップ（データ不足）: {model_id} ({len(group_df)}件)")
            continue

        logger.info(f"  学習中: {model_id}  ({len(group_df)}件)")
        X = preprocess(group_df, feature_cols)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_results = {}
        for model_name, model in get_models(contamination).items():
            try:
                model.fit(X_scaled)
                raw_scores = model.decision_function(X_scaled)
                scores = normalize_scores(raw_scores)

                model_results[model_name] = {
                    "model":  model,
                    "scaler": scaler,
                    "score_mean": float(scores.mean()),
                    "score_std":  float(scores.std()),
                    "threshold_warning": float(np.percentile(scores, 95)),
                    "threshold_danger":  float(np.percentile(scores, 99)),
                }

                save_dir = os.path.join(models_dir, "individual", model_id)
                os.makedirs(save_dir, exist_ok=True)
                joblib.dump(model,  os.path.join(save_dir, f"{model_name}.pkl"))
                joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

                dist_info = {
                    "model_id":     model_id,
                    "model_name":   model_name,
                    "n_samples":    len(group_df),
                    "feature_cols": feature_cols,
                    "score_mean":   float(scores.mean()),
                    "score_std":    float(scores.std()),
                    "score_p95":    float(np.percentile(scores, 95)),
                    "score_p99":    float(np.percentile(scores, 99)),
                    "trained_at":   datetime.now().isoformat(),
                }
                with open(os.path.join(save_dir, f"{model_name}_dist.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(dist_info, f, ensure_ascii=False, indent=2)

            except Exception as e:
                logger.error(f"    {model_name} 学習エラー: {e}")

        summary[model_id] = model_results

    logger.info(f"個別モデル学習完了: {len(summary)}グループ")
    return summary


def train_unified_model(df, config, models_dir):
    logger.info("=== 統合モデル学習開始 ===")
    contamination = config["anomaly_detection"]["contamination"]
    feature_cols  = get_available_features(df)

    summary = {}
    for operation in df["operation"].unique():
        action_df = df[df["operation"] == operation]
        model_id  = f"unified_{operation}"
        logger.info(f"  学習中: {model_id}  ({len(action_df)}件)")

        X = preprocess(action_df, feature_cols)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_results = {}
        for model_name, model in get_models(contamination).items():
            try:
                model.fit(X_scaled)
                raw_scores = model.decision_function(X_scaled)
                scores = normalize_scores(raw_scores)

                model_results[model_name] = {
                    "score_mean": float(scores.mean()),
                    "score_std":  float(scores.std()),
                }

                save_dir = os.path.join(models_dir, "unified", model_id)
                os.makedirs(save_dir, exist_ok=True)
                joblib.dump(model,  os.path.join(save_dir, f"{model_name}.pkl"))
                joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

                dist_info = {
                    "model_id":     model_id,
                    "model_name":   model_name,
                    "n_samples":    len(action_df),
                    "feature_cols": feature_cols,
                    "score_mean":   float(scores.mean()),
                    "score_std":    float(scores.std()),
                    "score_p95":    float(np.percentile(scores, 95)),
                    "score_p99":    float(np.percentile(scores, 99)),
                    "trained_at":   datetime.now().isoformat(),
                }
                with open(os.path.join(save_dir, f"{model_name}_dist.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(dist_info, f, ensure_ascii=False, indent=2)

            except Exception as e:
                logger.error(f"    {model_name} 学習エラー: {e}")

        summary[model_id] = model_results

    logger.info(f"統合モデル学習完了: {len(summary)}グループ")
    return summary


def compare_models(individual_summary, unified_summary, reports_dir):
    logger.info("=== モデル比較レポート生成 ===")
    os.makedirs(reports_dir, exist_ok=True)

    rows = []
    model_names = ["isolation_forest", "lof", "one_class_svm"]

    for model_id, results in individual_summary.items():
        for model_name in model_names:
            if model_name in results:
                r = results[model_name]
                rows.append({
                    "type":              "個別モデル",
                    "model_id":          model_id,
                    "algorithm":         model_name,
                    "score_mean":        r["score_mean"],
                    "score_std":         r["score_std"],
                    "threshold_warning": r.get("threshold_warning"),
                    "threshold_danger":  r.get("threshold_danger"),
                })

    for model_id, results in unified_summary.items():
        for model_name in model_names:
            if model_name in results:
                r = results[model_name]
                rows.append({
                    "type":              "統合モデル",
                    "model_id":          model_id,
                    "algorithm":         model_name,
                    "score_mean":        r["score_mean"],
                    "score_std":         r["score_std"],
                    "threshold_warning": None,
                    "threshold_danger":  None,
                })

    df_report = pd.DataFrame(rows)

    csv_path = os.path.join(reports_dir, "model_comparison.csv")
    df_report.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"比較CSV: {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("個別モデル vs 統合モデル　異常スコア分布比較", fontsize=13, fontweight="bold")
    colors = {"個別モデル": "#1d4ed8", "統合モデル": "#dc2626"}

    for i, algo in enumerate(model_names):
        ax = axes[i]
        sub = df_report[df_report["algorithm"] == algo]
        if len(sub) == 0:
            continue
        for mtype, grp in sub.groupby("type"):
            ax.bar(range(len(grp)), grp["score_mean"], yerr=grp["score_std"],
                   label=mtype, alpha=0.7, color=colors.get(mtype, "gray"), capsize=3)
        ax.set_title(algo.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("異常スコア（平均）")
        ax.set_ylim(0, 1)
        ax.axhline(0.40, color="orange", linestyle="--", linewidth=1, label="警告閾値")
        ax.axhline(0.65, color="red",    linestyle="--", linewidth=1, label="異常閾値")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(reports_dir, "model_comparison.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"比較グラフ: {fig_path}")

    recommendation = _recommend_model(df_report)
    summary_json = {
        "generated_at":             datetime.now().isoformat(),
        "individual_model_count":   len(individual_summary),
        "unified_model_count":      len(unified_summary),
        "recommendation":           recommendation,
        "details":                  rows
    }
    json_path = os.path.join(reports_dir, "model_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    logger.info(f"比較JSON: {json_path}")
    return df_report


def _recommend_model(df_report):
    if len(df_report) == 0:
        return "データ不足のため推奨不可"
    best = df_report.loc[df_report["score_std"].idxmin()]
    return (f"推奨: {best['type']} / {best['algorithm']} "
            f"（score_std={best['score_std']:.4f} が最小で最も安定）")


def save_baseline_if_not_exists(df, models_dir):
    """
    ベースラインモデルが存在しない場合のみ保存する
    （初回学習時のデータを基準として保持）
    """
    baseline_path = os.path.join(models_dir, "baseline_stats.json")
    if os.path.exists(baseline_path):
        logger.info("ベースラインは既に存在します。スキップします。")
        return

    feature_cols = get_available_features(df)
    baseline = {
        "created_at": datetime.now().isoformat(),
        "n_samples": len(df),
        "feature_cols": feature_cols,
        "stats": {}
    }

    for col in feature_cols:
        baseline["stats"][col] = {
            "mean": float(df[col].mean()),
            "std":  float(df[col].std()),
            "p25":  float(df[col].quantile(0.25)),
            "p50":  float(df[col].quantile(0.50)),
            "p75":  float(df[col].quantile(0.75)),
            "min":  float(df[col].min()),
            "max":  float(df[col].max()),
        }

    os.makedirs(models_dir, exist_ok=True)
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    logger.info(f"ベースライン保存: {baseline_path}")


def compare_with_baseline(df, models_dir, reports_dir):
    """
    現在のデータとベースラインを比較してドリフトを検出する
    """
    baseline_path = os.path.join(models_dir, "baseline_stats.json")
    if not os.path.exists(baseline_path):
        logger.warning("ベースラインが存在しません。スキップします。")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    feature_cols = get_available_features(df)
    drift_results = []

    for col in feature_cols:
        if col not in baseline["stats"]:
            continue
        b = baseline["stats"][col]
        current_mean = float(df[col].mean())
        baseline_mean = b["mean"]
        baseline_std  = b["std"]

        if baseline_std == 0:
            drift = 0.0
        else:
            drift = abs(current_mean - baseline_mean) / baseline_std

        status = "正常"
        if drift > 3.0:
            status = "異常"
        elif drift > 2.0:
            status = "警告"

        drift_results.append({
            "feature":        col,
            "baseline_mean":  round(baseline_mean, 4),
            "current_mean":   round(current_mean, 4),
            "drift_sigma":    round(drift, 4),
            "status":         status,
        })

    df_drift = pd.DataFrame(drift_results)
    os.makedirs(reports_dir, exist_ok=True)
    drift_path = os.path.join(reports_dir, "baseline_drift.csv")
    df_drift.to_csv(drift_path, index=False, encoding="utf-8-sig")
    logger.info(f"ドリフト検出結果: {drift_path}")

    # 警告・異常があればログに出力
    warnings_df = df_drift[df_drift["status"] != "正常"]
    if len(warnings_df) > 0:
        logger.warning(f"=== ドリフト検出: {len(warnings_df)}件 ===")
        for _, row in warnings_df.iterrows():
            logger.warning(
                f"  [{row['status']}] {row['feature']}: "
                f"baseline={row['baseline_mean']} → "
                f"current={row['current_mean']} "
                f"({row['drift_sigma']:.1f}σ)"
            )
    else:
        logger.info("ドリフト検出: 異常なし")

    return df_drift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features/features_latest.csv")
    parser.add_argument("--models",   default="models")
    parser.add_argument("--reports",  default="reports")
    parser.add_argument("--config",   default="config/settings.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    logger.info(f"特徴量読み込み: {args.features}")
    df = pd.read_csv(args.features)
    logger.info(f"  {len(df)}件 / {len(df.columns)}列")

    # ベースライン保存（初回のみ）
    save_baseline_if_not_exists(df, args.models)

    # ベースラインとの比較（ドリフト検出）
    compare_with_baseline(df, args.models, args.reports)

    individual_summary = {}
    unified_summary    = {}

    if config["anomaly_detection"]["individual_model"]:
        individual_summary = train_individual_models(df, config, args.models)

    if config["anomaly_detection"]["unified_model"]:
        unified_summary = train_unified_model(df, config, args.models)

    compare_models(individual_summary, unified_summary, args.reports)

    marker = {
        "trained_at":        datetime.now().isoformat(),
        "n_samples":         len(df),
        "individual_groups": len(individual_summary),
        "unified_groups":    len(unified_summary),
    }
    with open(os.path.join(args.models, "train_info.json"), "w", encoding="utf-8") as f:
        json.dump(marker, f, ensure_ascii=False, indent=2)

    logger.info("=== 学習完了 ===")


if __name__ == "__main__":
    main()
