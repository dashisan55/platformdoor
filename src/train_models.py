"""
モデル学習・比較スクリプト（分割CSV全読込対応版）

実行方法:
  python src/train_models.py --features_dir data/features
"""

import os
import glob
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


def get_models(contamination=0.01):
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
                with open(os.path.join(save_dir, f"{model_name}_dist.json"), "w", encoding="utf-8") as f:
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
                with open(os.path.join(save_dir, f"{model_name}_dist.json"), "w", encoding="utf-8") as f:
                    json.dump(dist_info, f, ensure_ascii=False, indent=2)

            except Exception as e:
                logger.error(f"    {model_name} 学習エラー: {e}")

        summary[model_id] = model_results

    logger.info(f"統合モデル学習完了: {len(summary)}グループ")
    return summary


def save_baseline_if_not_exists(df, models_dir):
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
    baseline_path = os.path.join(models_dir, "baseline_stats.json")
    if not os.path.exists(baseline_path):
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

        drift = abs(current_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0.0
        status = "正常" if drift <= 2.0 else "警告" if drift <= 3.0 else "異常"

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
    logger.info(f"ドリフト検出結果出力: {drift_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--models",       default="models")
    parser.add_argument("--reports",      default="reports")
    parser.add_argument("--config",       default="config/settings.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # 分割された全正常CSVを一括読み込み
    csv_files = sorted(glob.glob(os.path.join(args.features_dir, "features_normal_part*.csv")))
    if not csv_files:
        csv_files = [os.path.join(args.features_dir, "features_all.csv")]

    logger.info(f"特徴量ファイル読み込み対象: {len(csv_files)}件")
    df_list = [pd.read_csv(f) for f in csv_files if os.path.exists(f)]
    
    if not df_list:
        logger.error("学習用データが見つかりません。")
        return

    df = pd.concat(df_list, ignore_index=True)
    # 正常データ（label=0）のみで学習
    if 'label' in df.columns:
        df = df[df['label'] == 0]

    logger.info(f"学習用正常データ: {len(df)}件 / {len(df.columns)}列")

    save_baseline_if_not_exists(df, args.models)
    compare_with_baseline(df, args.models, args.reports)

    individual_summary = {}
    unified_summary    = {}

    if config["anomaly_detection"]["individual_model"]:
        individual_summary = train_individual_models(df, config, args.models)

    if config["anomaly_detection"]["unified_model"]:
        unified_summary = train_unified_model(df, config, args.models)

    marker = {
        "trained_at":        datetime.now().isoformat(),
        "n_samples":         len(df),
        "individual_groups": len(individual_summary),
        "unified_groups":    len(unified_summary),
    }
    os.makedirs(args.models, exist_ok=True)
    with open(os.path.join(args.models, "train_info.json"), "w", encoding="utf-8") as f:
        json.dump(marker, f, ensure_ascii=False, indent=2)

    logger.info("=== 学習完了 ===")


if __name__ == "__main__":
    main()
