"""
モデル学習・比較スクリプト（即時モデル生成・全データ対応版）
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

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRAIN] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_config(path="config/settings.yaml"):
    if not os.path.exists(path):
        return {"anomaly_detection": {"individual_model": True, "unified_model": True, "contamination": 0.01}}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


FEATURE_COLS = [
    "mean_current", "std_current", "max_current", "min_current", "range_current",
    "p25_current", "p50_current", "p75_current", "mean_diff", "max_diff",
    "energy", "rms_current", "skewness", "kurtosis",
    "mean_current_left", "mean_current_right", "max_current_left", "max_current_right",
]


def get_available_features(df):
    return [c for c in FEATURE_COLS if c in df.columns]


def preprocess(df, feature_cols):
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        med = X[col].median()
        X[col] = X[col].fillna(0.0 if np.isnan(med) else med)
    return X


def get_models(contamination=0.01):
    return {
        "isolation_forest": IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1),
        "lof": LocalOutlierFactor(n_neighbors=min(5, 20), contamination=contamination, novelty=True, n_jobs=-1),
        "one_class_svm": OneClassSVM(kernel="rbf", nu=contamination, gamma="scale"),
    }


def normalize_scores(raw_scores):
    mn, mx = raw_scores.min(), raw_scores.max()
    if mx == mn:
        return np.zeros_like(raw_scores)
    return 1.0 - (raw_scores - mn) / (mx - mn)


def train_individual_models(df, config, models_dir):
    logger.info("=== 個別モデル学習開始 ===")
    contamination = config.get("anomaly_detection", {}).get("contamination", 0.01)
    feature_cols  = get_available_features(df)

    summary = {}
    group_keys = ["car", "door", "operation"]
    groups = df.groupby(group_keys, dropna=False)

    for key, group_df in groups:
        car, door, operation = key
        model_id = f"car{car}_door{door}_{operation}"

        logger.info(f"  学習中: {model_id} ({len(group_df)}件)")
        X = preprocess(group_df, feature_cols)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_results = {}
        for model_name, model in get_models(contamination).items():
            try:
                # LOFのn_neighbors調整
                if model_name == "lof":
                    n_samples = len(X_scaled)
                    model.set_params(n_neighbors=max(1, min(5, n_samples - 1)))

                model.fit(X_scaled)
                raw_scores = model.decision_function(X_scaled)
                scores = normalize_scores(raw_scores)

                model_results[model_name] = {
                    "score_mean": float(scores.mean()),
                    "score_std":  float(scores.std()) if len(scores) > 1 else 0.1,
                }

                save_dir = os.path.join(models_dir, "individual", model_id)
                os.makedirs(save_dir, exist_ok=True)
                joblib.dump(model,  os.path.join(save_dir, f"{model_name}.pkl"))
                joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

                dist_info = {
                    "model_id":     model_id,
                    "model_name":   model_name,
                    "score_mean":   float(scores.mean()),
                    "score_std":    float(scores.std()) if len(scores) > 1 else 0.1,
                    "trained_at":   datetime.now().isoformat(),
                }
                with open(os.path.join(save_dir, f"{model_name}_dist.json"), "w", encoding="utf-8") as f:
                    json.dump(dist_info, f, ensure_ascii=False, indent=2)

            except Exception as e:
                logger.error(f"    {model_name} エラー: {e}")

        summary[model_id] = model_results

    return summary


def train_unified_model(df, config, models_dir):
    logger.info("=== 統合モデル学習開始 ===")
    contamination = config.get("anomaly_detection", {}).get("contamination", 0.01)
    feature_cols  = get_available_features(df)

    summary = {}
    for operation in df["operation"].unique():
        action_df = df[df["operation"] == operation]
        model_id  = f"unified_{operation}"
        logger.info(f"  学習中: {model_id} ({len(action_df)}件)")

        X = preprocess(action_df, feature_cols)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model_results = {}
        for model_name, model in get_models(contamination).items():
            try:
                if model_name == "lof":
                    n_samples = len(X_scaled)
                    model.set_params(n_neighbors=max(1, min(10, n_samples - 1)))

                model.fit(X_scaled)
                raw_scores = model.decision_function(X_scaled)
                scores = normalize_scores(raw_scores)

                model_results[model_name] = {
                    "score_mean": float(scores.mean()),
                    "score_std":  float(scores.std()) if len(scores) > 1 else 0.1,
                }

                save_dir = os.path.join(models_dir, "unified", model_id)
                os.makedirs(save_dir, exist_ok=True)
                joblib.dump(model,  os.path.join(save_dir, f"{model_name}.pkl"))
                joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

                dist_info = {
                    "model_id":     model_id,
                    "model_name":   model_name,
                    "score_mean":   float(scores.mean()),
                    "score_std":    float(scores.std()) if len(scores) > 1 else 0.1,
                    "trained_at":   datetime.now().isoformat(),
                }
                with open(os.path.join(save_dir, f"{model_name}_dist.json"), "w", encoding="utf-8") as f:
                    json.dump(dist_info, f, ensure_ascii=False, indent=2)

            except Exception as e:
                logger.error(f"    {model_name} エラー: {e}")

        summary[model_id] = model_results

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--models",       default="models")
    parser.add_argument("--reports",      default="reports")
    parser.add_argument("--config",       default="config/settings.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    csv_files = sorted(glob.glob(os.path.join(args.features_dir, "*.csv")))
    csv_files = [f for f in csv_files if "fault" not in os.path.basename(f)]
    
    df_list = [pd.read_csv(f) for f in csv_files if os.path.exists(f) and os.path.getsize(f) > 0]
    if not df_list:
        logger.error("データがありません。")
        return

    df = pd.concat(df_list, ignore_index=True)
    if 'label' in df.columns:
        df = df[df['label'] == 0]

    logger.info(f"学習データ: {len(df)}件")

    os.makedirs(args.models, exist_ok=True)
    train_individual_models(df, config, args.models)
    train_unified_model(df, config, args.models)

    logger.info("=== 全モデルの保存完了 ===")


if __name__ == "__main__":
    main()
