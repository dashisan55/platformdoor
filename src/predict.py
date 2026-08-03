"""
異常判定スクリプト

使い方:
  python src/predict.py \
    --features data/features/features_latest.csv \
    --models   models \
    --output   data/results/results_latest.json
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PREDICT] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_config(path="config/settings.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_scores(raw_scores):
    mn, mx = raw_scores.min(), raw_scores.max()
    if mx == mn:
        return np.zeros_like(raw_scores)
    return 1.0 - (raw_scores - mn) / (mx - mn)


def get_label(score, config):
    th = config["thresholds"]
    if score >= th["danger"]:
        return "danger"
    elif score >= th["warning"]:
        return "warning"
    return "normal"


def load_model_and_scaler(model_dir, model_name):
    model_path  = os.path.join(model_dir, f"{model_name}.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    try:
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        logger.error(f"モデルロードエラー: {model_dir} → {e}")
        return None, None


def load_dist_info(model_dir, model_name):
    path = os.path.join(model_dir, f"{model_name}_dist.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _map_score(raw_score, dist):
    mean = dist.get("score_mean", 0.0)
    std  = dist.get("score_std",  1.0)
    if std == 0:
        return 0.0
    z = abs(raw_score - mean) / std
    score = 1.0 / (1.0 + np.exp(-0.5 * (z - 2)))
    return float(min(1.0, max(0.0, score)))


def predict_single(row, feature_cols, model_dir, config):
    car       = str(row.get("car",       "1"))
    door      = str(row.get("door",      "1"))
    operation = str(row.get("operation", "open"))

    x = row[feature_cols].values.reshape(1, -1)
    x = np.nan_to_num(x, nan=0.0)

    results_by_model = {}
    model_names = ["isolation_forest", "lof", "one_class_svm"]

    # 個別モデルで判定
    individual_dir = os.path.join(
        model_dir, "individual",
        f"car{car}_door{door}_{operation}"
    )
    for mn in model_names:
        model, scaler = load_model_and_scaler(individual_dir, mn)
        dist = load_dist_info(individual_dir, mn)
        if model is None:
            results_by_model[f"individual_{mn}"] = {
                "score": None, "label": "unknown", "source": "individual"
            }
            continue
        x_scaled = scaler.transform(x)
        raw = model.decision_function(x_scaled)[0]
        normalized_score = _map_score(raw, dist) if dist else float(max(0.0, min(1.0, -raw * 0.5 + 0.5)))
        results_by_model[f"individual_{mn}"] = {
            "score":  round(float(normalized_score), 4),
            "label":  get_label(normalized_score, config),
            "source": "individual"
        }

    # 統合モデルで判定
    unified_dir = os.path.join(model_dir, "unified", f"unified_{operation}")
    for mn in model_names:
        model, scaler = load_model_and_scaler(unified_dir, mn)
        dist = load_dist_info(unified_dir, mn)
        if model is None:
            results_by_model[f"unified_{mn}"] = {
                "score": None, "label": "unknown", "source": "unified"
            }
            continue
        x_scaled = scaler.transform(x)
        raw = model.decision_function(x_scaled)[0]
        normalized_score = _map_score(raw, dist) if dist else float(max(0.0, min(1.0, -raw * 0.5 + 0.5)))
        results_by_model[f"unified_{mn}"] = {
            "score":  round(float(normalized_score), 4),
            "label":  get_label(normalized_score, config),
            "source": "unified"
        }

    # アンサンブルスコア（有効なスコアの平均）
    valid_scores = [v["score"] for v in results_by_model.values() if v["score"] is not None]
    ensemble_score = float(np.mean(valid_scores)) if valid_scores else 0.0
    ensemble_label = get_label(ensemble_score, config)

    # 動作種別の日本語表示
    action_jp = "開動作" if operation == "open" else "閉動作" if operation == "close" else operation

    return {
        "id":            f"{row.get('timestamp','')}_{car}_{door}_{operation}",
        "datetime":      str(row.get("timestamp", "")),
        "car":           car,
        "door":          door,
        "operation":     operation,
        "action":        action_jp,
        "score":         round(ensemble_score, 4),
        "label":         ensemble_label,
        "model_details": results_by_model,
        "features": {
            col: round(float(row[col]), 4)
            for col in feature_cols
            if col in row.index and not np.isnan(float(row[col]))
        }
    }


def predict_all(features_path, models_dir, output_path, config):
    df = pd.read_csv(features_path)
    logger.info(f"判定対象: {len(df)}件")

    # 除外列（メタデータ列）
    exclude = [
        "timestamp", "station", "line", "car", "door",
        "operation", "filename", "label"
    ]
    feature_cols = [c for c in df.columns if c not in exclude]
    logger.info(f"使用特徴量: {len(feature_cols)}列")

    results = []
    danger_count  = 0
    warning_count = 0

    for i, (_, row) in enumerate(df.iterrows()):
        result = predict_single(row, feature_cols, models_dir, config)
        results.append(result)
        if result["label"] == "danger":
            danger_count += 1
            logger.warning(f"  ⚠️  異常検知: {result['id']}  score={result['score']:.3f}")
        elif result["label"] == "warning":
            warning_count += 1
            logger.info(f"  △ 要注意: {result['id']}  score={result['score']:.3f}")
        if (i + 1) % 50 == 0:
            logger.info(f"  判定済み: {i+1}/{len(df)}")

    summary = {
        "predicted_at": datetime.now().isoformat(),
        "total":        len(results),
        "normal":       len(results) - danger_count - warning_count,
        "warning":      warning_count,
        "danger":       danger_count,
        "has_anomaly":  danger_count > 0,
        "results":      results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"判定完了 → 正常:{summary['normal']} 注意:{warning_count} 異常:{danger_count}")
    logger.info(f"結果保存: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features/features_latest.csv")
    parser.add_argument("--models",   default="models")
    parser.add_argument("--output",   default="data/results/results_latest.json")
    parser.add_argument("--config",   default="config/settings.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    predict_all(args.features, args.models, args.output, config)


if __name__ == "__main__":
    main()
