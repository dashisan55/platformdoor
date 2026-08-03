"""
特徴量抽出スクリプト（非存在時のファイル生成補強版）

使い方:
  python src/extract_features.py --input data/raw --output data/features
"""

import os
import glob
import argparse
import logging
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/settings.yaml") -> dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_csv_file(filepath: str) -> tuple:
    lines = None
    for encoding in ["cp932", "shift-jis", "utf-8", "utf-8-sig"]:
        try:
            with open(filepath, "r", encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    if lines is None:
        logger.error(f"文字コード判定失敗: {filepath}")
        return None, None

    meta = {}
    data_header_idx = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("#") or "," not in line:
            continue
        parts = line.split(",")
        key = parts[0].strip()
        val = parts[1].strip() if len(parts) > 1 else ""
        if   "駅名"     in key: meta["station"]       = val
        elif "番線"     in key: meta["line"]          = val
        elif "号車番号" in key: meta["car"]           = val
        elif "ドア番号" in key: meta["door"]          = val
        elif "動作日時" in key: meta["datetime"]      = val
        elif "動作種別" in key: meta["operation_raw"] = val
        elif "時間"     in key:
            data_header_idx = i + 1
            break

    if data_header_idx is None:
        return None, None

    data_lines = []
    for line in lines[data_header_idx:]:
        line = line.strip()
        if line.startswith("#") or line == "":
            continue
        data_lines.append(line)

    if len(data_lines) == 0:
        return None, None

    try:
        df = pd.read_csv(StringIO("\n".join(data_lines)), header=None)
    except Exception:
        return None, None

    return meta, df


def parse_excel(filepath: str) -> tuple:
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".xlsx":
        try:
            raw = pd.read_excel(filepath, header=None, engine="openpyxl")
        except Exception as e:
            logger.error(f"ファイル読み込みエラー: {filepath} → {e}")
            return None, None

        meta = {}
        data_start = None

        for i, row in raw.iterrows():
            cell = str(row.iloc[0]) if len(row) > 0 and pd.notna(row.iloc[0]) else ""
            val  = str(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else ""

            if "駅名"       in cell: meta["station"]      = val
            elif "番線"     in cell: meta["line"]          = val
            elif "号車番号"  in cell: meta["car"]           = val
            elif "ドア番号"  in cell: meta["door"]          = val
            elif "動作日時"  in cell: meta["datetime"]      = val
            elif "動作種別"  in cell: meta["operation_raw"] = val
            elif "時間"     in cell and data_start is None:
                data_start = i + 1

        if data_start is None:
            return None, None

        df = raw.iloc[data_start:].reset_index(drop=True)
        df.columns = range(len(df.columns))
        return meta, df

    elif ext == ".csv":
        return parse_csv_file(filepath)

    else:
        return None, None


def extract_features(filepath: str, label: int = 0, original_filename: str = None) -> dict:
    meta, df = parse_excel(filepath)
    if meta is None or df is None:
        return None

    try:
        time_col      = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        current_left  = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        current_right = pd.to_numeric(df.iloc[:, 2], errors='coerce')

        valid      = ~(time_col.isna() | current_left.isna())
        time_arr   = time_col[valid].values
        curr_left  = current_left[valid].values
        curr_right = current_right[valid].values if not current_right.isna().all() else curr_left
        curr_arr   = (curr_left + curr_right) / 2.0

        if len(curr_arr) < 10:
            return None

        filename = original_filename if original_filename else os.path.basename(filepath)
        timestamp_str = filename.split("_")[0]
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        except Exception:
            timestamp = datetime.now()

        op_raw = meta.get("operation_raw", "")
        if "開" in op_raw:
            operation = "open"
        elif "閉" in op_raw:
            operation = "close"
        else:
            if "開動作" in filename:
                operation = "open"
            elif "閉動作" in filename:
                operation = "close"
            else:
                operation = "unknown"

        return {
            "timestamp":          timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "station":            meta.get("station",  "unknown"),
            "line":               meta.get("line",     "unknown"),
            "car":                meta.get("car",      "unknown"),
            "door":               meta.get("door",     "unknown"),
            "operation":          operation,
            "filename":           filename,
            "mean_current":       float(np.mean(curr_arr)),
            "std_current":        float(np.std(curr_arr)),
            "max_current":        float(np.max(curr_arr)),
            "min_current":        float(np.min(curr_arr)),
            "range_current":      float(np.max(curr_arr) - np.min(curr_arr)),
            "p25_current":        float(np.percentile(curr_arr, 25)),
            "p50_current":        float(np.percentile(curr_arr, 50)),
            "p75_current":        float(np.percentile(curr_arr, 75)),
            "duration":           float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 0.0,
            "data_points":        int(len(curr_arr)),
            "mean_diff":          float(np.mean(np.abs(np.diff(curr_arr)))) if len(curr_arr) > 1 else 0.0,
            "max_diff":           float(np.max(np.abs(np.diff(curr_arr)))) if len(curr_arr) > 1 else 0.0,
            "energy":             float(np.sum(curr_arr ** 2)),
            "rms_current":        float(np.sqrt(np.mean(curr_arr ** 2))),
            "skewness":           float(pd.Series(curr_arr).skew()),
            "kurtosis":           float(pd.Series(curr_arr).kurtosis()),
            "mean_current_left":  float(np.mean(curr_left)),
            "mean_current_right": float(np.mean(curr_right)),
            "max_current_left":   float(np.max(curr_left)),
            "max_current_right":  float(np.max(curr_right)),
            "label":              label,
        }
    except Exception as e:
        logger.error(f"特徴量抽出エラー: {filepath} → {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="特徴量抽出")
    parser.add_argument("--input",  default="data/raw",      help="入力フォルダ")
    parser.add_argument("--output", default="data/features", help="出力フォルダ")
    parser.add_argument("--config", default="config/settings.yaml", help="設定ファイル")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    patterns = [
        os.path.join(args.input, "*.xlsx"),
        os.path.join(args.input, "*.csv"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(files)

    logger.info(f"対象ファイル数: {len(files)}")

    all_features = []
    for i, filepath in enumerate(files):
        features = extract_features(filepath)
        if features:
            all_features.append(features)

    df_features = pd.DataFrame(all_features)
    latest_path = os.path.join(args.output, "features_latest.csv")
    df_features.to_csv(latest_path, index=False, encoding="utf-8-sig")
    logger.info(f"特徴量CSV出力完了: {latest_path}（件数: {len(df_features)}）")


if __name__ == "__main__":
    main()
