"""
特徴量抽出スクリプト（朝イチ自動選別 & サブフォルダ全探索対応版）

使い方:
  python src/extract_features.py --input data/raw --output data/features
"""

import os
import re
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

# 朝の点検対象時間帯（03:30 〜 06:30）
TARGET_HOUR_START = 3
TARGET_HOUR_END   = 7


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

    data_lines = [l.strip() for l in lines[data_header_idx:] if l.strip() and not l.strip().startswith("#")]
    if not data_lines:
        return None, None

    try:
        df = pd.read_csv(StringIO("\n".join(data_lines)), header=None)
    except Exception:
        return None, None

    return meta, df


def extract_features(filepath: str, label: int = 0) -> dict:
    meta, df = parse_csv_file(filepath)
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

        filename = os.path.basename(filepath)
        timestamp_str = filename.split("_")[0]
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        except Exception:
            timestamp = datetime.now()

        op_raw = meta.get("operation_raw", "")
        if "開" in op_raw:       operation = "open"
        elif "閉" in op_raw:     operation = "close"
        else:
            operation = "open" if "開動作" in filename else "close" if "閉動作" in filename else "unknown"

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


def select_morning_files(all_files):
    """
    大量のファイル群から、ドアごと（全9ドア）の朝イチ（03:30〜06:30）の
    最初の開動作・閉動作だけを自動選別する
    """
    selected = []
    # ファイル名順（時系列順）にソート
    all_files = sorted(all_files, key=lambda f: os.path.basename(f))
    
    # ドア×動作ごとの検出フラグ (car_door_op -> bool)
    found_map = {}

    for fp in all_files:
        fn = os.path.basename(fp)
        m = re.match(r"^\d{8}(\d{2})\d{4}_.*_(\d+)号車_(\d+)番ドア_(開動作|閉動作)", fn)
        if not m:
            # マッチしない場合はそのまま候補とする
            selected.append(fp)
            continue

        hour = int(m.group(1))
        car  = m.group(2)
        door = m.group(3)
        op   = m.group(4)

        # 朝の点検時間帯かチェック
        if TARGET_HOUR_START <= hour < TARGET_HOUR_END:
            key = f"{car}_{door}_{op}"
            if key not in found_map:
                selected.append(fp)
                found_map[key] = True
                logger.info(f"  [朝イチ選別] {car}号車{door}番 ({op}): {fn}")

    # もし朝の時間帯で見つからなかった場合（昼間のテスト等）は、全ファイルを対象とする
    if not selected:
        logger.warning("朝の点検時間帯（03:30〜06:30）のデータが見当たらないため、全ファイルを判定します。")
        return all_files

    return selected


def main():
    parser = argparse.ArgumentParser(description="特徴量抽出")
    parser.add_argument("--input",  default="data/raw",      help="入力フォルダ")
    parser.add_argument("--output", default="data/features", help="出力フォルダ")
    parser.add_argument("--config", default="config/settings.yaml", help="設定ファイル")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # rawフォルダ内のすべてのサブフォルダから全CSVを再帰的に検索
    all_files = sorted(glob.glob(os.path.join(args.input, "**", "*.csv"), recursive=True) +
                       glob.glob(os.path.join(args.input, "*.csv")))
    all_files = sorted(list(set(all_files)))

    logger.info(f"検出されたCSV総数: {len(all_files)}")

    if len(all_files) == 0:
        logger.warning("対象ファイルが見つかりません（0件）")
        df_empty = pd.DataFrame()
        df_empty.to_csv(os.path.join(args.output, "features_latest.csv"), index=False)
        return

    # ★ 大量ファイルの中から「朝イチの開閉（計18ファイル等）」だけを自動選別！
    target_files = select_morning_files(all_files)
    logger.info(f"自動選別された朝イチ点検データ数: {len(target_files)}")

    all_features = []
    for filepath in target_files:
        features = extract_features(filepath)
        if features:
            all_features.append(features)

    df_features = pd.DataFrame(all_features)
    latest_path = os.path.join(args.output, "features_latest.csv")
    df_features.to_csv(latest_path, index=False, encoding="utf-8-sig")
    logger.info(f"特徴量CSV出力完了: {latest_path}（件数: {len(df_features)}）")


if __name__ == "__main__":
    main()
