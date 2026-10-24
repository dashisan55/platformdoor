def predict_all(features_path, models_dir, output_path, config):
    # ファイルが存在しない、または中身が空（0バイト）の場合は安全に終了
    if not os.path.exists(features_path) or os.path.getsize(features_path) == 0:
        logger.warning(f"特徴量ファイルが存在しないか空です: {features_path}（対象0件として処理）")
        summary = {
            "predicted_at": datetime.now().isoformat(),
            "total":        0,
            "normal":       0,
            "warning":      0,
            "danger":       0,
            "has_anomaly":  False,
            "results":      []
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return []

    try:
        df = pd.read_csv(features_path)
    except Exception as e:
        logger.warning(f"CSV読み込みエラー: {e}（対象0件として処理）")
        summary = {
            "predicted_at": datetime.now().isoformat(),
            "total":        0,
            "normal":       0,
            "warning":      0,
            "danger":       0,
            "has_anomaly":  False,
            "results":      []
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return []

    if len(df) == 0:
        logger.warning("特徴量データが0件です。")
        summary = {
            "predicted_at": datetime.now().isoformat(),
            "total":        0,
            "normal":       0,
            "warning":      0,
            "danger":       0,
            "has_anomaly":  False,
            "results":      []
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return []

    logger.info(f"判定対象: {len(df)}件")

    exclude = [
        "timestamp", "station", "line", "car", "door",
        "operation", "filename", "label"
    ]
    feature_cols = [c for c in df.columns if c not in exclude]

    results = []
    danger_count  = 0
    warning_count = 0

    for i, (_, row) in enumerate(df.iterrows()):
        result = predict_single(row, feature_cols, models_dir, config)
        results.append(result)
        if result["label"] == "danger":
            danger_count += 1
        elif result["label"] == "warning":
            warning_count += 1

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
    return results
