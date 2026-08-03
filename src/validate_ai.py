"""
検証AI：スクリプトの自動デバッグ・品質チェック
他のスクリプト実行後に呼び出してログ・出力を検証する
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VALIDATE-AI] %(message)s"
)
logger = logging.getLogger(__name__)


class ValidateAI:
    """自動検証・デバッグクラス"""

    def __init__(self, report_dir: str = "reports"):
        self.report_dir = report_dir
        self.results = []
        os.makedirs(report_dir, exist_ok=True)

    def check(self, name: str, condition: bool,
              detail: str = "", level: str = "ERROR"):
        status = "PASS" if condition else level
        self.results.append({
            "check": name, "status": status, "detail": detail
        })
        if condition:
            logger.info(f"  PASS: {name}")
        else:
            fn = logger.error if level == "ERROR" else logger.warning
            fn(f"  {level}: {name} -> {detail}")
        return condition

    def validate_features(self, csv_path: str) -> bool:
        logger.info("=== 特徴量CSV検証 ===")
        ok = True

        if not self.check("特徴量CSVが存在する", os.path.exists(csv_path),
                          f"{csv_path} が見つかりません"):
            return False

        df = pd.read_csv(csv_path)

        ok &= self.check("行数が1以上", len(df) > 0, f"行数={len(df)}")
        ok &= self.check("必須列が存在する",
                         all(c in df.columns for c in
                             ["car","door","action","side","datetime"]),
                         str(df.columns.tolist()))

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        nan_ratio = df[numeric_cols].isna().mean().mean()
        ok &= self.check("数値列の欠損率が10%以下", nan_ratio <= 0.10,
                         f"欠損率={nan_ratio:.1%}", level="WARN")

        if "action" in df.columns:
            actions = df["action"].unique().tolist()
            ok &= self.check("開動作・閉動作の両方が存在",
                             "開動作" in actions and "閉動作" in actions,
                             f"実際の動作種別={actions}", level="WARN")

        logger.info(f"特徴量CSV検証完了: {'OK' if ok else 'NG'}")
        return ok

    def validate_models(self, models_dir: str) -> bool:
        logger.info("=== モデルファイル検証 ===")
        ok = True
        import glob

        model_files = glob.glob(os.path.join(models_dir, "**", "*.pkl"),
                                recursive=True)
        ok &= self.check("モデルファイルが1件以上存在",
                         len(model_files) > 0,
                         f"検索パス={models_dir}")

        for mf in model_files:
            size = os.path.getsize(mf)
            ok &= self.check(f"モデルファイルが空でない: {os.path.basename(mf)}",
                             size > 100,
                             f"ファイルサイズ={size}bytes")

        has_individual = any("individual" in f for f in model_files)
        has_unified    = any("unified"    in f for f in model_files)
        self.check("個別モデルが存在", has_individual, level="WARN")
        self.check("統合モデルが存在", has_unified,    level="WARN")

        return ok

    def validate_results(self, results_path: str) -> bool:
        logger.info("=== 判定結果検証 ===")
        ok = True

        if not self.check("判定結果JSONが存在", os.path.exists(results_path),
                          results_path):
            return False

        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", [])
        ok &= self.check("判定結果が空でない", len(results) > 0)

        for r in results:
            ok &= self.check(
                f"スコアが0-1の範囲: {r.get('id','?')}",
                0.0 <= r.get("score", -1) <= 1.0,
                f"score={r.get('score')}"
            )
            ok &= self.check(
                f"判定ラベルが正しい: {r.get('id','?')}",
                r.get("label") in ["normal", "warning", "danger"],
                f"label={r.get('label')}"
            )

        danger_rate = sum(1 for r in results
                          if r.get("label") == "danger") / max(len(results), 1)
        self.check("異常率が50%以下", danger_rate <= 0.50,
                   f"異常率={danger_rate:.1%}", level="WARN")

        return ok

    def validate_notification_config(self) -> bool:
        logger.info("=== 通知設定検証 ===")
        mail_vars = ["MAIL_FROM", "MAIL_TO", "MAIL_PASSWORD"]
        for var in mail_vars:
            exists = var in os.environ and os.environ[var] != ""
            self.check(f"環境変数 {var} が設定済み", exists, level="WARN")

        teams_ok = "TEAMS_WEBHOOK_URL" in os.environ
        self.check("Teams Webhook URL が設定済み", teams_ok, level="WARN")
        return True

    def save_report(self) -> str:
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        total  = len(self.results)
        errors = [r for r in self.results if r["status"] == "ERROR"]
        warns  = [r for r in self.results if r["status"] == "WARN"]

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total, "passed": passed,
                "errors": len(errors), "warnings": len(warns)
            },
            "details": self.results
        }

        path = os.path.join(
            self.report_dir,
            f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        latest = os.path.join(self.report_dir, "validation_latest.json")
        with open(latest, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*50}")
        logger.info(f"検証完了: {passed}/{total} PASS  "
                    f"ERROR:{len(errors)}  WARN:{len(warns)}")
        logger.info(f"レポート: {path}")
        logger.info(f"{'='*50}")

        if errors:
            logger.error("=== 要対応エラー ===")
            for e in errors:
                logger.error(f"  * {e['check']}: {e['detail']}")

        return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="検証AI")
    parser.add_argument("--features", default="data/features/features_latest.csv")
    parser.add_argument("--models",   default="models")
    parser.add_argument("--results",  default="data/results/results_latest.json")
    parser.add_argument("--reports",  default="reports")
    args = parser.parse_args()

    ai = ValidateAI(report_dir=args.reports)
    ai.validate_features(args.features)
    ai.validate_models(args.models)
    if os.path.exists(args.results):
        ai.validate_results(args.results)
    ai.validate_notification_config()
    ai.save_report()


if __name__ == "__main__":
    main()
