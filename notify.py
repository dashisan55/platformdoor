"""
通知スクリプト（メール・Teams）

使い方:
  python src/notify.py --event morning_check \
                       --results data/results/results_latest.json
  python src/notify.py --event retrain_complete \
                       --report  reports/model_comparison.json
"""

import os
import json
import logging
import argparse
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [NOTIFY] %(message)s"
)
logger = logging.getLogger(__name__)


def send_email(subject: str, body_html: str, body_text: str = ""):
    """Gmail SMTPでメール送信"""
    mail_from     = os.environ.get("MAIL_FROM", "")
    mail_to       = os.environ.get("MAIL_TO", "")
    mail_password = os.environ.get("MAIL_PASSWORD", "")
    smtp_server   = "smtp.gmail.com"
    smtp_port     = 587

    if not all([mail_from, mail_to, mail_password]):
        logger.warning("メール設定が不完全です（環境変数を確認してください）")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = mail_from
    msg["To"]      = mail_to

    if body_text:
        msg.attach(MIMEText(body_text, "plain", "utf-8"))
    msg.attach(MIMEText(body_html, "html", "utf-8"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(mail_from, mail_password)
            server.sendmail(mail_from, mail_to.split(","), msg.as_string())
        logger.info(f"メール送信成功: {mail_to}")
        return True
    except Exception as e:
        logger.error(f"メール送信エラー: {e}")
        return False


def send_teams(title: str, body: str, color: str = "0076D7", facts: list = None):
    """Teams Incoming Webhookで通知"""
    webhook_url = os.environ.get("TEAMS_WEBHOOK_URL", "")
    if not webhook_url:
        logger.warning("Teams Webhook URLが設定されていません")
        return False

    card = {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type":    "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {
                        "type":   "TextBlock",
                        "text":   title,
                        "size":   "Large",
                        "weight": "Bolder",
                        "color":  "Attention" if color == "FF0000" else "Default"
                    },
                    {
                        "type": "TextBlock",
                        "text": body,
                        "wrap": True
                    }
                ] + ([{
                    "type":  "FactSet",
                    "facts": [{"title": f["title"], "value": f["value"]}
                               for f in (facts or [])]
                }] if facts else [])
            }
        }]
    }

    try:
        resp = requests.post(
            webhook_url,
            json=card,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if resp.status_code in [200, 202]:
            logger.info("Teams通知成功")
            return True
        else:
            logger.error(f"Teams通知失敗: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Teams通知エラー: {e}")
        return False


def notify_morning_check(results_path: str):
    """朝の点検結果をメール・Teamsで通知"""
    if not os.path.exists(results_path):
        logger.error(f"結果ファイルが見つかりません: {results_path}")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total       = data.get("total",   0)
    normal      = data.get("normal",  0)
    warning     = data.get("warning", 0)
    danger      = data.get("danger",  0)
    has_anomaly = data.get("has_anomaly", False)
    now         = datetime.now().strftime("%Y年%m月%d日 %H:%M")

    anomaly_items = [r for r in data.get("results", []) if r.get("label") == "danger"]
    warning_items = [r for r in data.get("results", []) if r.get("label") == "warning"]

    status_color = "#dc2626" if has_anomaly else "#16a34a"
    status_text  = "⚠️ 異常の予兆が検出されました" if has_anomaly else "✅ 異常なし（正常範囲内）"

    anomaly_rows = ""
    for item in anomaly_items:
        anomaly_rows += f"""
        <tr style="background:#fff5f5;">
          <td style="padding:6px 8px;">{item.get('datetime','')}</td>
          <td style="padding:6px 8px;">{item.get('car','')}号車</td>
          <td style="padding:6px 8px;">{item.get('door','')}番ドア</td>
          <td style="padding:6px 8px;">{item.get('side','')}</td>
          <td style="padding:6px 8px;">{item.get('action','')}</td>
          <td style="padding:6px 8px; color:#dc2626; font-weight:bold;">
            {item.get('score', 0):.3f}
          </td>
        </tr>"""

    warning_rows = ""
    for item in warning_items:
        warning_rows += f"""
        <tr style="background:#fefce8;">
          <td style="padding:6px 8px;">{item.get('datetime','')}</td>
          <td style="padding:6px 8px;">{item.get('car','')}号車</td>
          <td style="padding:6px 8px;">{item.get('door','')}番ドア</td>
          <td style="padding:6px 8px;">{item.get('side','')}</td>
          <td style="padding:6px 8px;">{item.get('action','')}</td>
          <td style="padding:6px 8px; color:#ca8a04; font-weight:bold;">
            {item.get('score', 0):.3f}
          </td>
        </tr>"""

    body_html = f"""
    <html><body style="font-family:Meiryo,sans-serif; color:#2c3e50;">
      <div style="max-width:700px; margin:0 auto;">
        <div style="background:#1a4a8a; color:white; padding:16px 20px; border-radius:8px 8px 0 0;">
          <h2 style="margin:0;">🚉 ホームドア異常予兆検知システム</h2>
          <p style="margin:4px 0 0; font-size:13px;">朝の点検結果レポート　{now}</p>
        </div>
        <div style="border:1px solid #d0d9e6; border-top:none; padding:20px; border-radius:0 0 8px 8px;">
          <div style="background:{status_color}; color:white; padding:12px 16px; border-radius:6px;
                      font-size:16px; font-weight:bold; margin-bottom:16px;">
            {status_text}
          </div>
          <table style="width:100%; border-collapse:collapse; margin-bottom:16px;">
            <tr>
              <td style="background:#f0f4f8; padding:12px; border-radius:6px; text-align:center; width:25%;">
                <div style="font-size:24px; font-weight:bold; color:#2c3e50;">{total}</div>
                <div style="font-size:12px; color:#6b7c93;">総判定数</div>
              </td>
              <td style="width:2%;"></td>
              <td style="background:#dcfce7; padding:12px; border-radius:6px; text-align:center; width:25%;">
                <div style="font-size:24px; font-weight:bold; color:#16a34a;">{normal}</div>
                <div style="font-size:12px; color:#16a34a;">正常</div>
              </td>
              <td style="width:2%;"></td>
              <td style="background:#fef9c3; padding:12px; border-radius:6px; text-align:center; width:25%;">
                <div style="font-size:24px; font-weight:bold; color:#ca8a04;">{warning}</div>
                <div style="font-size:12px; color:#ca8a04;">要注意</div>
              </td>
              <td style="width:2%;"></td>
              <td style="background:#fee2e2; padding:12px; border-radius:6px; text-align:center; width:25%;">
                <div style="font-size:24px; font-weight:bold; color:#dc2626;">{danger}</div>
                <div style="font-size:12px; color:#dc2626;">異常予兆</div>
              </td>
            </tr>
          </table>
          {"<h3 style='color:#dc2626;'>⚠️ 異常検知ドア一覧</h3><table style='width:100%;border-collapse:collapse;font-size:13px;'><tr style='background:#1a4a8a;color:white;'><th style='padding:6px 8px;'>日時</th><th>号車</th><th>ドア</th><th>左右</th><th>動作</th><th>スコア</th></tr>" + anomaly_rows + "</table>" if anomaly_rows else ""}
          {"<h3 style='color:#ca8a04;'>△ 要注意ドア一覧</h3><table style='width:100%;border-collapse:collapse;font-size:13px;'><tr style='background:#1a4a8a;color:white;'><th style='padding:6px 8px;'>日時</th><th>号車</th><th>ドア</th><th>左右</th><th>動作</th><th>スコア</th></tr>" + warning_rows + "</table>" if warning_rows else ""}
          <p style="font-size:11px; color:#6b7c93; margin-top:16px; border-top:1px solid #e5e7eb; padding-top:12px;">
            このメールはホームドア異常予兆検知システムが自動送信しました。
          </p>
        </div>
      </div>
    </body></html>
    """

    subject = (
        f"【異常予兆あり】ホームドア点検結果 {now}" if has_anomaly
        else f"【正常】ホームドア点検結果 {now}"
    )
    send_email(subject, body_html)

    facts = [
        {"title": "総判定数", "value": str(total)},
        {"title": "正常",     "value": str(normal)},
        {"title": "要注意",   "value": str(warning)},
        {"title": "異常予兆", "value": str(danger)},
    ]
    if anomaly_items:
        for item in anomaly_items[:3]:
            facts.append({
                "title": f"⚠️ {item.get('car')}号車{item.get('door')}番ドア",
                "value": f"スコア: {item.get('score', 0):.3f}"
            })

    send_teams(
        f"⚠️ ホームドア異常予兆検知 - {now}" if has_anomaly else f"✅ ホームドア点検完了（正常） - {now}",
        status_text,
        "FF0000" if has_anomaly else "00AA00",
        facts
    )


def notify_retrain_complete(report_path: str):
    """再学習完了をメール・Teamsで通知"""
    report = {}
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

    now            = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    recommendation = report.get("recommendation", "情報なし")

    body_html = f"""
    <html><body style="font-family:Meiryo,sans-serif; color:#2c3e50;">
      <div style="max-width:600px; margin:0 auto;">
        <div style="background:#1a4a8a; color:white; padding:16px 20px; border-radius:8px 8px 0 0;">
          <h2 style="margin:0;">🤖 モデル自動再学習 完了</h2>
          <p style="margin:4px 0 0; font-size:13px;">{now}</p>
        </div>
        <div style="border:1px solid #d0d9e6; border-top:none; padding:20px; border-radius:0 0 8px 8px;">
          <div style="background:#dcfce7; padding:12px; border-radius:6px;
                      margin-bottom:16px; font-weight:bold; color:#16a34a;">
            ✅ 学習が正常に完了しました
          </div>
          <table style="width:100%; border-collapse:collapse; font-size:13px;">
            <tr>
              <td style="padding:6px; color:#6b7c93;">個別モデル数</td>
              <td style="padding:6px; font-weight:bold;">{report.get('individual_model_count', '-')}グループ</td>
            </tr>
            <tr style="background:#f8f9fa;">
              <td style="padding:6px; color:#6b7c93;">統合モデル数</td>
              <td style="padding:6px; font-weight:bold;">{report.get('unified_model_count', '-')}グループ</td>
            </tr>
            <tr>
              <td style="padding:6px; color:#6b7c93;">推奨モデル</td>
              <td style="padding:6px; font-weight:bold; color:#1a4a8a;">{recommendation}</td>
            </tr>
          </table>
        </div>
      </div>
    </body></html>
    """

    send_email(f"【完了】ホームドア モデル再学習 {now}", body_html)
    send_teams(
        f"🤖 モデル再学習完了 - {now}",
        f"推奨モデル: {recommendation}",
        facts=[
            {"title": "個別モデル数", "value": str(report.get("individual_model_count", "-"))},
            {"title": "統合モデル数", "value": str(report.get("unified_model_count", "-"))},
            {"title": "推奨",        "value": recommendation},
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event",   required=True,
                        choices=["morning_check", "retrain_complete"])
    parser.add_argument("--results", default="data/results/results_latest.json")
    parser.add_argument("--report",  default="reports/model_comparison.json")
    args = parser.parse_args()

    if args.event == "morning_check":
        notify_morning_check(args.results)
    elif args.event == "retrain_complete":
        notify_retrain_complete(args.report)


if __name__ == "__main__":
    main()
