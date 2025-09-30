import os
import requests
import pandas as pd
from config import OUT_DIR, BOT_TOKEN, CHAT_ID

def save_results_csv(df, filename):
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path)
    return path

def notify_telegram(message: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        r = requests.post(url, data=payload, timeout=10)
        print("Telegram:", r.text)
    except Exception as e:
        print("Telegram error:", e)
