"""
Telegram Bot Notifier and Interactive Command Handler.
Allows real-time notifications and control via Telegram:
  /status       - Show current market and portfolio status
  /watchlist    - List active tickers
  /pin <TICKER> - Pin a ticker permanently
  /ban <TICKER> - Ban a ticker permanently
  /portfolio    - Show holdings and PnL
"""
from __future__ import annotations

import os
from typing import Optional

import requests

from src.utils.logger import logger


class TelegramNotifier:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else None

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.base_url or not self.chat_id:
            logger.debug(f"[Telegram] Credentials missing; message not sent: {text[:50]}...")
            return False

        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("[Telegram] Message delivered successfully.")
            return True
        except Exception as exc:
            logger.error(f"[Telegram] Failed to send message: {exc}")
            return False
