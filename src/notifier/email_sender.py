"""
Gmail Email Dispatcher using SMTP SSL/TLS.
Sends rich HTML daily summaries, weekly reviews, and monthly advisory pick reports.
"""
from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from src.utils.logger import logger


class EmailNotifier:
    def __init__(
        self,
        sender: Optional[str] = None,
        password: Optional[str] = None,
        recipient: Optional[str] = None,
    ):
        self.sender = sender or os.getenv("GMAIL_SENDER")
        self.password = password or os.getenv("GMAIL_APP_PASSWORD")
        self.recipient = recipient or os.getenv("GMAIL_RECIPIENT")

    def send_email(self, subject: str, html_body: str) -> bool:
        if not self.sender or not self.password or not self.recipient:
            logger.debug(f"[Email] SMTP credentials not set; email not sent: {subject}")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"Aegis Financial Agent <{self.sender}>"
            msg["To"] = self.recipient

            part = MIMEText(html_body, "html")
            msg.attach(part)

            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
                server.login(self.sender, self.password)
                server.sendmail(self.sender, [self.recipient], msg.as_string())

            logger.info(f"[Email] Successfully sent email: '{subject}' to {self.recipient}")
            return True
        except Exception as exc:
            logger.error(f"[Email] Failed to send email '{subject}': {exc}")
            return False
