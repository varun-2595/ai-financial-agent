"""
Centralized logger using loguru.
Import `logger` from this module everywhere.
"""
import sys
import os
from loguru import logger

_configured = False


def configure_logger(log_level: str | None = None) -> None:
    global _configured
    if _configured:
        return

    level = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()

    logger.remove()  # remove default handler

    # Console — coloured, human-readable
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # File — daily rotation, 30-day retention
    log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger.add(
        os.path.join(log_dir, "agent_{time:YYYY-MM-DD}.log"),
        level=level,
        rotation="00:00",      # new file every midnight
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    _configured = True


# Auto-configure on import
configure_logger()

__all__ = ["logger", "configure_logger"]
