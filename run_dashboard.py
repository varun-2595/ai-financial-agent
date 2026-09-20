"""
Launcher script for the Aegis Local Interactive Financial Dashboard.
Runs FastAPI backend on http://localhost:8000.

Usage:
  python run_dashboard.py
  python run_dashboard.py --port 8080
"""
import argparse
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description="Aegis Interactive Web Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload on code change")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"🚀 AEGIS FINANCIAL DASHBOARD STARTING AT http://{args.host}:{args.port}")
    logger.info("=" * 60)

    uvicorn.run(
        "src.dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
