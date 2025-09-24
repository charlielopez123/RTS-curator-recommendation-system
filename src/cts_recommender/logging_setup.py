import logging
import os
from typing import Optional

def setup_logging(level: Optional[str] = None) -> None:
    """
    Minimal logging setup.
    - Uses LOG_LEVEL env if level is None (default DEBUG).
    - Configures a single console handler via logging.basicConfig.
    """
    level_name = (level or os.getenv("LOG_LEVEL") or "DEBUG").upper()
    # Fallback to INFO if user passes something weird
    level_value = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level_value,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )