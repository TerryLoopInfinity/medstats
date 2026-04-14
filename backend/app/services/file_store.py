"""Simple file-based session store for uploaded DataFrames."""
import logging
import os
import uuid
from pathlib import Path

import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)


def _upload_dir() -> Path:
    p = Path(settings.UPLOAD_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_dataframe(df: pd.DataFrame) -> str:
    """Persist a DataFrame and return a unique file_id."""
    file_id = uuid.uuid4().hex
    path = _upload_dir() / f"{file_id}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved DataFrame %s (%d rows)", file_id, len(df))
    return file_id


def load_dataframe(file_id: str) -> pd.DataFrame:
    """Load a previously saved DataFrame by file_id."""
    path = _upload_dir() / f"{file_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"file_id '{file_id}' 不存在或已过期")
    return pd.read_parquet(path)
