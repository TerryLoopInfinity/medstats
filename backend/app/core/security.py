import logging
from pathlib import Path

import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)


class FileValidationError(ValueError):
    pass


def validate_upload_file(filename: str, content: bytes) -> None:
    """Validate uploaded file: extension, size, and content safety."""
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise FileValidationError(f"不支持的文件格式: {ext}，仅接受 .csv 和 .xlsx")

    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise FileValidationError(
            f"文件大小 {size_mb:.1f}MB 超过限制 {settings.MAX_FILE_SIZE_MB}MB"
        )

    # Reject files with null bytes (potential executable content)
    if b"\x00" in content[:1024]:
        raise FileValidationError("文件包含非法字节，已拒绝")


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """Validate DataFrame dimensions and return warnings."""
    warnings: list[str] = []

    if len(df) > settings.MAX_ROWS:
        raise FileValidationError(
            f"数据行数 {len(df)} 超过上限 {settings.MAX_ROWS}"
        )
    if len(df.columns) > settings.MAX_COLS:
        raise FileValidationError(
            f"数据列数 {len(df.columns)} 超过上限 {settings.MAX_COLS}"
        )
    if df.empty:
        raise FileValidationError("文件为空，无法分析")

    missing_ratio = df.isnull().mean().max()
    if missing_ratio > 0.5:
        warnings.append(f"部分变量缺失率超过 50%，请检查数据质量")

    return warnings
