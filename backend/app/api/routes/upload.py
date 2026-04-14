import io
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile

from app.core.security import FileValidationError, validate_dataframe, validate_upload_file
from app.models.upload import UploadResponse
from app.services.file_store import save_dataframe

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile) -> UploadResponse:
    content = await file.read()

    try:
        validate_upload_file(file.filename or "", content)
    except FileValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Parse
    try:
        if (file.filename or "").endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"文件解析失败: {e}")

    try:
        warnings = validate_dataframe(df)
    except FileValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    file_id = save_dataframe(df)
    preview = df.head(20).fillna("").values.tolist()

    return UploadResponse(
        file_id=file_id,
        filename=file.filename or "",
        rows=len(df),
        columns=len(df.columns),
        column_names=list(df.columns),
        preview=preview,
        warnings=warnings,
    )
