"""POST /api/analyze/psm — 倾向性得分匹配"""
import logging

from fastapi import APIRouter, HTTPException

from app.models.analysis import AnalysisRequest, AnalysisResult
from app.services.file_store import load_dataframe
from app.stats import psm

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=AnalysisResult)
async def analyze_psm(req: AnalysisRequest) -> AnalysisResult:
    try:
        df = load_dataframe(req.file_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        result = psm.run(df, req.params)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("PSM analysis failed")
        raise HTTPException(status_code=500, detail=f"分析失败：{e}")

    return result
