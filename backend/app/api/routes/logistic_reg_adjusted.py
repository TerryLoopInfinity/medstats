"""POST /api/analyze/logistic_reg_adjusted — Logistic 回归控制混杂偏倚"""
import logging

from fastapi import APIRouter, HTTPException

from app.models.analysis import AnalysisRequest, AnalysisResult
from app.services.file_store import load_dataframe
from app.stats import logistic_reg_adjusted

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=AnalysisResult)
async def analyze_logistic_reg_adjusted(req: AnalysisRequest) -> AnalysisResult:
    try:
        df = load_dataframe(req.file_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        result = logistic_reg_adjusted.run(df, req.params)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("logistic_reg_adjusted analysis failed")
        raise HTTPException(status_code=500, detail=f"分析失败：{e}")

    return result
