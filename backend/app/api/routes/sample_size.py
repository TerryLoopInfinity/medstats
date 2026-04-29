"""POST /api/analyze/sample_size — 在线样本量计算（无需上传文件）"""
import logging

from fastapi import APIRouter, HTTPException

from app.models.analysis import AnalysisRequest, AnalysisResult
from app.stats import sample_size

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=AnalysisResult)
async def analyze_sample_size(req: AnalysisRequest) -> AnalysisResult:
    # 样本量计算不依赖上传的数据文件，直接使用 params
    try:
        result = sample_size.run(req.params)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("样本量计算失败")
        raise HTTPException(status_code=500, detail=f"计算失败：{e}")

    return result
