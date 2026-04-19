"""Top-level analyze router — each method gets its own sub-router."""
from fastapi import APIRouter

from app.api.routes import (
    correlation,
    cox_reg,
    descriptive,
    hypothesis,
    linear_reg,
    linear_reg_adjusted,
    logistic_reg,
    logistic_reg_adjusted,
    prediction,
    psm,
    survival,
    table_one,
    ttest,
)

router = APIRouter()

router.include_router(descriptive.router, prefix="/descriptive")
router.include_router(table_one.router, prefix="/table_one")
router.include_router(ttest.router, prefix="/ttest")
router.include_router(hypothesis.router, prefix="/hypothesis")
router.include_router(correlation.router, prefix="/correlation")
router.include_router(linear_reg.router, prefix="/linear_reg")
router.include_router(linear_reg_adjusted.router, prefix="/linear_reg_adjusted")
router.include_router(logistic_reg.router, prefix="/logistic_reg")
router.include_router(logistic_reg_adjusted.router, prefix="/logistic_reg_adjusted")
router.include_router(survival.router, prefix="/survival")
router.include_router(cox_reg.router, prefix="/cox_reg")
router.include_router(psm.router, prefix="/psm")
router.include_router(prediction.router, prefix="/prediction")
