"""Top-level analyze router — each method gets its own sub-router."""
from fastapi import APIRouter

from app.api.routes import descriptive

router = APIRouter()

router.include_router(descriptive.router, prefix="/descriptive")
