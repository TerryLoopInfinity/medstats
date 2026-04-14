"""TODO: implement this stats module."""
import logging

import pandas as pd

from app.models.analysis import AnalysisResult

logger = logging.getLogger(__name__)


def run(df: pd.DataFrame, params: dict) -> AnalysisResult:
    raise NotImplementedError("This module is not yet implemented")
