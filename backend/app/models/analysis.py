from typing import Any

from pydantic import BaseModel, Field


class TableResult(BaseModel):
    title: str = ""
    headers: list[str]
    rows: list[list[Any]]


class ChartResult(BaseModel):
    title: str = ""
    chart_type: str  # e.g. "bar", "line", "scatter", "kaplan_meier"
    option: dict[str, Any]  # ECharts option JSON


class AnalysisResult(BaseModel):
    method: str
    tables: list[TableResult] = Field(default_factory=list)
    charts: list[ChartResult] = Field(default_factory=list)
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)


class AnalysisRequest(BaseModel):
    file_id: str
    params: dict[str, Any] = Field(default_factory=dict)
