"""Smoke tests for the upload endpoint."""
import io

import pandas as pd
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
def sample_csv() -> bytes:
    df = pd.DataFrame({"age": [30, 45, 60], "sbp": [120, 135, 150], "group": ["A", "B", "A"]})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_upload_csv(sample_csv):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/upload",
            files={"file": ("sample.csv", sample_csv, "text/csv")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["rows"] == 3
    assert "age" in data["column_names"]
