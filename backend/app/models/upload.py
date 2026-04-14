from pydantic import BaseModel


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    rows: int
    columns: int
    column_names: list[str]
    preview: list[list]  # first 20 rows as list of lists
    warnings: list[str]
