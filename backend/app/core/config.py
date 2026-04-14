from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # File upload limits
    MAX_FILE_SIZE_MB: int = 10
    MAX_ROWS: int = 100_000
    MAX_COLS: int = 500
    ALLOWED_EXTENSIONS: set[str] = {".csv", ".xlsx"}

    # Storage
    UPLOAD_DIR: str = "/tmp/medstats_uploads"

    # Stats defaults
    DEFAULT_CI: float = 0.95
    P_VALUE_DECIMALS: int = 3


settings = Settings()
