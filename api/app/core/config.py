from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 모델
    writing_model_name: str = "Onjeom/writing-ai"
    writing_max_seq_length: int = 1536

    # 서버
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # 모델 없이 API 스키마만 테스트할 때 사용 (MOCK_MODEL=true)
    mock_model: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
