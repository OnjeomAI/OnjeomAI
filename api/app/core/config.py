from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # korean_qa 모델 (이성진)
    BASE_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"
    ADAPTER_PATH: str = "./models/korean_qa"
    HF_REPO_ID: str = "Onjeom/korean_qa"
    CHROMA_PATH: str = "./chroma_db"
    EMBEDDING_MODEL: str = "jhgan/ko-sroberta-multitask"

    # writing 모델 (김우주)
    writing_model_name: str = "Onjeom/writing-ai"
    writing_max_seq_length: int = 1536
    mock_model: bool = False

    # 서버
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
