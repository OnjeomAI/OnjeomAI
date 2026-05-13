from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BASE_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"
    ADAPTER_PATH: str = "./models/korean_qa"   # 로컬 경로 (있으면 우선 사용)
    HF_REPO_ID: str = "Onjeom/korean_qa"       # 로컬 없을 때 HF에서 자동 다운로드
    CHROMA_PATH: str = "./chroma_db"
    EMBEDDING_MODEL: str = "jhgan/ko-sroberta-multitask"

    class Config:
        env_file = ".env"


settings = Settings()
