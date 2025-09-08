import os
from functools import lru_cache
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "Keefy")
    debug: bool = os.getenv("APP_DEBUG", "false").lower() == "true"
    secret_key: str = os.getenv("SECRET_KEY", "change_me")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", 5432))
    db_user: str = os.getenv("DB_USER", "appuser")
    db_password: str = os.getenv("DB_PASSWORD", "appsecret")
    db_name: str = os.getenv("DB_NAME", "appdb")

    media_dir: str = os.getenv("MEDIA_DIR", "app/media")

    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

@lru_cache
def get_settings() -> Settings:
    return Settings()