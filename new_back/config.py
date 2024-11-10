from pathlib import Path
from pydantic import BaseModel
import os

WEIGHTS_PATH = "hope_bert"

if os.environ.get("mode", "dev") == 'prod':
    DB_PATH = "/static/db/db.sqlite3"
    FAISS_INDEX_PATH = "/static/db/vacancy_index.index"
else:
    DB_PATH = "./db.sqlite3"

print(DB_PATH)

class DBSettings(BaseModel):
    url: str = f"sqlite+aiosqlite:///{DB_PATH}"
    url_sync: str = f"sqlite:///{DB_PATH}"


class Settings(BaseModel):
    db: DBSettings = DBSettings()
    static_dir_path: Path = "./static"

settings = Settings()
