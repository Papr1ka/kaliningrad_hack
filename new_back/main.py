from contextlib import asynccontextmanager
from fastapi import FastAPI

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from views import router as views_router
from config import settings

app = FastAPI()

app.mount("/static", StaticFiles(directory=settings.static_dir_path), name="static")

origins = [
    "http://213.171.28.36:5173",
    "https://213.171.28.36:5173",
    "http://localhost:5173",
    "https://localhost:5173",
    # "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(views_router)


@app.get("/")
def hello():
    return {"hello": "world"}
