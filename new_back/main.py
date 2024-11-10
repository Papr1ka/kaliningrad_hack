from contextlib import asynccontextmanager
from fastapi import FastAPI

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from views import router as views_router
from config import settings
from transformers import AutoModel
import torch
import torch.nn as nn

class BertMultiRegressor(nn.Module):
    """
    Модель мульти регрессора на основе эмбеддингов BERT
    """
    def __init__(self, bert_model_name, output_size):
        super(BertMultiRegressor, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size + 7, output_size)

    def forward(self,
                input_ids,
                attention_mask=None,
                audio_traits=None):

        # Текстовые данные обрабатываются в BERT
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )

        # Берём [CLS] токен для получения эмбеддингов текста
        last_hidden_state = outputs.last_hidden_state[:, 0, :]

        # Объединяем числовые данные с эмбеддингами текста
        result = torch.concat((last_hidden_state, audio_traits), dim=1)

        outputs = self.regressor(result).to(torch.float64)

        return {
            "logits": outputs
        }

app = FastAPI()

app.mount("/static", StaticFiles(directory=settings.static_dir_path), name="static")

origins = [
    "http://213.171.28.36:5173",
    "https://213.171.28.36:5173",
    "http://localhost:5173",
    "https://localhost:5173",
    "*"
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
