import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from config import WEIGHTS_PATH
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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


# Модель регрессора
tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_PATH + '/hope_bert/')
model = torch.load(WEIGHTS_PATH + '/hope_bert/hope', map_location=device).to(device)


def get_vector(text_descr, emotion, transcription, audio_features):
    """
    Получение значений OCEAN с помощью регрессора на основе BERT

    Args:
      text_descr (str): Описание кадра, сгенерированное BLIP
      emotion (str): Определённая с помощью fer эмоция
      transcription (str): Транскрипция видео-визитки
      audio_features (list[float]): Метрики по аудио
    Returns:
        numpy array: Вектор размерности (1, 5), значения которого соответствуют
            метрикам 'extraversion', 'neuroticism', 'agreeableness',
           'conscientiousness', 'interview', 'openness',
           каждое значение лежит в интервале от 0 до 1
    """

    # Вектор метрик аудио прибавляется к эмбеддингам BERT перед линейным слоем
    # Поэтому важно, чтобы он был подходящей размерности
    assert len(audio_features) == 7

    # Объединение текстовых данных
    text = f"There is {text_descr}. This person feels {emotion}. This person says: {transcription}"
    inp = tokenizer([text], padding='max_length', truncation=True, max_length=512, return_tensors='pt')

    # Обработка численных данных
    inp['audio_traits'] = torch.FloatTensor([audio_features])

    # Получение выходов модели
    tokenized_res = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        outputs = model(**tokenized_res)

    metrics = outputs['logits'].cpu().detach().numpy().tolist()[0]
    vector_res = np.array([[metrics[0], metrics[1], metrics[2], metrics[3], metrics[5]]]).astype(np.float32)

    return vector_res
