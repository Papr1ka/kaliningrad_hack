import os
import cv2
from fer import FER
import pandas as pd
import subprocess

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks
import pickle

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# BLIP, обработка кадра
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Детектор эмоции
emotion_detector = FER(mtcnn=True)

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
tokenizer = AutoTokenizer.from_pretrained('hope_bert')
model_regressor = BertMultiRegressor('hope_bert', 6)
model_regressor.load_state_dict(torch.load('hope_bert/hope.pth'))


def extract_first_frame_from_video(video_path):
    """
    Извлекает первый кадр из видеофайла.
    Args:
      video_path (str): Путь к видеофайлу.
    Returns:
      Image: Кадр в формате PIL.
    """
    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл: {video_path}")
    
    # Переходим к первому кадру (индекс 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Читаем первый кадр
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Преобразуем кадр в объект Image из PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return image
    else:
        raise ValueError("Не удалось захватить первый кадр из видео.")


def generate_caption(img):
    """
    Генерирует описание для изображения с использованием модели BLIP.
    Args:
      img (Image): Изображение в формате PIL.
    Returns:
      str: Сгенерированное описание изображения.
    """
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption


def analyze_emotions(img):
    """
    Определяет эмоцию человека на изображении с использованием FER.
    Args:
      img (Image): Изображение в формате PIL.
    Returns:
      str: Название эмоции.
    """
    img = np.array(img)
    analysis = emotion_detector.top_emotion(img)
    if analysis:
        return analysis[0]
    return None


def extract_audio_features(video_path):
    """
    Извлекает аудиофичеры из видео, такие как средняя частота, энергия, темп речи, паузы.
    Args:
      video_path (str): Путь к видеофайлу.
    Returns:
      list[float]: Вектор из 7 аудио-фичей.
    """
    audio_output_path = "temp_audio.wav"
    command = [
        "ffmpeg", 
        "-i", video_path, "-vn", "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", "-y", audio_output_path
    ]
    subprocess.run(command, check=True)


    y, sr = librosa.load(audio_output_path)


    # Извлечение pitch (основная частота, F0)
    sound = parselmouth.Sound(audio_output_path)
    pitch = call(sound, "To Pitch", 0.0, 75, 500)  # Типичный диапазон частот для человеческого голоса
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan  # Обработка беззвучных сегментов

    # Среднее и дисперсия pitch
    mean_pitch = np.nanmean(pitch_values)
    variance_pitch = np.nanvar(pitch_values)

    # Извлечение energy (энергия)
    frame_length = 2048
    hop_length = 512
    energy = np.array([
        np.sum(np.abs(y[i:i + frame_length]**2))
        for i in range(0, len(y), hop_length)
    ])

    # Среднее и дисперсия энергии
    mean_energy = np.mean(energy)
    variance_energy = np.var(energy)

    # 3. Speaking Rate (темп речи)
    peaks, _ = find_peaks(energy, height=np.max(energy) * 0.3)
    speaking_rate = len(peaks) / (len(y) / sr)  # Приблизительная оценка слогов в секунду

    # Паузы
    pause_threshold = 0.02 * np.max(energy)
    pause_durations = []
    current_pause_duration = 0
    for e in energy:
        if e < pause_threshold:
            current_pause_duration += hop_length / sr
        elif current_pause_duration > 0:
            pause_durations.append(current_pause_duration)
            current_pause_duration = 0

    # Средняя длительность пауз и количество пауз
    num_pauses = len(pause_durations)
    mean_pause_duration = np.mean(pause_durations) if pause_durations else 0

    ### Формирование вектора признаков ###
    feature_vector = [
        mean_pitch,          # Средняя частота (Hz)
        variance_pitch,      # Дисперсия частоты (Hz^2)
        mean_energy,         # Средняя энергия
        variance_energy,     # Дисперсия энергии
        speaking_rate,       # Темп речи (слоги в секунду)
        num_pauses,          # Количество пауз
        mean_pause_duration  # Средняя длительность пауз (секунды)
    ]
    if os.path.exists(audio_output_path):
        os.remove(audio_output_path)
    return feature_vector





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
        outputs = model_regressor(**tokenized_res)

    metrics = outputs['logits'].cpu().detach().numpy().tolist()[0]
    vector_res = np.array([[metrics[0], metrics[1], metrics[2], metrics[3], metrics[5]]]).astype(np.float32)

    return vector_res

def generate_features(video_path, transcription):
    """
    Генерация всех фичей из видео: описание, эмоция, аудиофичеры, и Big Five.
    Args:
      video_path (str): Путь к видео.
      transcription (str): Транскрипция.
    Returns:
      np.array: Вектор OCEAN.
    """
    img = extract_first_frame_from_video(video_path)
    caption = generate_caption(img)
    emotion = analyze_emotions(img)
    audio_features = extract_audio_features(video_path)
    return get_vector(caption, emotion, transcription, audio_features)


def all_videos(file_path):
    """
    Обработка всех видео в файле для вычисления метрик OCEAN.
    Args:
      file_path (str): Путь к файлу с данными.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
        i =0
        for video, transcription in data.items():
            i += 1
            if i <3:
                video_path = 'train_dataset_vprod_encr_train/validation/' + video
                start_time = time.time()
                features = generate_features(video_path, transcription)
                print(features)
                print(ocean_to_mbti(features))
                print(ocean_to_holland_codes(features))
                print("Время выполнения", time.time() - start_time)
            else:
                break


def ocean_to_mbti(data):
    """
    Преобразует метрики OCEAN в MBTI.
    Args:
      data (np.array): Вектор OCEAN.
    Returns:
      str: Тип MBTI.
    """
    data = data[0]
    mbti = ""
    # Extraversion (E/I)
    mbti += 'E' if data[0] >= 0.5 else 'I'  # Introvert/Extrovert

    # Openness (N/S)
    mbti += 'N' if data[4] >= 0.5 else 'S'  # Sensing/Intuition

    # Agreeableness (F/T)
    mbti += 'F' if data[2] >= 0.5 else 'T'  # Feeling/Thinking

    # Conscientiousness (J/P)
    mbti += 'J' if data[3] >= 0.5 else 'P'  # Judging/Percieving

    return mbti


def ocean_to_holland_codes(data):
    """
    Преобразует метрики OCEAN в Holland code (RIASEC).
    Args:
      data (np.array): Вектор OCEAN.
    Returns:
      str: Тип Holland code.
    """
    holland_scores = {'Realistic': 0,
                      'Investigative': 0,
                      'Artistic': 0,
                      'Social': 0,
                      'Enterprising': 0,
                      'Conventional': 0}
    # Realistic (R): High Conscientiousness and low Agreeableness
    holland_scores['Realistic'] = (data[3] + (1 - data[2])) / 2

    # Investigative (I): High Openness and low Extraversion
    holland_scores['Investigative'] = (data[4] + (1 - data[0])) / 2

    # Artistic (A): High Openness and low Conscientiousness
    holland_scores['Artistic'] = (data[4] + (1 - data[3])) / 2

    # Social (S): High Agreeableness and high Extraversion
    holland_scores['Social'] = (data[2] + data[0]) / 2

    # Enterprising (E): High Extraversion and low Agreeableness
    holland_scores['Enterprising'] = (data[0] + (1 - data[2])) / 2

    # Conventional (C): High Conscientiousness and low Openness
    holland_scores['Conventional'] = (data[3] + (1 - data[4])) / 2

    return max(holland_scores)



# file_path = 'train_dataset_vprod_encr_train/transcription/transcription_validation.pkl'
# all_videos(file_path)
# video_path = 'train_dataset_vprod_encr_train/train_data/ZXeO5dRFrj0.005.mp4'
# transcription_file = 'train_dataset_vprod_encr_train/annotation/annotation_validation.csv'
#
# #video_path = 'train_dataset_vprod_encr_train/train_data/_01AyUz9J9I.002.mp4'
# start_time = time.time()
#
# caption = generate_features(video_path)
# print(caption)
# print("Время выполнения", time.time() - start_time)
#
# video_path = 'train_dataset_vprod_encr_train/train_data/_01AyUz9J9I.002.mp4'
# start_time = time.time()
#
# caption = generate_features(video_path)
# print(caption)
# print("Время выполнения", time.time() - start_time)



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

