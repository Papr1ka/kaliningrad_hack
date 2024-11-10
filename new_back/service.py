import os
from typing import List
import aiofiles
from fastapi import UploadFile
from sqlmodel import select, func, col
from time import time

from models import User, Vacancy, Video
# from .schemas import CreateProfile, CreateVacancy, UpdateOCEAN, UpdateProfile, UpdateUser, UpdateVacancy
from sqlalchemy.ext.asyncio import AsyncSession
# from .get_plots import get_ocean_plot
# from sqlmodel import select
from config import settings

# Тут везде, где _user, для пользователя будет недоступен profile!
# Будет misset_greenlet, чтобы был доступен, надо session.refresh(user, ['profile'])

# async def create_profile(profile: CreateProfile, session: AsyncSession) -> Profile:
#     new_profile = Profile(**profile.model_dump(), owner=profile)
#     session.add(new_profile)
#     await session.commit()

#     return new_profile

# async def get_profile_by_username(user_id: int, session: AsyncSession) -> Profile:
#     return await session.get(Profile, user_id)

# async def get_profile(user_id: int, session: AsyncSession) -> Profile:
#     return await session.get(Profile, user_id)


# async def delete_profile(user: Profile, session: AsyncSession) -> None:
#     await session.delete(user)
#     await session.commit()

# async def update_profile(
#     profile: Profile, to_update: UpdateProfile, session: AsyncSession
# ) -> Profile:
#     for name, value in to_update.model_dump().items():
#         setattr(profile, name, value)
#     session.add(profile)
#     await session.commit()
#     return profile

def get_file_path(user_id: int, filename: str):
    return os.path.join(settings.static_dir_path, f"{user_id}_{time()}_{filename}")

async def save_video(video_file: UploadFile, full_path) -> None:
    async with aiofiles.open(full_path, "wb") as out_file:
        content = await video_file.read()
        await out_file.write(content)


async def save_user_video(
    user: User, video_file: UploadFile, transcription: str, session: AsyncSession
) -> Video:
    video_full_path = get_file_path(user.id, video_file.filename)

    await save_video(video_file, video_full_path)

    await session.refresh(user, ["video"])

    if user.video is not None and os.path.exists(user.video.video_path):
        os.remove(user.video.video_path)

    video = Video(owner=user, video_path=video_full_path, transcription=transcription)
    session.add(video)
    await session.commit()
    return video


async def get_all_vacancies(session: AsyncSession, skip: int, count: int, filter: str | None):
    if (filter is not None):
        filtered = select(Vacancy).where(col(Vacancy.name).contains(filter))
    else:
        filtered = select(Vacancy)

    vacancies: List[Vacancy] = list((await session.execute(filtered.offset(skip).limit(count))).scalars().all())
    overall_count = await session.scalar(select(func.count(Vacancy.id)))
    for v in vacancies:
        await session.refresh(v, ["top_users", "ocean"])
    return vacancies, overall_count

async def get_all_users(session: AsyncSession, skip: int, count: int, filter: str | None):
    if (filter is not None):
        filtered = select(User).where(col(User.name).contains(filter))
    else:
        filtered = select(User)
    
    users: List[User] = list((await session.execute(filtered.offset(skip).limit(count))).scalars().all())
    
    overall_count = await session.scalar(select(func.count(User.id)))
    for v in users:
        await session.refresh(v, ["top_vacancies", "video", "ocean"])
    return users, overall_count

def bigfive_to_mbti(data):
    print("MBTI calculation" + "/n" * 5, data)
    # ENACO
    # Extraversion (E/I)
    mbti = ""

    mbti += 'E' if data[0] >= 0.5 else 'I'

    # Openness (N/S)
    mbti += 'N' if data[4] >= 0.5 else 'S'

    # Agreeableness (F/T)
    mbti += 'F' if data[2] >= 0.5 else 'T'

    # Conscientiousness (J/P)
    mbti += 'J' if data[3] >= 0.5 else 'P'

    return mbti.lower()

NBTI_TO_TITLE = {
    'INTJ': 'Architect',
    'INTP': 'Logician',
    'ENTJ': 'Commander',
    'ENTP': 'Debater',
    'INFJ': 'Advocate',
    'INFP': 'Mediator',
    'ENFJ': 'Protagonist',
    'ENFP': 'Campaigner',
    'ISTJ': 'Logistician',
    'ISFJ': 'Defender',
    'ESTJ': 'Executive',
    'ESFJ': 'Consul',
    'ISTP': 'Virtuoso',
    'ISFP': 'Adventurer',
    'ESTP': 'Entrepreneur',
    'ESFP': 'Entertainer'
}

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

# async def create_vacancy(
#     vacancy_to_create: CreateVacancy, profile: Profile, session: AsyncSession
# ) -> Vacancy:
#     vacancy = Vacancy(**vacancy_to_create.model_dump(), owner=profile)
#     session.add(vacancy)
#     await session.commit()

#     return vacancy


# async def delete_vacancy(vacancy: Vacancy, session: AsyncSession) -> None:
#     await session.delete(vacancy)
#     await session.commit()


# async def update_vacancy(
#     vacancy: Vacancy, to_update: UpdateVacancy, session: AsyncSession
# ) -> Vacancy:
#     for name, value in to_update.model_dump().items():
#         setattr(vacancy, name, value)
#     session.add(vacancy)
#     await session.commit()
#     return vacancy


# async def get_vacancy_plot(vacancy_id: int):
#     ocean = get_ocean_by_id(vacancy_id)
#     plot = get_ocean_plot(ocean)
#     return plot

# async def get_all_vacancies(session: AsyncSession):
#     query = select(Vacancy)
#     result = await session.execute(query)
#     return result.scalars().all()

# async def get_all_users(session: AsyncSession):
#     query = select(Profile)
#     result = await session.execute(query)
    
#     profiles = []

#     for profile in result:
#         await session.refresh(profile, ["video", "top_vacancies"])
#         profiles.append(profile)
#     return profiles
