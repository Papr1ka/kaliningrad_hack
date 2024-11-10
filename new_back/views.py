from random import random
from typing import Annotated, List, Optional
from fastapi import APIRouter, Depends, Path, Query, status, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from service import get_all_users, get_all_vacancies, save_user_video, bigfive_to_mbti, NBTI_TO_TITLE, ocean_to_holland_codes

from models import OCEANVacancy, User, Vacancy, Video, UserVacancyLink, OCEAN

from dependencies import session_dependency
from fastapi import Form

import numpy as np

from full_pipeline import generate_features

from get_plots import get_ocean_plot
from search_faiss import find_simmilar, add

router = APIRouter(prefix="/api", tags=["api"])


@router.post("/create_user", status_code=status.HTTP_201_CREATED)
async def create_user_view(
    name: Annotated[str, Form()], video: UploadFile, transcription: Annotated[str, Form()], session: Annotated[AsyncSession, Depends(session_dependency)]
):
    new_user = User(name=name)
    session.add(new_user)
    await session.commit()
    created_video: Video = await save_user_video(new_user, video, transcription, session)

    features = [float(i) for i in list(generate_features(created_video.video_path, transcription)[0])]
    # features = [random() for _ in range(6)]
    ocean = OCEAN(owner=new_user, extraversion=features[0], neuroticism=features[1], agreeableness=features[2], conscientiousness=features[3], openness=features[4])

    session.add(ocean)
    await session.commit()

    print("OCEAN ADDED" + "/n" * 5)

    features_dict = {
        'extraversion': features[0],
        'neuroticism': features[1],
        'agreeableness': features[2],
        'conscientiousness': features[3],
        'openness': features[4],
    }
    plot = get_ocean_plot(features_dict)
    # vacancies, count = await get_all_vacancies(session)
    simillar = find_simmilar(np.array(list(features_dict.values())).reshape((1, 5)).astype(np.float32))
    vacancies = []

    print(simillar)

    for v in simillar:
        v_id = int(v['ID'])
        probability = float(v['Score'])
        vacancy = await session.get(Vacancy, v_id)

        link = UserVacancyLink(user_id=new_user.id, vacancy_id=v_id, probability=probability)

        session.add(link)

        # vacancy.top_users.append(new_user)
        # await session.refresh(vacancy)
        vacancies.append({
                "name": vacancy.name,
                "probability": probability
            })
    
    await session.commit()

    ocean_vector = list(features_dict.values())
    nbti_tag = bigfive_to_mbti(ocean_vector)

    return {
        "name": new_user.name,
        "plot": plot,
        "vacancies": vacancies,
        "verdict": nbti_tag,
        "nbti": NBTI_TO_TITLE[nbti_tag.upper()],
        "holland": ocean_to_holland_codes(ocean_vector)
    }

@router.get("/get_users", status_code=status.HTTP_200_OK)
async def get_users_view(session: Annotated[AsyncSession, Depends(session_dependency)], skip: Annotated[Optional[int], Query] = 0, count: Annotated[Optional[int], Query] = 10, filter: Annotated[Optional[str | None], Query] = None):
    users, count = await get_all_users(session, skip, count, filter)
    result = []
    for user in users[::-1]:
        ocean = user.ocean
        features_dict = {
            'extraversion': ocean.extraversion,
            'neuroticism': ocean.neuroticism,
            'agreeableness': ocean.agreeableness,
            'conscientiousness': ocean.conscientiousness,
            'openness': ocean.openness,
        }

        plot = get_ocean_plot(features_dict)
        top_vacancies: List[UserVacancyLink] = user.top_vacancies
        top_vacancies.sort(key=lambda x: x.probability)
        top_ten = top_vacancies[:10]
        vacancies = []
        for link in top_ten:
            await session.refresh(link, ["vacancy"])
            vacancies.append({
                "name": link.vacancy.name,
                "probability": link.probability
            })
        ocean_vector = list(features_dict.values())
        nbti_tag = bigfive_to_mbti(ocean_vector)
        result.append({
            "name": user.name,
            "plot": plot,
            "vacancies": vacancies,
            "verdict": nbti_tag,
            "nbti": NBTI_TO_TITLE[nbti_tag.upper()],
            "holland": ocean_to_holland_codes(ocean_vector)
        })
    return {
        "data": result,
        "count": count
    }

@router.get("/get_vacancies", status_code=status.HTTP_200_OK)
async def get_vacancies_view(session: Annotated[AsyncSession, Depends(session_dependency)], skip: Annotated[Optional[int], Query] = 0, count: Annotated[Optional[int], Query] = 10, filter: Annotated[Optional[str | None], Query] = None):
    vacancies, count = await get_all_vacancies(session, skip, count, filter)
    result = []

    for vacancy in vacancies:
        ocean = vacancy.ocean
        features_dict = {
            'extraversion': ocean.extraversion,
            'neuroticism': ocean.neuroticism,
            'agreeableness': ocean.agreeableness,
            'conscientiousness': ocean.conscientiousness,
            'openness': ocean.openness,
        }

        plot = get_ocean_plot(features_dict)
        
        top_users: List[UserVacancyLink] = vacancy.top_users
        top_users.sort(key=lambda x: x.probability)
        top_ten = top_users[:10]

        users = []
        for link in top_ten:
            await session.refresh(link, ["user"])
            users.append({
                "name": link.user.name,
                "probability": link.probability
            })
        ocean_vector = list(features_dict.values())
        nbti_tag = bigfive_to_mbti(ocean_vector)
        result.append({
            "name": vacancy.name,
            "plot": plot,
            "users": users,
            "verdict": nbti_tag,
            "nbti": NBTI_TO_TITLE[nbti_tag.upper()],
            "holland": ocean_to_holland_codes(ocean_vector)
        })
    return {
        "data": result,
        "count": count
    }


@router.post("/create_vacancy", status_code=status.HTTP_201_CREATED)
async def create_vacancy_view(name: Annotated[str, Form()],
                              extraversion: Annotated[float, Form()],
                              neuroticism: Annotated[float, Form()],
                              agreeableness: Annotated[float, Form()],
                              conscientiousness: Annotated[float, Form()],
                              openness: Annotated[float, Form()],
                              session: Annotated[AsyncSession, Depends(session_dependency)]):
    
    new_ocean = OCEANVacancy(name=name, extraversion=extraversion, neuroticism=neuroticism, agreeableness=agreeableness, conscientiousness=conscientiousness, openness=openness)
    vacancy = Vacancy(name=name, ocean=new_ocean)
    session.add(vacancy)
    await session.commit()

    ocean = new_ocean
    features_dict = {
        'extraversion': ocean.extraversion,
        'neuroticism': ocean.neuroticism,
        'agreeableness': ocean.agreeableness,
        'conscientiousness': ocean.conscientiousness,
        'openness': ocean.openness,
    }

    add(vacancy.id, np.array(list(features_dict.values())).astype(np.float32))

    plot = get_ocean_plot(features_dict)
    ocean_vector = list(features_dict.values())
    nbti_tag = bigfive_to_mbti(ocean_vector)

    return {
        "name": name,
        "plot": plot,
        "users": [],
        "verdict": nbti_tag,
        "nbti": NBTI_TO_TITLE[nbti_tag.upper()],
        "holland": ocean_to_holland_codes(ocean_vector)
    }
