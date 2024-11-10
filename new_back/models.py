from typing import List, Optional
from sqlmodel import SQLModel
from sqlmodel import Field, Relationship


class UserVacancyLink(SQLModel, table=True):
    """
    Таблица Many-To-Many, для User и Vacancy
    Один пользователь может иметь несколько (топ 10) vacancies, которые для него наиболее хорошо подходят
    Одна вакансия может иметь несколько (топ 10) пользователей, которые для неё подходят
    """

    __tablename__ = "user_vacancy"

    user_id: Optional[int] = Field(
        default=None, foreign_key="users.id", primary_key=True, ondelete="CASCADE"
    )
    vacancy_id: Optional[int] = Field(
        default=None, foreign_key="vacancies.id", primary_key=True, ondelete="CASCADE"
    )
    probability: float

    user: "User" = Relationship(back_populates="top_vacancies")
    vacancy: "Vacancy" = Relationship(back_populates="top_users")


class User(SQLModel, table=True):
    """
    Класс со всей инфой пользователя
    One-To-One к User
    """

    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True, index=True, nullable=False)

    name: Optional[str]

    # Вакансии, которые подходят пользователю как соискателю, one-to-many
    top_vacancies: List["UserVacancyLink"] = Relationship(
        back_populates="user", cascade_delete=True
    )

    # Последняя видеовизитка, которую он загрузил, one-to-one
    video: Optional["Video"] = Relationship(
        cascade_delete=True,
        sa_relationship_kwargs={"uselist": False},
        back_populates="owner",
    )

    ocean: Optional["OCEAN"] = Relationship(
        cascade_delete=True,
        sa_relationship_kwargs={"uselist": False},
        back_populates="owner",
    )

class OCEAN(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True, index=True, nullable=False)
    owner: "User" = Relationship(back_populates="ocean")
    owner_id: int = Field(default=None, foreign_key="users.id", ondelete="CASCADE")
    extraversion: float
    neuroticism: float
    agreeableness: float
    conscientiousness: float
    openness: float

class OCEANVacancy(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True, index=True, nullable=False)
    vacancy: "Vacancy" = Relationship(back_populates="ocean")
    vacancy_id: int = Field(default=None, foreign_key="vacancies.id", ondelete="CASCADE")
    extraversion: float
    neuroticism: float
    agreeableness: float
    conscientiousness: float
    openness: float

class Video(SQLModel, table=True):
    """
    Класс для хранения видео-визиток
    """

    __tablename__ = "videos"

    id: int | None = Field(default=None, primary_key=True, index=True, nullable=False)

    # one-to-one
    owner: "User" = Relationship(back_populates="video")
    owner_id: int = Field(default=None, foreign_key="users.id", ondelete="CASCADE")

    # Путь к видео-визитке (mp4 файл)
    video_path: str
    transcription: str


class Vacancy(SQLModel, table=True):
    """
    Таблица вакансий
    """

    __tablename__ = "vacancies"

    id: int | None = Field(default=None, primary_key=True, index=True, nullable=False)

    # Кто создал вакансию - FK к User
    name: str

    # Топ подходящих к вакансии пользователей
    top_users: List["UserVacancyLink"] = Relationship(back_populates="vacancy")

    ocean: "OCEANVacancy" = Relationship(
        cascade_delete=True,
        sa_relationship_kwargs={"uselist": False},
        back_populates="vacancy",
    )
