import pandas as pd

jobs = pd.read_csv("./jobs.csv")



from sqlmodel import SQLModel, create_engine, Session
from config import settings
from models import Vacancy, OCEANVacancy

sqlite_url = settings.db.url_sync

engine = create_engine(sqlite_url, echo=True)  
session = Session(engine)


def create_db_and_tables():  
    SQLModel.metadata.create_all(engine)
    for row in jobs.values:
        ocean = OCEANVacancy(extraversion=row[2], neuroticism=row[3], agreeableness=row[4], conscientiousness=row[5], openness=row[6])
        job = Vacancy(id=row[0], name=row[1])
        ocean.vacancy = job
        session.add(ocean)
    session.commit()

# More code here later 👈

if __name__ == "__main__":  
    create_db_and_tables()
