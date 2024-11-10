FROM python:3.10

WORKDIR /back

COPY ./requirements.txt /back/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /back/requirements.txt

COPY . /back/app

VOLUME [ "/static" ]
VOLUME [ "/weights" ]
EXPOSE 8000
ENV mode=prod

CMD [ "fastapi", "run", "app/main.py"]
