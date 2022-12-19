FROM python:3.10-slim

USER root

RUN apt-get update \
    && apt-get install git -y

# RUN useradd --create-home -s /bin/bash ASKCOS
# USER ASKCOS

WORKDIR /askcos/context

COPY . .

RUN pip install -r requirements.txt --no-cache-dir

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
