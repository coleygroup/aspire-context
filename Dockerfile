FROM mambaorg/micromamba:latest

# 1. setup barebones env
USER root

RUN apt-get update \
    && apt-get install git -y

# 2. setup base python env
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /askcos/context/env.yaml

RUN micromamba install -y -n base -f /askcos/context/env.yaml \
    && micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# 3. setup actual python env
WORKDIR /askcos/context

COPY --chown=$MAMBA_USER:$MAMBA_USER . .

RUN pip install . --no-deps --no-cache-dir

CMD ["uvicorn", "app.main:app", "--reload"]
