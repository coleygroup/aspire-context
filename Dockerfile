FROM mambaorg/micromamba:latest AS base

USER root

COPY env.yml /tmp/env.yml

ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN micromamba install -y -n base -f env.yml \
    && pip install pyscreener \
    && micromamba clean --all --yes

USER askcos

# ---------------------- #

COPY --chown=askcos:askcos . /usr/local/askcos-core

WORKDIR /home/askcos

ENV PYTHONPATH=/usr/local/askcos-core${PYTHONPATH:+:${PYTHONPATH}}

