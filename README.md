# Askcos Context Service
a standalone context recommendation module from Askcos rewritten as a web service in FastAPI for ASPIRE project

## Table of Contents

- [link](#first-steps)
- [Deployment](#deployment)
    - [Docker](#docker-preferred)
    - [Native](#native)
- [Testing](#testing)

## First steps

0. (if necessary) install git LFS: `git lfs install`
1. clone this repo: `git clone THIS_REPO`
2. pull the files from git LFS: `git lfs pull`

## Deployment

### Docker (preferred)

**Automatic** (via `docker compose`)
1. `docker compose up -d`.

    - the `-d` flag starts the container in the background so you can go about your business.

**Manual**
1. `docker build -t TAG_NAME .`.

    - we usually supply `-t ASKCOS_CONTEXT`
    - By default, all commands are run under the `mambauser` user. If you feeled compelled, you can change this value by supplying `--build-arg MAMBA_USER=USERNAME` to `docker build`.

1. `docker run -d --name NAME -p 8000:8000 TAG_NAME`

    - `TAG_NAME` should be same as the above command
    - `-p 8000:8000` means (roughly) "expose port `8000` on the host machine to port `8000` on the container." It's standard for web services to be mapped to a port in the range [8000, 9000)

___

### Native
_Note_: all steps should be run from the top-level directory of this repo

0. (if necessary) install (micro)conda
1. build the conda environment: `conda env create -f env.yaml -n NAME`
1. activate the environment: `conda activate NAME`
1. install the `app` package: `pip install . --no-deps`
1. run the service: `uvicorn app.main:app --host localhost --port 8000`

## Testing
1. to check if the service is running:
    ```
    $ curl -X GET localhost:8000/health
    {"message":"Alive!"}
    ```