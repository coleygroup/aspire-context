# Askcos Context Service
a standalone context recommendation module from Askcos rewritten as a web service in FastAPI for ASPIRE project

## Deployment

### Docker

**Manual**
1. `docker build -t TAG_NAME .`.

    - we usually supply `-t ASKCOS_CONTEXT`
    - By default, all commands are run under the `mambauser` user. If you feeled compelled, you can change this value by supplying `--build-arg MAMBA_USER=USERNAME` to `docker build`.

1. `docker run -d --name NAME -p 8000:8000 TAG_NAME`

    - `TAG_NAME` should be same as the above command
    - `-p 8000:8000` means (roughly) "expose port `8000` on the host machine to port `8000` on the container." It's standard for web servies to be mapped to port `8000 + i`, where `i` is a number in `[0, 1000]`

**using `docker compose`**
1. `docker compose up -d`.

    - the `-d` flag starts the container in the background so you can go about your business.