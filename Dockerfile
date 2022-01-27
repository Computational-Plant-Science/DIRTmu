FROM mambaorg/micromamba:0.19.1

USER root
RUN apt-get update && apt-get install -y python-tk
USER $MAMBAUSER

COPY . /opt/code
COPY --chown=$MAMBA_USER:$MAMBA_USER dirtmu_environment.yml /opt/code/dirtmu_environment.yml

RUN micromamba create --yes --file /opt/code/dirtmu_environment.yml && \
    micromamba clean --all --yes

# set the default conda environment name
ENV ENV_NAME="DIRTmu_env"

# fix matplotlib Qt issue

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip uninstall -y matplotlib && \
    python -m pip install --upgrade pip && \
    pip install matplotlib
