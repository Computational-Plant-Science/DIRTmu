FROM mambaorg/micromamba:0.19.1

LABEL maintainer="Peter Pietrzyk"

# install tk for python2 (need to temporarily switch to root since micromamba sets the default user)
USER root
RUN apt-get update && apt-get install -y python-tk
USER $MAMBAUSER

# copy source
COPY . /opt/code
COPY --chown=$MAMBA_USER:$MAMBA_USER dirtmu_environment.yml /opt/code/dirtmu_environment.yml

# create the conda environment
RUN micromamba create --yes --file /opt/code/dirtmu_environment.yml && \
    micromamba clean --all --yes

# set the default conda environment name
ENV ENV_NAME="DIRTmu_env"

# fix matplotlib Qt issue (https://stackoverflow.com/a/52353715, don't ask me why this works)
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip uninstall -y matplotlib && \
    python -m pip install --upgrade pip && \
    pip install matplotlib
