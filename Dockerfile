FROM mambaorg/micromamba:0.19.1
LABEL maintainer="Peter Pietrzyk"

# install tk for python3 (need to temporarily switch to root since micromamba sets the default user)
USER root
RUN apt-get update && apt-get install -y python3-tk curl bzip2
USER $MAMBAUSER

# copy source and configure conda environment
WORKDIR /opt/code
COPY . /opt/code
COPY --chown=$MAMBA_USER:$MAMBA_USER dirtmu_environment_python3.yml /opt/code/dirtmu_environment_python3.yml
RUN micromamba create --yes --file /opt/code/dirtmu_environment_python3.yml && micromamba clean --all --yes
ENV ENV_NAME="DIRTmu_env_py3"

# fix matplotlib Qt issue (https://stackoverflow.com/a/52353715)
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip uninstall -y matplotlib && \
    python -m pip install --upgrade pip && \
    pip install matplotlib

# environment should automatically activate
RUN chmod +x activate_env.sh
ENV BASH_ENV=/opt/code/activate_env.sh