FROM mambaorg/micromamba:2.3.2-debian13-slim
LABEL maintainer="Dias Bakhtiyarov"

ARG NEW_MAMBA_USER=skyterra
ARG NEW_MAMBA_USER_ID=1000
ARG NEW_MAMBA_USER_GID=1000
USER root

RUN usermod "--login=${NEW_MAMBA_USER}" "--home=/home/${NEW_MAMBA_USER}" \
        --move-home "-u ${NEW_MAMBA_USER_ID}" "${MAMBA_USER}" && \
    groupmod "--new-name=${NEW_MAMBA_USER}" \
        "-g ${NEW_MAMBA_USER_GID}" "${MAMBA_USER}" && \
    echo "${NEW_MAMBA_USER}" > "/etc/arg_mamba_user" &&\
    :

ENV MAMBA_USER=$NEW_MAMBA_USER
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml

RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

WORKDIR /project

COPY --chown=$MAMBA_USER:$MAMBA_USER . /project

ENV PROJ_LIB=/opt/conda/share/proj

CMD ["python", "delineate.py", "-b", "batch_sample.yaml"]