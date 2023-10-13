# basic python3 image as base
#FROM harbor2.vantage6.ai/infrastructure/algorithm-base:latest
FROM pmateus/algorithm-base:1.0.0
# This is a placeholder that should be overloaded by invoking
# docker build with '--build-arg PKG_NAME=...'
ARG PKG_NAME="v6_LinReg_py"

#set working directory
#WORKDIR /app

#ENV PYTHONPATH="/home/swier/miniconda3/envs/vantage6/bin/python"
# install federated algorithm
#COPY ./requirements.txt requirements.txt
#RUN pip install -r requirements.txt

COPY . /app

#FROM python 

RUN pip install /app

ENV PKG_NAME=${PKG_NAME}

# Tell docker to execute `docker_wrapper()` when the image is run.
CMD python -c "from v6_LinReg_py.docker_wrapper import docker_wrapper; docker_wrapper('${PKG_NAME}')"
