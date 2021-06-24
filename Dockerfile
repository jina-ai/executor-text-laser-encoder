FROM jinaai/jina:2.0.0rc9

# install git
RUN apt-get -y update && apt-get install -y git

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN export CACHE_DIR=$(python -c 'from jina.hubble import JINA_HUB_CACHE_DIR; print(JINA_HUB_CACHE_DIR)')/laser_encoder && mkdir $CACHE_DIR && python -m laserembeddings download-models $CACHE_DIR

# setup the workspace
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
