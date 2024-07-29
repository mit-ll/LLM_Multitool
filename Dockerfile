FROM ubuntu:23.10

MAINTAINER Darrell Ricke <Darrell.Ricke@ll.mit.edu>

################################################################################
# Author: Darrell O. Ricke, Ph.D.  (email: Darrell.Ricke@ll.mit.edu)
#
# RAMS request ID  1026639
# 
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
# This material is based upon work supported by the Department of the Air Force 
# under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, 
# conclusions or recommendations expressed in this material are those of the 
# author(s) and do not necessarily reflect the views of the Department of the 
# Air Force.
# 
# © 2024 Massachusetts Institute of Technology.
# 
# The software/firmware is provided to you on an As-ls basis
# Delivered to the U.S. Government with Unlimited Rights, as defined in 
# DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright 
# notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 
# or DFARS 252.227-7014 as detailed above. Use of this work other than as 
# specifically authorized by the U.S. Government may violate any copyrights that 
# exist in this work.
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
################################################################################

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8

RUN apt-get update && apt-get install -y build-essential coreutils \
    wget bzip2 git g++ gfortran libreadline6-dev libncurses5-dev xorg-dev libpng-dev libbz2-dev \
    liblzma-dev libpcre3-dev make libcairo2-dev libgtk2.0-dev \
    locales libcurl4-nss-dev \
    language-pack-en language-pack-en-base \
    git curl unzip bc tabix \
    libssl-dev libgit2-dev libssh2-1-dev \
    gcc zip \
    python3.11 gcc zip python3-dev \
    zlib1g-dev libbz2-dev liblzma-dev pigz libncurses5-dev \
    libreadline-dev \
    openssl \
    gnupg2 \
    libmysqlclient-dev \
    nodejs \
    sqlite3 \
    ruby-full rubygems vim libyaml-dev libsqlite3-dev default-jre

RUN mkdir /S

RUN gem install bundler \
    && gem install mysql2 \
    && gem install sqlite3 \
    && gem install rake \
    && gem install tzinfo-data \
    && gem install rails

WORKDIR /S
RUN curl https://bootstrap.pypa.io/pip/3.6/get-pip.py -o get-pip.py \
    && python3 get-pip.py

COPY dependencies/wgetrc /etc
WORKDIR /S
# RUN cd /S && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /S/miniconda.sh
# RUN bash /S/miniconda.sh -b -p /S/miniconda/
# ENV PATH="/S/miniconda/bin/:$PATH"
# RUN export PATH="/S/miniconda/bin/:$PATH"

# RUN conda create -n llama2
# RUN conda init bash
# RUN conda activate llama2
COPY dependencies/llms_requirements.txt /S
RUN pip install -r llms_requirements.txt \
    && pip install llama-cpp-python \
    && pip install fastapi uvicorn sse-starlette requests \
    && pip install transformers \
    && pip install gradio \
    && pip install langchain \
    && pip install "langserve[all]" \
    && pip install langchain_openai \
    && pip install langchainhub \
    && pip install langgraph \
    && pip install scipy \
    && pip install einops \
    && pip install bitsandbytes \
    && pip install accelerate

RUN pip install numba

RUN git clone https://github.com/facebookresearch/llama.git
WORKDIR /S/llama
RUN pip install .
WORKDIR /S
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    apt-get update
RUN apt-get install -y nvidia-container-toolkit

ENV HF_HOME=/io
ENV HF_HUB_CACHE=/io/hub
ENV HF_ASSETS_CACHE=/io/assets

# Crystal LLM is being released:
# COPY crystal_llm.tar /S
# WORKDIR /S
# RUN tar -xf crystal_llm.tar
# WORKDIR /S/crystal_llm
# RUN bundle update

WORKDIR /io

COPY dependencies/entrypoint.sh /usr/bin
RUN chmod +x /usr/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]
EXPOSE 3000
EXPOSE 7860
EXPOSE 8888
