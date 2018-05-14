FROM wqael/notebooks:conda2-py2-pytorch

RUN apt-get -qq update \
    && apt-get -qq install -y \
      build-essential \
      libhdf5-dev \
    && apt-get -qq clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip --no-cache-dir install -q -U \
      lmdb

RUN conda install h5py protobuf -c anaconda --quiet && \
    conda install visdom -c conda-forge --quiet && \
    conda clean -ya --quiet

# IPython
EXPOSE 8888

WORKDIR /notebooks
