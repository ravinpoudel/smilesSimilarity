FROM continuumio/miniconda3:4.9.2


RUN apt-get --allow-releaseinfo-change -y update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "--login", "-c"]

# Update Miniconda
RUN conda update -n base conda && \
    conda install -y -c conda-forge -c bioconda pandas mordred  scikit-learn scipy && \
    conda clean -afy

RUN pip install rdkit

ENV PATH="~/miniconda3/bin:$PATH"


RUN mkdir /data
WORKDIR /data
COPY sm.py /data/sm.py
COPY main.py /data/main.py
COPY README.md /data/README.md


ENTRYPOINT ["python", "main.py"]

#CMD bash

