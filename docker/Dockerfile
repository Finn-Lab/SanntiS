FROM continuumio/miniconda3

WORKDIR /docker

COPY environment.yml .
COPY interproscan-5.52-86.0.NO_DATA.tar.bz2 .
RUN tar -xf interproscan-5.52-86.0.NO_DATA.tar.bz2 --directory /opt \
  && rm interproscan-5.52-86.0.NO_DATA.tar.bz2

RUN conda env create -f environment.yml
SHELL ["/bin/bash", "--login", "-c"]
#RUN echo "conda init bash" >> ~/.bashrc
RUN echo "conda activate bgc" >> ~/.bashrc

SHELL ["conda", "run", "-n", "bgc", "/bin/bash", "-c"]
ENV PATH="/opt/interproscan:${PATH}"

RUN python -c "import sanntis"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "bgc", "sanntis"]
