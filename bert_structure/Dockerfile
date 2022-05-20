# FROM python:3.6
# FROM tensorflow/tensorflow
FROM nvidia/cuda:11.0-base
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
# FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
#  orch/pytorch:1.7.0-cuda11.0-cudnn8-devel
# RUN python -m pip install --upgrade pip

RUN pip install --upgrade pip
# RUN apt install socat --> pra mandar pra tcp
# RUN pip install tensorflow
# RUN echo pip -v
# RUN pip install torch==1.5.0
# RUN pip install torch
# RUN pip install torchvision==0.8.2+cpu
RUN pip install numpy==1.19.4
RUN pip install tensorflow==2.4.0 
RUN pip install pyyaml
# RUN pip install keras --no-deps 
RUN pip install pandas==1.2.0
RUN pip install transformers==4.15.0
RUN pip install datasets==1.16.1
RUN pip install -U scikit-learn==0.24.0


# RUN pip install torchtext==0.10.1

RUN pip install spacy==3.2.4
RUN pip install matplotlib==3.4.3
RUN pip install pydot==1.4.2
RUN pip install graphviz==0.20
RUN python3 -m spacy download pt_core_news_sm
# RUN pip install scikit-learn
# RUN pip install matplotlib
# RUN pip install h5py
# RUN pip install bayesian-optimization


# RUN pip install nltk

COPY prepare.py /workspace/prepare.py
USER root
# RUN ["chmod", "+x", "opt/source-code/commands.sh"]
RUN python3 /workspace/prepare.py

# ENTRYPOINT /bin/bash opt/source-code/commands.sh teste-gcn
#ENTRYPOINT /bin/bash opt/source-code/Simple/commands.sh
ENTRYPOINT ["/workspace/commands.sh"]

# COPY teste1/.vector_cache /opt/source-code/teste1/.vector_cache
# COPY teste1/*.py /opt/source-code/teste1/
# COPY Datasets /opt/source-code/Datasets
# COPY Embeddings /opt/source-code/Embeddings
# COPY teste1/variables.json /opt/source-code/teste1/


##ADD requirements.txt

#RUN pip3 install -r opt/source-code/requirements.txt

# RUN ls
# RUN ls opt/
# RUN ls opt/source-code/




