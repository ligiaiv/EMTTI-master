# FROM python:3.6
# FROM tensorflow/tensorflow
FROM pytorch/pytorch

RUN pip3 install --upgrade pip
# RUN pip3 install tensorflow
# RUN echo pip -v
# RUN pip3 install torch==1.5.0
# RUN pip3 install torch
# RUN pip3 install torchvision==0.8.2+cpu
RUN pip3 install numpy==1.19.4
RUN pip3 install tensorflow==2.4.0 
RUN pip3 install pyyaml
# RUN pip3 install keras --no-deps 
RUN pip3 install pandas==1.2.0
RUN pip3 install transformers==4.15.0
RUN pip3 install datasets==1.16.1
RUN pip3 install -U scikit-learn==0.24.0

# RUN pip3 install scikit-learn
# RUN pip3 install matplotlib
# RUN pip3 install h5py
# RUN pip3 install bayesian-optimization


# RUN pip3 install nltk

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




