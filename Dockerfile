# start with a Python base image
FROM python:3.9

# maintainer info
LABEL maintainer='kagines1@up.edu.ph'

# initialize a work directory 
RUN mkdir -p /agent-api
WORKDIR /agent-api

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
# upgrade pip with no cache
RUN pip install --no-cache-dir -U pip
# place the requirements.txt file into the work directory
COPY requirements.txt .
# install package dependencies enumerated in the requirements file into the container (no cache dir allows the packages to not be saved locally)
# RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install -r requirements.txt
# copy the directory that contains all the code for the container, recommended that this is placed at the end to optimize container image build times
COPY . .
ENV PATH=$PATH:/agent-api/app
ENV PYTHONPATH /agent-api/app
# the commands (similar to what you would enter in the cmd) to run the app that you placed in the work directory
CMD python app/main.py
# CMD uvicorn app.main:app --host 0.0.0.0 --port 8000