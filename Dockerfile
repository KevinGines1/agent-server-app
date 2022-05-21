# start with a Python base image
FROM python:3.9
ENV PYTHONUNBUFFEREED 1
EXPOSE 8000
# initialize a work directory 
WORKDIR /code
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
# place the requirements.txt file into the work directory
COPY ./requirements.txt /code/requirements.txt
# install package dependencies enumerated in the requirements file into the container (no cache dir allows the packages to not be saved locally)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# copy the directory that contains all the code for the container, recommended that this is placed at the end to optimize container image build times
COPY ./app/ /code/app/
ENV PATH=$PATH:/code/app
ENV PYTHONPATH /code/app
# the commands (similar to what you would enter in the cmd) to run the app that you placed in the work directory
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--reload"]
# CMD uvicorn app.main:app --host 0.0.0.0 --port 8000