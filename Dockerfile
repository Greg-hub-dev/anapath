FROM python:3.10.6-buster

# Installer les dépendances système pour OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


COPY anapath /anapath
COPY anapath/frontend/requirements.txt /requirements.txt
COPY setup.py /setup.py


RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

#local
#CMD uvicorn anapath.api.fast:app --reload

#Distance
CMD uvicorn anapath.api.fast:app --host 0.0.0.0 --port $PORT
