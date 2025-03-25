FROM python:3.10.6-buster

# Installer les dépendances système pour OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*



COPY setup.py /setup.py
COPY wagon-bootcamp-448714-f262f727bd9d.json /googleapp/wagon-bootcamp-448714-f262f727bd9d.json
COPY requirements_prod.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
#RUN pip install -e .
RUN mkdir -p model

COPY anapath /anapath


#local
CMD uvicorn anapath.api.fast_phikon:app --reload --host 0.0.0.0 --port $PORT
