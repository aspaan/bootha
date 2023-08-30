FROM python:3.10
COPY requirements.txt /app/
WORKDIR /app

RUN mkdir -p /model

# Download the file using curl
RUN curl -o /model/GFPGANv1.4.pth https://geophotostorage.blob.core.windows.net/bootha-model/GFPGANv1.4.pth

RUN curl -o /model/inswapper_128.onnx https://geophotostorage.blob.core.windows.net/bootha-model/inswapper_128.onnx

RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'libsm6'\
    'libxext6'  -y
RUN pip install --upgrade pip

COPY . /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]



#CMD ["main.py"]
#