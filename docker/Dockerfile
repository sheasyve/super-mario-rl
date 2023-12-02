FROM python:3.10

WORKDIR /opt/mario

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    libgl1-mesa-glx && \
    pip install --upgrade pip

COPY requirements.txt /opt/mario/requirements.txt
COPY lib /opt/mario/lib

RUN pip install -r requirements.txt

CMD ["/bin/bash"]