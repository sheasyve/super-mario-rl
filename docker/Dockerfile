FROM python:3.10

WORKDIR /opt/mario

RUN pip install --upgrade pip

COPY requirements.txt /opt/mario/requirements.txt
COPY lib /opt/mario/lib

RUN pip install -r requirements.txt

CMD ["/bin/bash"]