FROM python:3.10

RUN apt-get update && \
    apt-get upgrade && \
    apt-get install -y \
    freeglut3-dev && \
    pip install --upgrade pip && \
    pip install \
    gym_super_mario_bros \
    matplotlib \
    nes_py \
    pyglet \
    stable_baselines3

WORKDIR /opt/mario/src

CMD ["/bin/bash"]