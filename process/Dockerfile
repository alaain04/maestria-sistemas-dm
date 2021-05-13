FROM python:3.9-slim-buster
ADD . app
WORKDIR /app
RUN apt update && apt-get --assume-yes install gcc libc-dev g++
RUN pip3 install -r requirements
CMD ["python3", "-u","server.py"]