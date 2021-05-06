FROM python:3.9-slim-buster
ADD . app
WORKDIR /app
RUN pip3 install -r requirements
CMD ["python3", "consumer.py"]