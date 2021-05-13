FROM python:3.9-slim-buster
ADD . app
WORKDIR /app
RUN pip3 install -r requirements
EXPOSE 8050
CMD ["python3", "-u","dash-server.py"]
