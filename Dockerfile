FROM python:3.11.5-buster
WORKDIR /app
COPY . /app
RUN apt update -y && apt install awscli -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python3", "app.py"]