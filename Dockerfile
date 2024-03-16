FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install ./dist/ml_spaceship-0.1.0-py3-none-any.whl

CMD ["python3", "./ml_spaceship/flask_app.py"]