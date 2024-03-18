# Обо мне
ФИО: Елисеев Юрий Германович<br/>
Группа: 972201
# Как пользоваться с docker
- Скачайте проект с репозитория
- Создайте на вашем диске "С" директорию "data", в которую поместите ваши train.csv и test.csv
- Откройте терминал
    - Если у вас Linux или Mac, то нет проблем
    - Если у вас Windows, то используйте WSL2
- Зайдите в корневой файл установленного репозитория
- Введите `foo@bar:~$ docker compose up`
- Для обучения модели отправьте POST запрос - http://0.0.0.0:5000/train?dataset=/app/data/path/to/train.cs
  - Вместо `path/to/train.cs` укажите путь к `train.csv` из директории `/c/data/`. Например: если `train.csv` лежит у вас на пути `/c/data/train.csv`, то укажите путь `/app/data/train.cs`
  - После обучения, обученная модель появится в папке `/c/data/`
- Для предсказаний модели отправьте POST запрос http://0.0.0.0:5000/predict?dataset=/app/data/path/to/test.csv
  - Вместо `path/to/test.cs` укажите путь к `test.csv` из директории `/c/data/`. Например: если `test.csv` лежит у вас на пути `/c/data/test.csv`, то укажите путь `/app/data/test.cs`
  - После выполнения запроса, результат предсказания появится в папке `/c/data/`
- В папке `/c/data/` появится файл `log_file.log` внутри которого можно будет увидеть как проходило обучение и предсказние и среднюю точность модели
# Как пользоваться без docker
- Скачайте проект с репозитория
- Создайте директорию, в которую поместите ваши train.csv и test.csv
- Откройте терминал
- Зайдите в корневой файл установленного репозитория
- Введите `foo@bar:~$ pip install ml_spaceship-0.1.0-py3-none-any.whl`
- Предсказать и обучить модель можно через flask
  - Запустите файл flask_app.py
  - Для обучения модели отправьте POST запрос - http://0.0.0.0:5000/train?dataset=/path/to/train.cs
    - Вместо `path/to/train.cs` укажите путь к `train.csv`. Например: если `train.csv` лежит у вас на пути `/c/data/train.csv`, то укажите путь `/c/data/train.csv
    - После обучения, обученная модель появится в папке `./data/` внутри корневого файла установленного репозитория
  - Для предсказаний модели отправьте POST запрос http://0.0.0.0:5000/predict?dataset=/path/to/test.csv
    - Вместо `path/to/test.cs` укажите путь к `test.csv`. Например: если `test.csv` лежит у вас на пути `/c/data/test.csv`, то укажите путь `/c/data/test.csv`
    - После выполнения запроса, результат предсказания появится в папке `./data/` внутри корневого файла установленного репозитория
- Предсказать и обучить модель можно через командную строку
  - Перед обучением и предсказанием модели необходимо перейти в папку `ml_spaceship`
  - Для обучения модели введите в командную строку `foo@bar:~$ python model.py train --dataset=/path/to/train.csv`, `/path/to/train.csv` заменять аналогично flask
  - Для предсказаний модели введите в командную строку `foo@bar:~$ python model.py predict --dataset=/path/to/test.csv`, `/path/to/test.csv` заменять аналогично flask
- В папке `./data/` появится файл `log_file.log` внутри которого можно будет увидеть как проходило обучение и предсказние и среднюю точность модели
# Чем пользовался
- [python = "3.10.12"](https://www.python.org/)
- WSL2
- Пакеты
  - [fire = "0.5.0"](https://google.github.io/python-fire/guide/)
  - [numpy = "1.26.4"](https://numpy.org/)
  - [catboost = "1.2.2"](https://catboost.ai/)
  - [pandas = "2.2.0"](https://khashtamov.com/ru/pandas-introduction/)
  - [optuna = "3.5.0"](https://optuna.org/)
  - [scikit-learn = "1.2.2"](https://scikit-learn.org/stable/)
  - [Flask = "3.0.2"](https://flask.palletsprojects.com/en/3.0.x/)
  - [poetry](https://python-poetry.org/docs/)
- Расширения
  - Docker
  - Pylance
  - Python
  - Python Debugger
- IDE
  - [VS code](https://code.visualstudio.com/)
  - [Anaconda](https://www.anaconda.com/)
  - [Jupyter notebook](https://jupyter.org/)
