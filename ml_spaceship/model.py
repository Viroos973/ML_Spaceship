import fire
import numpy as np
import catboost as cb
import pandas as pd
import optuna
import os
import pickle
import logging
from sklearn.impute import KNNImputer

#Настраиваем систему логирования в файл
logging.basicConfig(level=logging.INFO, filename="./data/log_file.log", filemode="w", format="%(asctime)s %(levelname)s %(message)s")

# Определение списков признаков
categorical_features = ['HomePlanet', 'Destination', 'Cabin_deck', 'Cabin_side']
regression_features = ['Age', 'Cabin_num']
bool_features = ['CryoSleep', 'VIP']
spending_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
drop_features = ['PassengerId', 'Cabin', 'Name']

def divisionFeatures(df):
    """
    Функция для деления признаков из столбца Cabin на отдельные признаки: палуба, номер и сторона
    А также для отделения группы пассажира от его id
    """
    try:
        logging.info("Beginning of dividing the Cabin and PassengerId attributes into subfeatures")

        #Забираем группу пассажиров из их id
        df['PassengerGroup'] = df['PassengerId'].apply( lambda x: x.split('_')[0]).astype(int)

        #Делим кабину на признаки: палуба, номер и сторона
        df['Cabin'] = df['Cabin'].fillna('nann/-1/nann')

        df['Cabin_deck'] = df['Cabin'].apply( lambda x: x.split('/')[0])
        df['Cabin_num'] = df['Cabin'].apply( lambda x: x.split('/')[1]).astype(int)
        df['Cabin_side'] = df['Cabin'].apply( lambda x: x.split('/')[2])

        df['Cabin_deck'] = df['Cabin_deck'].replace('nann', np.nan)
        df['Cabin_num'] = df['Cabin_num'].replace(-1, np.nan)
        df['Cabin_side'] = df['Cabin_side'].replace('nann', np.nan)

        logging.info("Dividing the Cabin and PassengerId attributes into subfeatures was successful")

        return df
    except Exception:
        logging.error("Something went wrong when dividing features into sub-features", exc_info=True)
    
def fillnaAndDrop(df):
    """
    Функция для заполнения пропущенных значений в датасете и удаления ненужных признаков.
    """

    try:
        logging.info("Start filling in gaps and removing unnecessary features")

        #Заполняем категориальные признаки их модой
        for features in categorical_features:
            df[features] = df[features].fillna(df[features].mode().iloc[0])

        #Заполняем численные признаки их медианой
        for features in regression_features:
            df[features] = df[features].fillna(df[features].median())

        #Переводим булевые признаки в числовые признаки
        for features in bool_features:
            df[features] = df[features].astype(float)

        #Удаляем ненужные признаки
        df = df.drop(columns=drop_features)
        
        #Заполняем булевые признаки и признаки с тратами пассажиров с помощью KNNImputer
        bool_imputer = KNNImputer(n_neighbors=1)
        spending_imputer = KNNImputer(n_neighbors=4)
        
        df[bool_features] = bool_imputer.fit_transform(df[bool_features])
        df[spending_features] = spending_imputer.fit_transform(df[spending_features])

        logging.info("Filling in the gaps and removing unnecessary features was successful")

        return df
    except Exception:
        logging.error("Something went wrong while filling in the blanks and deleting unnecessary features.", exc_info=True)
    
def encode_cat(df):
    """
    Функция для кодирования категориальных признаков в датасете.
    """

    try:
        logging.info("Beginning of coding of categorical features")

        for features in categorical_features:
            #Перебираем категориальные признаки и вытаскиваем их уникальные значения
            values = set(df[features])

            #Создаем новые булевые признаки из уникальных значений категориальных
            for v in values:
                df[features + '_' + v] = df[features] == v

            #Удаляем ненужные категориальные признаки
            df = df.drop(columns=features)

        #Удаляем признак 'Cabin_side_S', потому что у нас уже есть признак 'Cabin_side_P'
        df = df.drop(columns='Cabin_side_S')

        logging.info("Coding of categorical features was successful")

        return df
    except Exception:
        logging.error("Something went wrong while encoding categorical features", exc_info=True)

def kfold(X, y, k=5):
    """
    Функция для выполнения K-Fold разделения датасета на обучающие и валидационные наборы.
    """

    try:
        logging.info("Beginning of K-Fold splitting the dataset")

        #Создаем маску для деления Dataset
        pre_mask = np.arange(y.size) % k

        X_trains = []
        y_trains = []
        X_vals = []
        y_vals = []

        #Делим Dataset по маске на k частей
        for i in range(k):
            val_mask = pre_mask == i
            
            y_vals.append(y[val_mask])
            y_trains.append(y[~val_mask])
            X_vals.append(X[val_mask])
            X_trains.append(X[~val_mask])

        result = []

        #Записываем все в результат и возвращаем его
        for i in range(k):
            result.append(((X_trains[i], y_trains[i]), (X_vals[i], y_vals[i])))

        logging.info("K-Fold splitting of the dataset was successful")

        return result
    except Exception:
        logging.error("Something went wrong during K-Fold splitting the dataset", exc_info=True)

class My_Classifier_Model(object):
    def __init__(self, dataset):
        """
        Конструктор класса для инициализации модели с заданным путем датасета.
        """
        logging.info("Beginning of model initialization")

        self.dataset = dataset
        self.X_train = None
        self.y_train = None

        logging.info("Model initialization was successful")

    def optune_optimize(self, trial):
        """
        Функция для оптимизации параметров модели CatBoostClassifier с помощью библиотеки Optuna.
        """

        try:
            logging.info("Start optimizing the parameters of the CatBoostClassifier model")

            #Задаем диапозоны для параметров модели CatBoost
            param = {
                "depth": trial.suggest_int("depth", 1, 12),
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3)
            }

            #Создаем модель CatBoost с заданными параметрами
            model = cb.CatBoostClassifier(**param, silent=True)
            accs = []

            #Обучаем модели, для каждой вычисляем точность и записавыем их в массив accs
            for ((X_train, y_train), (X_val, y_val)) in kfold(self.X_train, self.y_train, 10):
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                pred = pred == 'True'
                accs.append(np.mean(pred == y_val))


            #Вычисляем среднюю точность модели и возвращаем ее
            acc_mean = np.mean(accs)

            logging.info("Optimization of model parameters was successful {}".format(acc_mean))

            return acc_mean
        except Exception:
            logging.error("Something went wrong while optimizing model parameters", exc_info=True)

    def train(self):
        """
        Метод для обучения модели на данных из заданного датасета и сохранения её артифактов.
        """

        try:
            logging.info("Starting model training")

            #Получаем Dataset для обучения модели
            df = pd.read_csv(self.dataset)
            #Настраиваем Dataset
            df = divisionFeatures(df)
            df = fillnaAndDrop(df)
            df = encode_cat(df)

            #Задаем целевой и обучающие признаки
            target = 'Transported'
            self.y_train = df[target]
            self.X_train = df.drop(columns = target).values

            #Оптимизируем параметры модели
            study = optuna.create_study(direction="maximize")
            study.optimize(self.optune_optimize, n_trials=10)

            #Обучаем модель с лучшими параметрами
            model = cb.CatBoostClassifier(**study.best_params, silent=True)
            model.fit(self.X_train, self.y_train)

            #Сохраняем получившуюся модель
            os.makedirs('./data/model/', exist_ok=True)
            with open("./data/model/model.pkl", 'wb') as file:
                pickle.dump(model, file)
                
            logging.info("Model training was successful with accuracy {:.2f}".format(study.best_value))
        except Exception:
            logging.error("Something went wrong while training the model", exc_info=True)

    def predict(self):
        """
        Метод для использования обученной модели для предсказания результатов на новых данных.
        """

        try:
            logging.info("Start predicting results on new data")

            #Получаем Dataset для предсказания целевых признаков
            df = pd.read_csv(self.dataset)

            #Создаем результат с id пассажира
            result = pd.DataFrame()
            result['PassengerId'] = df['PassengerId']

            #Настраиваем Dataset
            df = divisionFeatures(df)
            df = fillnaAndDrop(df)
            X_test = encode_cat(df)

            #Открываем обученную модель
            with open("./data/model/model.pkl", 'rb') as file:
                model = pickle.load(file)

            #предсказываем и записываем предсказания в рузультат
            pred = model.predict(X_test)
            result['Transported'] = pred

            #Сохраняем результат
            result.to_csv('./data/results.csv', index=False)

            logging.info("Prediction of results using new data was successful")
        except Exception:
            logging.error("Something went wrong when predicting results from new data", exc_info=True)

if __name__ == '__main__':
    try:
        logging.info("Launching the application")

        # Запуск модели через командную строку с помощью Fire
        fire.Fire(My_Classifier_Model)

        logging.info("Shutting down the application")
    except Exception:
        logging.error("Something went wrong", exc_info=True)