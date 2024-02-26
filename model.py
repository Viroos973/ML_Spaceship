import fire
import numpy as np
import catboost as cb
import pandas as pd
import optuna
import os
import pickle
from sklearn.impute import KNNImputer

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

    #Забираем группу пассажиров из их id
    df['PassengerGroup'] = df['PassengerId'].apply( lambda x: x.split('_')[0]).astype(int)

    #Делим кабину на признаки: палуба, номер и сторона
    df['Cabin'].fillna('nann/-1/nann', inplace=True)

    df['Cabin_deck'] = df['Cabin'].apply( lambda x: x.split('/')[0])
    df['Cabin_num'] = df['Cabin'].apply( lambda x: x.split('/')[1]).astype(int)
    df['Cabin_side'] = df['Cabin'].apply( lambda x: x.split('/')[2])

    df['Cabin_deck'] = df['Cabin_deck'].replace('nann', np.nan)
    df['Cabin_num'] = df['Cabin_num'].replace(-1, np.nan)
    df['Cabin_side'] = df['Cabin_side'].replace('nann', np.nan)
    
    return df
    
def fillnaAndDrop(df):
    """
    Функция для заполнения пропущенных значений в датасете и удаления ненужных признаков.
    """

    #Заполняем категориальные признаки их модой
    for features in categorical_features:
        df[features].fillna(df[features].mode().iloc[0], inplace=True)

    #Заполняем численные признаки их медианой
    for features in regression_features:
        df[features].fillna(df[features].median(), inplace=True)

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
    
    return df
    
def encode_cat(df):
    """
    Функция для кодирования категориальных признаков в датасете.
    """

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
    return df

def kfold(X, y, k=5):
    """
    Функция для выполнения K-Fold разделения датасета на обучающие и валидационные наборы.
    """

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

    return result

class My_Classifier_Model(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.X_train = None
        self.y_train = None

    def optune_optimize(self, trial):
        """
        Функция для оптимизации параметров модели CatBoostClassifier с помощью библиотеки Optuna.
        """

        #Задаем диапозоны для параметров модели CatBoost
        param = {
            "depth": trial.suggest_int("depth", 1, 12),
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3)
        }

        #Создаем модель CatBoost с заданными параметрами
        model = cb.CatBoostClassifier(**param)
        accs = []

        #Обучаем модели, для каждой вычисляем точность и записавыем их в массив accs
        for ((X_train, y_train), (X_val, y_val)) in kfold(self.X_train, self.y_train, 10):
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            pred = pred == 'True'
            accs.append(np.mean(pred == y_val))

        #Вычисляем среднюю точность модели и возвращаем ее
        acc_mean = np.mean(accs)
        return acc_mean

    def train(self):
        """
        Метод для обучения модели на данных из заданного датасета и сохранения её артифактов.
        """

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
        model = cb.CatBoostClassifier(**study.best_params)
        model.fit(self.X_train, self.y_train)

        #Сохраняем получившуюся модель
        os.makedirs('./data/model/', exist_ok=True)
        with open("./data/model/model.pkl", 'wb') as file:
            pickle.dump(model, file)

        #TODO Доделать регистрацию модели

    def predict(self):
        """
        Метод для использования обученной модели для предсказания результатов на новых данных.
        """

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

if __name__ == '__main__':
    # Запуск модели через командную строку с помощью Fire
    fire.Fire(My_Classifier_Model)