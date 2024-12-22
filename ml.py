import os
from config import TRAIN_DATA_FOLDER_NAME, PREDICT_WINDOW_SIZE, TRAIN_STEP
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from features import FeatureExtractor
from utils import transform_data_to_time_series_format


class GestureRecognitionModel:
    def __init__(self, train_data_folder_name: str = TRAIN_DATA_FOLDER_NAME):
        self.train_data_folder_name = train_data_folder_name
        self.model = None

    def train(self):
        model = None
        x, y = self.__handle_train_data_file()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        model.fit(x_train, y_train)

        score = model.score(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)

        print("train accuracy: %.2f%%" % (100 * score))
        print('test  accuracy: %.2f%%' % (100 * accuracy))
        print("=" * 50)
        print(model)
        print("=" * 50)
        self.model = model

    def predict(self, x):
        data = np.array(x)
        fe = FeatureExtractor(data=data)
        return self.model.predict(fe.get_features().reshape(1, -1))

    def __handle_train_data_file(self):
        return self.__generate_features_based_data()

    def __generate_time_series_based_data(self):
        labels = []
        tmp_df = []
        instance_id = 0

        for filename in os.listdir(f"./{self.train_data_folder_name}"):
            if not filename.endswith(".csv"):
                continue

            # handle label
            lab = filename.split("_")[0]

            # handle data
            df_chunk = pd.read_csv(f"./{self.train_data_folder_name}/{filename}", header=None)

            # split data
            blocks = self.__split_dataframe(df=df_chunk)

            # handle data
            for df in blocks:
                tmp_df.append(transform_data_to_time_series_format(df=df, instance_id=instance_id))
                labels.append(lab)
                instance_id += 1

        data = pd.concat(tmp_df)
        return data, np.array(labels)

    def __generate_features_based_data(self):
        labels = []
        features = []

        for filename in os.listdir(f"./{self.train_data_folder_name}"):
            if not filename.endswith(".csv"):
                continue

            # handle label
            lab = filename.split("_")[0]

            # handle data
            df_chunk = pd.read_csv(f"./{self.train_data_folder_name}/{filename}", header=None)

            # split data
            blocks = self.__split_dataframe(df=df_chunk)

            # handle data
            for df in blocks:
                fe = FeatureExtractor(data=df.to_numpy())
                features.append(fe.get_features())
                labels.append(lab)

        return np.array(features), np.array(labels)

    @staticmethod
    def __split_dataframe(df: pd.DataFrame, block_size=PREDICT_WINDOW_SIZE, step=TRAIN_STEP):
        blocks = []
        start = 0
        while start + block_size <= len(df):
            block = df.iloc[start:start + block_size]
            blocks.append(block)
            start += step

        return blocks
