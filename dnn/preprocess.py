"""
Data Preprocessing

Author: Anjum Azra
"""

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:

    def __init__(self, csv_file):

        self.csv_file = csv_file

    def process(self):

        df = pd.read_csv(self.csv_file)

        # Remove impossible engagements
        df = df[df["Rmax"] > 0]

        X = df.drop(columns=["Rmax"])

        y = df["Rmax"]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
        )

        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)

        X_valid = scaler.transform(X_valid)

        X_test = scaler.transform(X_test)

        joblib.dump(
            scaler,
            "models/scaler.pkl",
        )

        return (

            X_train,
            X_valid,
            X_test,

            y_train.values,
            y_valid.values,
            y_test.values,

        )