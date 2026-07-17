"""
WEZ Deep Neural Network Model

Author: Anjum Azra
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam


def build_model():

    model = Sequential(

        [

            Input(shape=(7,)),

            Dense(
                128,
                activation="relu",
            ),

            Dropout(0.20),

            Dense(
                64,
                activation="relu",
            ),

            Dropout(0.20),

            Dense(
                32,
                activation="relu",
            ),

            Dense(
                16,
                activation="relu",
            ),

            Dense(
                8,
                activation="relu",
            ),

            Dense(
                1,
                activation="linear",
            ),

        ]

    )

    model.compile(

        optimizer=Adam(
            learning_rate=0.001,
        ),

        loss="mse",

        metrics=["mae"],

    )

    model.summary()

    return model


if __name__ == "__main__":

    build_model()