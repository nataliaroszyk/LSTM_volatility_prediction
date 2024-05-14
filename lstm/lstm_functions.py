import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

# Convert to LSTM-friendly format
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Function to create the LSTM model
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='tanh', input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(LSTM(128, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Function to decide if retraining is needed
def should_retrain(counter, interval=252):
    return counter % interval == 0

