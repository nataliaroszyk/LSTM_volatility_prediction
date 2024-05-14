import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch

# Load your data
df = pd.read_excel('data/sp500_lstm.xlsx')

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter for dates >= 2000 year
df = df[df.Date >= '2000-01-01']

# Setup features and target column
feature_columns = [col for col in df.columns if col not in ['volatility', 'Date']]
target_column = 'volatility'

# Convert to LSTM-friendly format
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 22  # Sequence length
X, y = create_dataset(df[feature_columns], df[target_column].values.reshape(-1, 1), time_steps) # [samples, time steps, features]
input_shape = (time_steps, X.shape[2])

index_15_years = 252*15

# Hyperparameter tuning on the first 15 years of data
tune_X, tune_y = X[:index_15_years], y[:index_15_years]

# Modified create_model function to incorporate hyperparameters
def build_model(hp):
    model = Sequential()
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(LSTM(units=hp.Choice('units_lstm_' + str(i), [32,64,128]),
                       activation=hp.Choice('activation_' + str(i), ['tanh', 'relu']),
                       return_sequences=True if i < hp.get('num_layers') - 1 else False,
                       input_shape=input_shape))
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.3, step=0.1)))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', [0.01, 0.001, 0.0001])),
                  loss=hp.Choice('loss', ['mean_squared_error', 'mean_absolute_error']))
    return model

# Configure the tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=50,
    executions_per_trial=3,
    directory='model_tuning',
    project_name='LSTM_Tuning'
)

# Define your early stopping callback
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Start hyperparameter search
tuner.search(tune_X, tune_y, epochs=50, validation_split=0.6, callbacks=[early_stopping], verbose = 1)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best hyperparameters: {best_hps.values}")
