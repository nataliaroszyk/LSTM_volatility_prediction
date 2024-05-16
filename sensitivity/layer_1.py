import pandas as pd

from garch.garch_functions import train_garch_model, garch_data_prep
from sensitivity.sensitivity_model_function import create_model_sensitivity
from lstm.LSTM import create_dataset, should_retrain

import os
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Load the datasets
# sp_lstm data
sp_lstm = pd.read_excel('data/sp500_lstm.xlsx')
# sp_garch
sp = pd.read_excel('data/sp500.xlsx')
#vix
vix = pd.read_excel('data/vix.xlsx')
vix['Date'] = pd.to_datetime(vix['Date'])

#----- VIX -----#
vix = vix.rename(columns={'Close': 'Close_vix'})
vix.drop(columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace = True)

#----- GARCH -----#
sp_garch = garch_data_prep(sp)
garch_results = train_garch_model(sp_garch, '2000-01-03')

# Merge with lstm and VIX dataset
sp_garch = pd.merge(sp_lstm, garch_results, on=['Date'], how='left')
sp_garch = sp_garch.rename(columns={'prediction': 'predicted_volatility_garch'}) 

sp_garch_vix = pd.merge(sp_garch, vix, on=['Date'], how='left')

df = sp_garch_vix # Dataset for LSTM-GARCH with VIX input model
#df.to_excel('data/sp500_lstm_garch_vix.xlsx')

#----- LSTM- GARCH with VIX input -----#
# Setup features and target column
feature_columns = [col for col in df.columns if col not in ['volatility', 'Date']]
target_column = 'volatility'

time_steps = 22  # Sequence length
X, y = create_dataset(df[feature_columns], df[target_column].values.reshape(-1, 1), time_steps) # [samples, time steps, features]

# Set up for predictions
input_shape = (time_steps, X.shape[2])
model_save_path = 'lstm_garch_vix_layer_1.h5'

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

initial_train_size = 12 * 252
validation_size = 3 * 252

results = []
counter = 0

# Walk forward prediction with model refitting every 252 days
for i in range(len(df) - initial_train_size - validation_size - 1):
    # Check if there is enough data for the test set to form a complete sequence
    if (i + initial_train_size + validation_size + time_steps > len(X)):
        print("Not enough data to form a complete sequence for testing. Ending predictions.")
        break  # Exit the loop if there isn't enough data left for a full sequence

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit scaler on current training data
    scaler_X.fit(X[i:i+initial_train_size].reshape(-1, X.shape[2]))
    scaler_y.fit(y[i:i+initial_train_size].reshape(-1, 1))

    # Transform train, validation, and test data
    train_X = scaler_X.transform(X[i:i+initial_train_size].reshape(-1, X.shape[2])).reshape(-1, time_steps, X.shape[2])
    train_y = scaler_y.transform(y[i:i+initial_train_size].reshape(-1, 1)).reshape(-1, 1)
    val_X = scaler_X.transform(X[i+initial_train_size:i+initial_train_size+validation_size].reshape(-1, X.shape[2])).reshape(-1, time_steps, X.shape[2])
    val_y = scaler_y.transform(y[i+initial_train_size:i+initial_train_size+validation_size].reshape(-1, 1)).reshape(-1, 1)
    test_X = scaler_X.transform(X[i+initial_train_size+validation_size:i+initial_train_size+validation_size+1].reshape(-1, X.shape[2])).reshape(-1, time_steps, X.shape[2])
    test_y = scaler_y.transform(y[i+initial_train_size+validation_size:i+initial_train_size+validation_size+1].reshape(-1, 1)).reshape(-1, 1)

    if should_retrain(counter) or not os.path.exists(model_save_path):
        model = create_model_sensitivity(input_shape, lstm_layers=1)
        model.fit(train_X, train_y, epochs=100, batch_size=64, validation_data=(val_X, val_y), verbose=0, callbacks=[early_stopping])
        model.save_weights(model_save_path)
    else:
        model = create_model_sensitivity(input_shape, lstm_layers=1)
        model.load_weights(model_save_path)
        model.fit(train_X[-1].reshape(1, *train_X[-1].shape), train_y[-1].reshape(1, 1), epochs=1, verbose=0)

    predicted = model.predict(test_X)
    predicted = scaler_y.inverse_transform(predicted.reshape(-1, 1))
    actual = scaler_y.inverse_transform(test_y.reshape(-1, 1))
    mae = mean_absolute_error(actual, predicted)

        # Print current results
    current_result = {
        'train_start': df['Date'][i+time_steps],
        'train_end': df['Date'][i+initial_train_size+time_steps-1],
        'validation_start': df['Date'][i+initial_train_size+time_steps],
        'validation_end': df['Date'][i+initial_train_size+validation_size+time_steps-1],
        'test_date': df['Date'][i+initial_train_size+validation_size+time_steps],
        'prediction': predicted.flatten()[0],
        'actual': actual.flatten()[0],
        'mae': mae
    }
    print(current_result)

    results.append(current_result)
    counter += 1

lstm_garch_vix_layers_1_results = pd.DataFrame(results)
lstm_garch_vix_layers_1_results.to_excel('results/results_lstm_garch_vix_1_layer.xlsx')




