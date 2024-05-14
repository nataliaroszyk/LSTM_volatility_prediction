import pandas as pd

from lstm.lstm_functions import create_dataset, should_retrain, create_model

import os
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Load sp_lstm data
df = pd.read_excel('data/sp500_lstm.xlsx')

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filter for dates >= 2000 year
df = df[df.Date >= '2000-01-01']

# Setup features and target column
feature_columns = [col for col in df.columns if col not in ['volatility', 'Date']]
target_column = 'volatility'

time_steps = 22  # Sequence length
X, y = create_dataset(df[feature_columns], df[target_column].values.reshape(-1, 1), time_steps) # [samples, time steps, features]

# Set random seeds for reproducibility
#os.environ['PYTHONHASHSEED'] = '3'
#random.seed(3)
#np.random.seed(3)
#tf.random.set_seed(3)

# Setup for predictions
input_shape = (time_steps, X.shape[2])
model_save_path = 'lstm.h5'

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
        model = create_model(input_shape)
        model.fit(train_X, train_y, epochs=100, batch_size=64, validation_data=(val_X, val_y), verbose=0, callbacks=[early_stopping])
        model.save_weights(model_save_path)
    else:
        model = create_model(input_shape)
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

lstm_results = pd.DataFrame(results)
lstm_results.to_excel('results/results_lstm.xlsx')
