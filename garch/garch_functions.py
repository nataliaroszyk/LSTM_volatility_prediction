import pandas as pd
import numpy as np
from arch import arch_model

def garch_data_prep(df):

    """ Prepare data for GARCH model prediction."""
 
    # Ensure 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate daily log returns
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Calculate rolling volatility using log returns
    rolling_window_size = 22
    df['volatility'] = df['log_returns'].rolling(window=rolling_window_size).std()

    # Scale log returns for model input
    df['scaled_log_returns'] = df['log_returns'] * 100

    # Remove any rows with NaN values which are the result of rolling calculations
    df.dropna(inplace=True)

    # Filter data to start from a specific date, assuming you want to start from 1985
    df = df[df.Date >= '1985-01-01']

    return df

def train_garch_model(data, start_date):
    """ Train GARCH model and predict future volatility based on historical log returns. """
    train_data = data[data['Date'] < pd.to_datetime(start_date)]
    train_size = len(train_data)

    rolling_predictions = []
    for i in range(train_size, len(data)):
        train = data['scaled_log_returns'][i - train_size:i]
        model = arch_model(train, vol='Garch', p=2, q=2)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=1, reindex=False)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))

    predictions_df = pd.DataFrame(rolling_predictions, index=data['Date'][train_size:])
    predictions_df.columns = ['prediction']
    predictions_df['prediction'] = predictions_df['prediction'] / 100
    return predictions_df
