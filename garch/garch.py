import pandas as pd
import numpy as np
from arch import arch_model

# Download the dataset
sp = pd.read_excel('data/sp500.xlsx')
sp['Date'] = pd.to_datetime(sp['Date'])  # Ensure 'Date' is in datetime format

# Calculate daily log returns
sp['log_returns'] = np.log(sp['Close'] / sp['Close'].shift(1))

# Calculate rolling volatility using log returns
rolling_window_size = 22
sp['volatility'] = sp['log_returns'].rolling(window=rolling_window_size).std()

# Scale log returns for model input
sp['scaled_log_returns'] = sp['log_returns'] * 100
sp.dropna(inplace=True)

# Filter data starting from 1985 to use the first 15 years for training
sp = sp[sp.Date >= '1985-01-01']

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

predicted_volatility = train_garch_model(sp, '2000-01-03')

# Prepare actual and predicted volatility for comparison
actual_volatility = sp[['Date', 'volatility']].set_index('Date')
actual_volatility = actual_volatility.rename(columns={'volatility': 'actual'})

# Merge actual and predicted data
merged_df = sp.merge(actual_volatility, on='Date', how='outer').merge(predicted_volatility, on='Date', how='outer')
merged_df.dropna(inplace = True)
garch_results = merged_df.drop(columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume','scaled_log_returns', 'volatility'])
print(garch_results)

# Save results
garch_results.to_excel('results/garch_results.xlsx')
