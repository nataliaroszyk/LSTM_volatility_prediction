import pandas as pd
import numpy as np

sp = pd.read_excel('data/sp500.xlsx')

rolling_window_size = 22

# Calculate daily log returns
sp['log_returns'] = np.log(sp['Close'] / sp['Close'].shift(1))

# Calculate volatility as the std of daily log returns
sp['volatility'] = sp['log_returns'].rolling(window=rolling_window_size).std()

# Add lagged volatility
lag_days = 1
for i in range(1, lag_days + 1):
    sp[f'lagged_volatility_{i}'] = sp['volatility'].shift(i)

sp.dropna(inplace=True)

# Clean dataset
sp.Date = pd.to_datetime(sp.Date)
sp.dropna(inplace = True)
sp.drop(columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], inplace = True)

# Filter for dates >= 2000 year
sp_2000 = sp[sp.Date >= '01-01-2000']

# Save
sp_2000.to_excel('data/sp500_lstm.xlsx')