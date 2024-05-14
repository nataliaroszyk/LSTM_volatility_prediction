import pandas as pd
import numpy as np
from arch import arch_model
from garch.garch_functions import garch_data_prep, train_garch_model

# Download the dataset
df = pd.read_excel('data/sp500.xlsx')
sp = garch_data_prep(df)

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
