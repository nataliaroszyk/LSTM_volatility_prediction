import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

garch = pd.read_excel('results/results_garch.xlsx')
lstm = pd.read_excel('results/results_lstm.xlsx')
lstm_garch = pd.read_excel('results/results_lstm_garch.xlsx')
lstm_garch_vix = pd.read_excel('results/results_lstm_garch_vix.xlsx')
lstm_garch_vix_pct_change = pd.read_excel('results/results_lstm_garch_vix_pct_change.xlsx')
lstm_garch_vix_1_layer = pd.read_excel('results/results_lstm_garch_vix_1_layer.xlsx')
lstm_garch_vix_3_layers = pd.read_excel('results/results_lstm_garch_vix_3_layers.xlsx')
lstm_garch_vix_lookback_5 = pd.read_excel('results/results_lstm_garch_vix_lookback_5.xlsx')
lstm_garch_vix_lookback_66 = pd.read_excel('results/results_lstm_garch_vix_lookback_66.xlsx')
lstm_garch_vix_mae_loss = pd.read_excel('results/results_lstm_garch_vix_mae_loss.xlsx')
lstm_garch_vix_relu = pd.read_excel('results/results_lstm_garch_vix_relu.xlsx')

garch = garch.loc[garch['Date'] >= '2015-02-13', :]
print(f"DataFrame: {'garch'} | MAE : {mean_absolute_error(garch.actual, garch.prediction)} | RMSE: {np.sqrt(mean_squared_error(garch.actual, garch.prediction))}") 

lstm_dfs = {
    'lstm' : lstm,
    'lstm_garch' : lstm_garch,
    'lstm_garch_vix' : lstm_garch_vix,
    'lstm_garch_vix_pct_change': lstm_garch_vix_pct_change,
    'lstm_garch_vix_1_layer' : lstm_garch_vix_1_layer,
    'lstm_garch_vix_3_layers': lstm_garch_vix_3_layers,
    'lstm_garch_vix_lookback_5': lstm_garch_vix_lookback_5,
    'lstm_garch_vix_lookback_66': lstm_garch_vix_lookback_66,
    'lstm_garch_vix_mae_loss': lstm_garch_vix_mae_loss,
    'lstm_garch_vix_reluu': lstm_garch_vix_relu
}

for name, df in lstm_dfs.items():
    df.dropna(axis = 0, inplace = True)

    mae = mean_absolute_error(df.actual, df.prediction)
    rmse = np.sqrt(mean_squared_error(df.actual, df.prediction))
    print(f"DataFrame: {name} | MAE : {mae} | RMSE: {rmse}")


#----- Plots-----#
import matplotlib.pyplot as plt

def plot_data(dataframes, labels, colors, linestyles, title, x_label, y_label):

    plt.figure(figsize=(12, 6))

    for df, label, color, linestyle in zip(dataframes, labels, colors, linestyles):
        plt.plot(df['Date'], df[df.columns[1]], label=label, color=color, linestyle=linestyle)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.show()


# GARCH
plot_data(
    dataframes=[garch[['Date','prediction']], garch[['Date','actual']]],
    labels=['Predicted Volatility', 'Actual Values'],
    colors=['royalblue', 'red'],
    linestyles=['-', '--'],
    title='GARCH Model Rolling Window Predictions vs. Actual Values',
    x_label='Date',
    y_label='Values'
)

# LSTM
plot_data(
    dataframes=[lstm[['Date','prediction']], lstm[['Date','actual']]],
    labels=['LSTM', 'Actual Values'],
    colors=['gold', 'red'],
    linestyles=['-', '--'],
    title='LSTM vs. Actual Values',
    x_label='Date',
    y_label='Values'
)


# LSTM- GARCH
plot_data(
    dataframes=[lstm_garch[['Date','prediction']], lstm_garch[['Date','actual']]],
    labels=['LSTM-GARCH', 'Actual Values'],
    colors=['deepskyblue', 'red'],
    linestyles=['-', '--'],
    title='LSTM-GARCH vs. Actual Values',
    x_label='Date',
    y_label='Values'
)

# LSTM- GARCH with VIX input
plot_data(
    dataframes=[lstm_garch_vix[['Date','prediction']], lstm_garch_vix[['Date','actual']]],
    labels=['LSTM-GARCH with VIX Input', 'Actual Values'],
    colors=['springgreen', 'red'],
    linestyles=['-', '--'],
    title='LSTM-GARCH with VIX input vs. Actual Values',
    x_label='Date',
    y_label='Values'
)

#----- Sensitivity -----#
# MAE vs. MSE
plot_data(
    dataframes=[lstm_garch_vix_mae_loss[['Date','prediction']],lstm_garch_vix[['Date','prediction']], lstm_garch_vix[['Date','actual']]],
    labels=['MAE loss function', 'MSE loss function', 'Actual Values'],
    colors=['purple','springgreen', 'red'],
    linestyles=['-','-', '--'],
    title='LSTM-GARCH with VIX input (MSE and MAE loss function) vs. Actual Values',
    x_label='Date',
    y_label='Values'
)

# Log returns vs. Pct change
plot_data(
    dataframes=[lstm_garch_vix[['Date','prediction']],lstm_garch_vix_pct_change[['Date','prediction']], lstm_garch_vix[['Date','actual']]],
    labels=['Log Returns Input', 'Percentage Change in Price Input', 'Actual Values'],
    colors=['springgreen','gold', 'red'],
    linestyles=['-','-', '--'],
    title='LSTM-GARCH with VIX input (Log Returns and Percentage Change in Price Input) vs. Actual Values',
    x_label='Date',
    y_label='Values'
)

# Sequence length 5, 22 vs. 66 days
plot_data(
    dataframes=[lstm_garch_vix_lookback_66[['Date','prediction']],lstm_garch_vix[['Date','prediction']], lstm_garch_vix_lookback_5[['Date','prediction']], lstm_garch_vix[['Date','actual']]],
    labels=['Sequence of 66 days', 'Sequence of 22 days', 'Sequence of 5 days', 'Actual Values'],
    colors=['purple', 'springgreen','gold', 'red'],
    linestyles=['-','-', '-', '--'],
    title='LSTM-GARCH with VIX input (lookback 5, 22 and 66 days) vs. Actual Values',
    x_label='Date',
    y_label='Values'
)

# Number of LSTM layers 1, 2 vs 3
plot_data(
    dataframes=[lstm_garch_vix_3_layers[['Date','prediction']],lstm_garch_vix[['Date','prediction']], lstm_garch_vix_1_layer[['Date','prediction']], lstm_garch_vix[['Date','actual']]],
    labels=['3 LSTM layers', '2 LSTM layers', '1 LSTM layer', 'Actual Values'],
    colors=['purple', 'springgreen','gold', 'red'],
    linestyles=['-','-', '-', '--'],
    title='LSTM-GARCH with VIX input (2- Layer vs. 1- and 3- Layer ) vs. Actual Values',
    x_label='Date',
    y_label='Values'
)

# Activation function ReLU
plot_data(
    dataframes=[lstm_garch_vix_relu[['Date','prediction']],lstm_garch_vix[['Date','prediction']], lstm_garch_vix[['Date','actual']]],
    labels=['ReLU/ReLU Activation Function', 'Tanh/ Tanh Activation Function', 'Actual Values'],
    colors=['gold', 'springgreen', 'red'],
    linestyles=['-','-', '-', '--'],
    title='LSTM-GARCH with VIX input (ReLU vs. Tanh Activation Function) vs. Actual Values',
    x_label='Date',
    y_label='Values'
)