# LSTM volatility prediction

## Overview
This repository is part of a master thesis titled "The Hybrid Forecast of S&P 500 Volatility Ensembled from VIX, GARCH, and LSTM Models." It features a collection of models designed to predict stock market volatility using advanced econometric and machine learning techniques. The project includes GARCH, LSTM, LSTM-GARCH, and LSTM-GARCH with VIX input models, each leveraging time series data to understand and forecast market fluctuations. The focus is on combining traditional econometric methods with modern deep learning approaches to enhance the accuracy and robustness of volatility predictions.


## Project Structure
- `data/`: Directory containing the datasets used for training and testing the models.
- `garch/`: Contains scripts and resources related to the GARCH model.
- `lstm/`: Contains scripts and resources for the LSTM model.
- `lstm_garch/`: Scripts and resources for the combined LSTM-GARCH model.
- `lstm_garch_vix/`: Includes the LSTM-GARCH model enhanced with VIX input data.
- `models_comparison/`: Scripts for comparing the performance of different models.
- `results/`: Output results from model predictions and evaluations.
- `sensitivity/`: Analysis of model sensitivity to various input parameters.
- `README.md`: This file, which provides an overview of the project and guidance on how to use the resources.

## Model Details
### Architectures
- **GARCH Model**: Utilizes generalized autoregressive conditional heteroskedasticity to model volatility.
- **LSTM Model**: Employs Long Short-Term Memory units to capture long-term dependencies in time series data.
- **LSTM-GARCH Model**: Combines LSTM and GARCH methodologies to leverage both deep learning and econometric models.
- **LSTM-GARCH with VIX Input**: Integrates market sentiment data from the VIX index into the LSTM-GARCH model for enhanced predictive accuracy.

## Usage
To use the models and scripts in this project, follow these simple steps:

### Step 1: Clone the Repository
First, clone the repository to your local machine using the following command:
'git clone https://github.com/nataliaroszyk/LSTM_volatility_prediction.git'

### Step 2: Navigate to the Project Directory
After cloning, change into the project directory:
'cd LSTM_volatility_prediction'

### Step 3: Install Dependencies
Ensure that you have Python 3.8 or higher installed, then install the required Python packages:
'pip install -r requirements.txt'

### Step 4: Run the Models
Navigate to the directory of the model you want to run, and execute the appropriate script. For example, to run the LSTM model:
cd lstm
python lstm.py



