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

## Computing Environment
The scripts and models in this project were executed on Google Cloud, utilizing the following hardware specifications to ensure efficient processing and performance:
- **Runtime type**: Python 3
- **Hardware Accelerator**: 
  - TPU
- **High RAM**: Enabled for increased memory capacity, facilitating larger datasets and more complex model training without performance degradation.

Using these settings will help in achieving similar results as those documented in this project, maintaining consistency across different runs and ensuring that the models perform as expected.

## Usage
To use the models and scripts in this project, follow these simple steps:

### Step 1: Clone the Repository
First, clone the repository to your local machine using the following command:
`git clone https://github.com/nataliaroszyk/LSTM_volatility_prediction.git`

### Step 2: Navigate to the Project Directory
After cloning, change into the project directory:
`cd LSTM_volatility_prediction`

### Step 3: Install Dependencies
Ensure that you have Python 3.8 or higher installed, then install the required Python packages:
`pip install -r requirements.txt`

### Step 4: Run the Models
Navigate to the directory of the model you want to run, and execute the appropriate script. For example, to run the LSTM model:
`cd lstm`
`python3 lstm.py`

Repeat similar steps for other models by navigating into their respective directories and running their scripts.

### Step 5: Review Results
After running the models, you can check the results in the `results/` directory.

## Sensitivity Analysis
The `sensitivity/` directory contains scripts to analyze the impact of different parameters and settings on LSTM-GARCH with VIX input model performance:

- `layer_1.py`: Analyzes how using a single LSTM layer affects the model performance.
- `layer_3.py`: Analyzes the impact of using three LSTM layers.
- `mae_loss.py`: Measures the effect of Mean Absolute Error as the loss function on model training.
- `pct_change_input.py`: Examines how input features' percentage change impacts predictions.
- `relu.py`: Tests the performance with the ReLU activation function in neural layers.
- `sensitivity_model_function.py`: General function for running sensitivity tests on various model configurations.
- `sequence_length_5.py`: Explores the impact of using a sequence length of 5.
- `sequence_length_66.py`: Studies the effects of a longer sequence length of 66.

To run any sensitivity analysis script, navigate to the `sensitivity/` directory and execute the desired script. For example:
`cd sensitivity`
`python3 layer_1.py`

## Model Comparison
The `models_comparison/` directory contains scripts and methodologies for comparing the predictive accuracy and robustness of different models using key performance indicators such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). This comparison extends across various model configurations and sensitivity analysis results.

### Key Comparisons
- **Basic Model Performance**: Compare the base models (GARCH, LSTM, LSTM-GARCH, LSTM-GARCH with VIX input) in terms of their MAE and RMSE to determine baseline accuracy.
- **Impact of Input Changes**: Evaluate how changes in input types (log returns vs. percentage changes) affect model performance.
- **Effect of Network Depth**: Analyze the impact of using different numbers of LSTM layers (1, 2, 3) on the model's ability to capture volatility dynamics.
- **Sequence Length Sensitivity**: Compare models trained with different lookback periods (5 days, 22 days, 66 days) to see how the amount of historical data influences predictions.
- **Loss Function Variations**: Assess how using different loss functions (MAE vs. MSE) impacts model training and forecast accuracy.
- **Activation Function**: Investigate the effects of different activation functions (ReLU vs. Tanh) on the learning process and prediction outcomes.

### Running Model Comparisons
To run the model comparisons, navigate to the `models_comparison/` directory and execute the comparison scripts.




