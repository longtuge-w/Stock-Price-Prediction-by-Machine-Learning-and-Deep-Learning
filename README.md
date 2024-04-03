# Stock-Price-Prediction-by-Machine-Learning-and-Deep-Learning

# Note on Python Environment

## Installing the Packages

You can install all the packages using pip. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

It's recommended to use a virtual environment for your Python projects. This helps to manage dependencies and keep your project isolated from other projects or system-wide Python settings. If you are not already using a virtual environment, you can set one up using `virtualenv` or `conda` (if you are using Anaconda).

### Using virtualenv

1. Install virtualenv if you haven't already.

   ```bash
   pip install virtualenv
   ```
2. Create a virtual environment in your project directory.

   ```bash
   virtualenv venv
   ```
3. Activate the virtual environment:

   - On Windows, use the activation script in the `Scripts` folder of the virtual environment.
     ```bash
     .\venv\Scripts\activate
     ```
   - On Unix or MacOS, use the `source` command to activate the environment.
     ```bash
     source venv/bin/activate
     ```
4. Once the virtual environment is activated, you can then run the `pip install` command as shown above.

### Using Conda

1. Create a new conda environment specifying the desired Python version.
   ```bash
   conda create -n myenv python=3.7
   ```
2. Activate the conda environment using the `conda activate` command.
   ```bash
   conda activate myenv
   ```
3. Follow the same `pip install` step as above after activating the conda environment.

Remember to replace the environment name and Python version with your preferences.

# Wang_Chongrui_test_prob1.py Overview

The `Wang_Chongrui_test_prob1.py` script demonstrates the functionality of a custom data loader `CustomCSVData` in Backtrader, a Python library used for backtesting trading strategies.

#### CustomCSVData Class Overview

- **Purpose**: To load and preprocess financial data from CSV files in various formats for backtesting.
- **Class Inheritance**: Inherits from `bt.feeds.GenericCSVData`.
- **Parameters**:
  - `csvformat`: Specifies the format of the CSV file ('type1', 'type2', 'type3').
  - Other common parameters like `dtformat` (date format) are also defined.

#### CustomCSVData Methods

- **`__init__`**: Initializes the data loader and configures it based on the CSV format.
- **`configure_loader_type1`**: Configures the loader for 'type1' CSV format with specific column mappings.
- **`configure_loader_type2`**: Adjusts the loader for 'type2' format with a different set of column mappings.
- **`configure_loader_type3`**: Sets up the loader for 'type3' format, potentially involving more complex preprocessing steps.
- **`start`**: Reads the CSV data, performs any necessary sorting or processing based on the CSV format, and saves the sorted data to a temporary file for further loading.

#### Script Functionality

- **`test` Function**: Demonstrates how to use the `CustomCSVData` class with Backtrader.
  - Initializes a Backtrader `Cerebro` instance.
  - Adds data in different CSV formats to the Cerebro instance for backtesting.
  - Runs the backtesting simulation and prints confirmation messages.

#### Running the Script

To execute the script and test the custom data loader, use the command:

```bash
python Wang_Chongrui_test_prob1.py
```

# Wang_Chongrui_main.py Script Overview

This script is designed for running machine learning and deep learning models for stock return prediction.

## Imports

- **Custom Utility Modules**: Modules for specific ML and DL tasks.
- **Standard Modules**: `argparse`, `json`, `os`, `pandas`.

## Functions

### `add_arguments`

- **Purpose**: Define and parse command-line arguments.
- **Arguments**:
  - `--config`: JSON configuration file path.
  - `--model_type`: Model type (`deep_learning`, `machine_learning`, `None`).
  - `--function`: Function to run.
  - `--task`: Task type (`classification`, `regression`, `backtest`).
  - `--data_source`: Data source.

### `validate_arguments`

- **Purpose**: Validate the parsed arguments.
- **Checks**: Configuration file existence, function compatibility, task type, data source.

### `main`

- **Functionality**: Parses arguments, loads configurations, sets globals, and calls specific functions based on task and model type.
- **Tasks**: Handles `classification`, `regression`, and `backtest`.

## Main Execution

- The script runs the `main()` function when executed as the main program.

# Using `get_score_ML` for Machine Learning in Stock Return Prediction

## Function Overview

`get_score_ML` is designed to apply machine learning models to predict stock returns, handling tasks like training, testing, and evaluation.

### Parameters

- **train_df, test_df**: Training and testing datasets.
- **save_dir**: Directory for saving outputs.
- **model_name**: Name of the ML model.
- **ignore_cols**: Columns to exclude from the analysis.
- **method**: Task type ('classification' or 'regression').
- **params, hyperparams**: Model and hyperparameter settings.

## Execution Command

Run the function using a command like:

```bash
python Wang_Chongrui_main.py --config Wang_Chongrui_LightGBM.json --model_type machine_learning --function get_score_ML --task classification --data_source 10stocks
```

# JSON Configuration File for Machine Learning Model

The JSON configuration file specifies settings for running a machine learning model. Here is a detailed explanation of each section in the file:

## Model Configuration

- **model_name**: Specifies the machine learning model to be used. For example, `"LightGBM"`.

## Ignored Columns

- **ignore_cols**: A list of column names in the dataset that should be ignored during the model training and testing process.

## Save Directory

- **save_dir**: Path to the directory where the model and other output files will be saved.

## Classification Settings

- **classification**: Configuration specific to classification tasks.
  - **method**: Specifies the task type, here "classification".
  - **params**: Model-specific parameters. Set to null if default parameters are to be used.
  - **hyperparams**: Settings for hyperparameter tuning.
    - **uniform**: Specifies continuous hyperparameters with their range.
    - **int**: Specifies integer hyperparameters with their range.
    - **choice**: For categorical hyperparameters (empty in this case).
    - **maximum**: Optimization goal (e.g., true for maximizing or false for minimizing).
    - **scoring**: Metric used for evaluating the model performance.

## Regression Settings

- **regression**: Configuration specific to regression tasks. Similar structure to classification, but with different settings tailored for regression.

```json
{
    "model_name": "LightGBM",
    "ignore_cols": ["date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "ticker", "return", "label"],
    "save_dir": ".",
    "classification": {
        "method": "classification",
        "params": null,
        "hyperparams": {
            "uniform": {
                "learning_rate": [0.01, 1],
                "subsample": [0.1, 1],
                "colsample_bytree": [0.1, 1],
                "reg_lambda": [0.01, 10],
                "reg_alpha": [0.01, 10]
            },
            "int": {
                "n_estimators": [100, 1000],
                "num_leaves": [20, 150],
                "max_depth": [3, 20]
            },
            "choice": {},
            "maximum": true,
            "scoring": "accuracy"
        }
    },
    "regression": {
        "method": "regression",
        "params": null,
        "hyperparams": {
            "uniform": {
                "learning_rate": [0.01, 1],
                "subsample": [0.1, 1],
                "colsample_bytree": [0.1, 1],
                "reg_lambda": [0.01, 10],
                "reg_alpha": [0.01, 10]
            },
            "int": {
                "n_estimators": [100, 1000],
                "num_leaves": [20, 150],
                "max_depth": [3, 20]
            },
            "choice": {},
            "maximum": false,
            "scoring": "neg_mean_squared_error"
        }
    }
}

```

# `get_score_ML_ensemble` Function for Ensemble Machine Learning Models

The `get_score_ML_ensemble` function is designed for training ensemble machine learning models. It can handle both classification and regression tasks using different ensemble methods.

## Function Overview

- **Purpose**: Apply ensemble machine learning techniques to stock return prediction.
- **Parameters**:
  - `train_df`, `test_df`: DataFrames for training and testing.
  - `save_dir`: Directory for saving the model and factors.
  - `model_names`: List of model names used in the ensemble.
  - `ignore_cols`: Columns to ignore in the dataset.
  - `method`: Task type, either 'classification' or 'regression'.
  - `ensemble_method`: Type of ensemble method ('vote' or 'bag').
  - `params`: Model-specific parameters.

## Ensemble Methods

- **Voting**: Combines predictions from multiple models. In 'classification', it uses a soft voting strategy, while in 'regression', it averages the predictions.
- **Bagging**: Trains the same model on different subsets of the dataset. Useful for reducing variance and improving robustness.

### Running the Function

To execute the `get_score_ML_ensemble` function, use the following command:

```bash
python Wang_Chongrui_main.py --config Wang_Chongrui_machine_learning_ensemble.json --model_type machine_learning --function get_score_ML_ensemble --task classification --data_source 10stocks
```

# JSON Configuration for `get_score_ML_ensemble`

The JSON configuration file specifies settings for the ensemble machine learning models. Here's a breakdown of each parameter in the file:

## Common Parameters for Both Classification and Regression

- **save_dir**: The directory where the model and output files will be saved.
- **model_names**: A list of names of the machine learning models to be included in the ensemble.
- **method**: Specifies the type of task - either 'classification' or 'regression'.
- **ensemble_method**: The ensemble technique to be used. Options include 'vote' for voting ensemble and 'bag' for bagging ensemble.
- **ignore_cols**: Columns from the dataset to be ignored during model training and testing.
- **params**: Model-specific parameters. Set to `null` if default parameters are to be used.

### Example Structure

```json
{
    "classification": {
        "save_dir": ".",
        "model_names": ["LightGBM", "LightGBM", "LightGBM", "LightGBM", "LightGBM"],
        "method": "classification",
        "ensemble_method": "vote",
        "ignore_cols": ["date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "ticker", "return", "label"],
        "params": null
    },
    "regression": {
        "save_dir": ".",
        "model_names": ["LightGBM", "LightGBM", "LightGBM", "LightGBM", "LightGBM"],
        "method": "classification",
        "ensemble_method": "vote",
        "ignore_cols": ["date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "ticker", "return", "label"],
        "params": null
    }
}

```

# JSON Configuration for `get_DL_score` Deep Learning Function

The JSON configuration file specifies settings for training deep learning models. Below is an explanation of each parameter in the file:

## Global Settings

These settings apply to all models irrespective of the task (classification or regression):
-**BATCH_SIZE**: The size of each batch of data during training.

- **SEQ_LEN**: The length of the input sequences.
- **LEARNING_RATE**: The learning rate for the optimizer.
- **MOMENTUM**: The momentum factor in the optimizer.
- **WEIGHT_DECAY**: The weight decay (L2 penalty) used for regularization.
- **MAX_EPOCH**: The maximum number of epochs for training.
- **TEST_SIZE**: The proportion of the dataset to include in the test split.
- **RANDOM_SEED**: The seed used for random number generation.
- **PATIENCE**: Number of epochs with no improvement after which training will be stopped.

### Model-Specific Configuration for Classification

- **save_dir**: Directory to save the trained model and results.
- **model_name**: The name of the deep learning model, e.g., "LSTM".
- **loss_name**: The loss function used, e.g., "BCE" for Binary Cross-Entropy.
- **method**: The type of task, here "classification".
- **ignore_cols**: Columns in the dataset to be ignored during model training.

### Model-Specific Configuration for Regression

- **model_name**: Same as above, but for regression models.
- **loss_name**: The loss function for regression, e.g., "IC".
- **method**: Set to "regression" for regression tasks.
- **ignore_cols**: Columns to ignore, similar to classification.

## Key Steps

1. Validate method and create necessary directories.
2. Prepare data loaders for training, validation, and testing.
3. Initialize and potentially load a pre-trained model.
4. Train the model or load the best model state.
5. Evaluate the model on the test data.
6. Calculate accuracy and precision metrics.
7. Save predictions and metrics.

### Running the Function

To execute the `get_DL_score` function for training deep learning models, use the following command:

```bash
python Wang_Chongrui_main.py --config Wang_Chongrui_Deep_Learning.json --model_type deep_learning --function get_DL_score --task classification --data_source 10stocks
```

## JSON Configuration

The JSON file provides global settings and specific model configurations.

### Global Settings

Includes batch size, sequence length, learning rate, momentum, weight decay, maximum epochs, test size, random seed, and patience for early stopping.

### Model-Specific Configuration

```json
{
    "global_settings": {
        "BATCH_SIZE": 256,
        "SEQ_LEN": 30,
        "LEARNING_RATE": 1e-4,
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 5e-5,
        "MAX_EPOCH": 100,
        "TEST_SIZE": 0.2,
        "RANDOM_SEED": 42,
        "PATIENCE": 5
    },
    "classification": {
        "save_dir": ".",
        "model_name": "LSTM",
        "loss_name": "BCE",
        "method": "classification",
        "ignore_cols": ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "return", "label"]
    },
    "regression": {
        "save_dir": ".",
        "model_name": "LSTM",
        "loss_name": "IC",
        "method": "regression",
        "ignore_cols": ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "return", "label"]
    }
}

```

### Explanation of `get_DL_score_LTR` Function for Deep Learning

The `get_DL_score_LTR` function is specifically designed for training deep learning models using a Learning to Rank (LTR) approach, suitable for regression tasks.

#### Function Overview

- **Purpose**: Implement deep learning models for regression tasks using LTR.
- **Parameters**:
  - `train_df`, `test_df`: DataFrames for training and testing.
  - `save_dir`: Directory to save the model and results.
  - `model_name`: Name of the deep learning model.
  - `loss_name`: Specifies the loss function, tailored for LTR.
  - `ignore_cols`: Columns to be ignored in the dataset.
  - `method`: Specifies the task type, set to 'regression' for LTR.

#### Key Steps

1. Validation of the method as 'regression'.
2. Creation of necessary directories for storing models and results.
3. Preparation of data loaders specifically for LTR tasks.
4. Initialization and training of the deep learning model.
5. Evaluation of the model on test data.
6. Calculation of accuracy and precision metrics.
7. Saving the predictions in a CSV file.

### Running the `get_DL_score_LTR` Function

Execute the `get_DL_score_LTR` function with the following command:

```bash
python Wang_Chongrui_main.py --config Wang_Chongrui_Deep_Learning_LTR.json --model_type deep_learning --function get_DL_score_LTR --task regression --data_source 10stocks
```

### JSON Configuration Parameters for `get_DL_score_LTR`

The JSON configuration file for the `get_DL_score_LTR` function sets up parameters for training deep learning models, specifically for regression tasks using the Learning to Rank (LTR) approach.

#### Global Settings

- **BATCH_SIZE**: Number of samples per gradient update.
- **SEQ_LEN**: Length of the sequence data.
- **LEARNING_RATE**: Step size at each iteration of the learning process.
- **MOMENTUM**: Parameter to accelerate SGD in the relevant direction.
- **WEIGHT_DECAY**: L2 penalty, a regularization technique.
- **MAX_EPOCH**: Maximum number of training epochs.
- **TEST_SIZE**: Proportion of the dataset to be used as the test set.
- **RANDOM_SEED**: Seed for random number generation to ensure reproducibility.
- **PATIENCE**: Number of epochs to wait for improvement before stopping.

#### Model-Specific Settings for Regression

- **save_dir**: Path to save the model and output files.
- **model_name**: Name of the deep learning model to be used.
- **loss_name**: Name of the loss function specific to the model.
- **method**: Type of machine learning task, set to 'regression' for LTR.
- **ignore_cols**: List of columns to be ignored during model training and testing.

Example:

```json
{
    "global_settings": {
        "BATCH_SIZE": 256,
        "SEQ_LEN": 30,
        "LEARNING_RATE": 1e-4,
        "MOMENTUM": 0.9,
        "WEIGHT_DECAY": 5e-5,
        "MAX_EPOCH": 100,
        "TEST_SIZE": 0.2,
        "RANDOM_SEED": 42,
        "PATIENCE": 5
    },
    "regression": {
        "save_dir": ".",
        "model_name": "LSTM",
        "loss_name": "ListMLE",
        "method": "regression",
        "ignore_cols": ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "return", "label"]
    }
}

```

### Explanation of `run_backtest` Function for Backtesting

The `run_backtest` function is designed to perform backtesting on financial models, particularly useful for evaluating the performance of trading strategies based on historical data.

#### Function Overview

- **Purpose**: To conduct backtesting using historical price data and machine learning models.
- **Parameters**:
  - `model_name`: Name of the machine learning model used for predictions.
  - `method`: Specifies the type of machine learning task (e.g., 'classification').
  - `ticker`: The stock ticker for which the backtest is to be run.
  - `price_data_dir`: Directory containing the price data files.

#### Key Steps in the Function

1. **Data Loading**: Loads historical price data for the specified ticker from the given directory.
2. **Data Preprocessing**: Renames columns, sets the index to date, and organizes data for backtesting.
3. **Backtrader Setup**: Initializes the backtesting environment with the specified strategy and data.
4. **Running the Backtest**: Executes the backtesting simulation and prints the portfolio values before and after the backtest.
5. **Result Processing**: Analyzes the strategy's performance using metrics like returns and positions.
6. **Report Generation**: Creates a detailed HTML report on the strategy's performance compared to a benchmark.

### Running the `run_backtest` Function

To execute the `run_backtest` function for backtesting, use the following command:

```bash
python Wang_Chongrui_main.py --config Wang_Chongrui_backtest.json --model_type None --function None --task backtest --data_source None
```

### JSON Configuration Parameters for `run_backtest`

The JSON configuration file for the `run_backtest` function sets up parameters for backtesting:

- **model_name**: The machine learning model used for generating trading signals.
- **method**: The type of machine learning task, typically 'classification' or 'regression'.
- **tickers**: A list of stock tickers to run the backtest on.
- **price_data_dir**: Path to the directory where the price data and prediction files are stored.

### Json Example:

```json
{
    "backtest": {
        "model_name": "LightGBM",
        "method": "classification",
        "tickers": ["FE"],
        "price_data_dir": "./Factor/"
    }
}

```
