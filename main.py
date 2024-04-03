from Wang_Chongrui_Machine_Learning_utils import *
from Wang_Chongrui_Deep_Learning_utils import *
from Wang_Chongrui_utils import *

import argparse
import json
import os
import pandas as pd


def add_arguments():
    parser = argparse.ArgumentParser(description='Run deep learning models for stock return prediction.')
    parser.add_argument('--config', required=True, help='Path to the JSON configuration file')
    parser.add_argument('--model_type', choices=['deep_learning', 'machine_learning', 'None'], required=True, help='Task type: deep_learning or machine_learning')
    parser.add_argument('--function', choices=['get_DL_score', 'get_DL_score_LTR', 'get_score_ML', 'get_score_ML_ensemble', 'None'], required=True, help='Function to run: get_DL_score or get_DL_score_LTR')
    parser.add_argument('--task', choices=['classification', 'regression', 'backtest'], required=True, help='Task type: classification, regression, or backtest')
    parser.add_argument('--data_source', required=True, help='Data source: "10stocks", "allstocks", or a path to a .csv file')
    return parser.parse_args()


def validate_arguments(args):
    if not os.path.exists(args.config):
        raise ValueError("The specified configuration file does not exist.")

    if args.model_type == 'deep_learning' and args.function not in ['get_DL_score', 'get_DL_score_LTR']:
        raise ValueError("Invalid function choice. Choose either 'get_DL_score' or 'get_DL_score_LTR'.")
    
    if args.model_type == 'machine_learning' and args.function not in ['get_score_ML', 'get_score_ML_ensemble']:
        raise ValueError("Invalid function choice. Choose either 'get_score_ML' or 'get_score_ML_ensemble'.")

    if args.task not in ['classification', 'regression', 'backtest']:
        raise ValueError("Invalid task choice. Choose either 'classification', 'regression', or 'backtest'.")

    if args.data_source not in ['10stocks', 'allstocks', 'None'] and not args.data_source.endswith('.csv'):
        raise ValueError("Invalid data source. It must be either '10stocks', 'allstocks', or a valid .csv file path.")
    

def main():
    args = add_arguments()
    validate_arguments(args)

    # Load configurations from JSON file
    with open(args.config) as file:
        config = json.load(file)

    # Set global variables
    if config.get('global_settings'):
        globals().update(config['global_settings'])

    # Call the specified function based on the command-line argument
    method_config = config[args.task]

    if args.task in ['classification', 'regression']:
        # Load or generate the data
        if args.data_source == '10stocks':
            filename = "tickers.csv"
            data_start_date = "1999-07-01"
            data_end_date = "2021-12-31"
            start_date = '2001-01-01'
            end_date = '2021-11-12'
            # Assuming a function to generate data for 10 stocks
            train_df, test_df = prepare_10stocks_data(filename, start_date, end_date, data_start_date, data_end_date)
        elif args.data_source == 'allstocks':
            # Assuming a function to generate data for all stocks
            filename = './stock_dfs/stock_dfs'
            train_df, test_df = prepare_allstocks_data(filename)
        else:
            data_df = pd.read_csv(args.data_source)
            train_df, test_df = split_dataset(data_df, train_ratio=0.6)

        # Run the appropriate model based on the model type
        if args.model_type == 'deep_learning':
            if args.function == 'get_DL_score':
                get_DL_score(train_df, test_df, method_config['save_dir'], method_config['model_name'], method_config['loss_name'], method_config['ignore_cols'], method_config['method'])
            elif args.function == 'get_DL_score_LTR':
                get_DL_score_LTR(train_df, test_df, method_config['save_dir'], method_config['model_name'], method_config['loss_name'], method_config['ignore_cols'], method_config['method'])
        elif args.model_type == 'machine_learning':
            if args.function == 'get_score_ML':
                get_score_ML(train_df, test_df, config['save_dir'], config['model_name'], config['ignore_cols'], method_config['method'], method_config['params'], method_config['hyperparams'])
            elif args.function == 'get_score_ML_ensemble':
                get_score_ML_ensemble(train_df, test_df, method_config['save_dir'], method_config['model_names'], method_config['ignore_cols'], method_config['method'], method_config['ensemble_method'], method_config['params'])
    elif args.task == 'backtest':
        for ticker in method_config['tickers']:
            run_backtest(method_config['model_name'], method_config['method'], ticker, method_config['price_data_dir'])
        return None


if __name__ == '__main__':
    main()