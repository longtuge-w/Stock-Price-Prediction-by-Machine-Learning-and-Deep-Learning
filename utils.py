import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import backtrader as bt
from scipy.stats import zscore, rankdata, spearmanr, pearsonr
from datetime import timedelta
import quantstats as qs


OHLCV = ['open', 'high', 'low', 'close', 'volume']


def read_tickers(filename="tickers.csv"):
    tickers = pd.read_csv(filename)
    tickers = tickers.values.flatten()
    return tickers


def get_stock_data(start_date: str, end_date: str, tickers: np.array):

    # Fetch the stock data and add ticker column
    def fetch_stock_data(ticker):
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        df['Ticker'] = ticker  # Add ticker as a column
        return df

    # Concatenate data from all tickers into a single DataFrame
    stock = pd.concat([fetch_stock_data(ticker) for ticker in tickers])

    # Reset index to ensure the date is a column and not an index
    stock.reset_index(inplace=True)

    # Ensure the date column is of type datetime64
    stock['Date'] = pd.to_datetime(stock['Date'])
    stock['Date'] = stock['Date'].dt.strftime('%Y-%m-%d')


    stock.rename(columns={'Date': 'date', 'Ticker': 'ticker'}, inplace=True)
    return stock


def preprocess_data(df, ignore_cols):
    print('Filling nan value ...')
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    print('Clipping outliers ...')
    # Perform calculations for each date in one go
    def clip_mad(group):
        cols = group.columns.difference(ignore_cols)
        median = group[cols].median()
        mad = np.abs(group[cols] - median).median()
        threshold = 3 * mad
        lower, upper = median - threshold, median + threshold
        group[cols] = group[cols].clip(lower, upper, axis=1)
        return group

    df = df.groupby('date').apply(clip_mad)

    # Reset index if 'date' is part of the index
    if 'date' in df.index.names:
        df = df.reset_index(drop=True)

    print('Normalizing features ...')
    # Normalize in one go
    cols_to_normalize = df.columns.difference(ignore_cols)
    df[cols_to_normalize] = df.groupby('date')[cols_to_normalize].transform(lambda x: zscore(x, nan_policy='omit'))

    return df


def combine_stock_data(folder_path):
    """
    Reads all CSV files in the specified folder, each representing a stock,
    adds a 'ticker' column with the file name, and concatenates them into a single DataFrame.

    Parameters:
    folder_path (str): The path to the folder containing the CSV files.

    Returns:
    pandas.DataFrame: A combined DataFrame with all stock data and a 'ticker' column.
    """
    all_dataframes = []

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Extract ticker name from filename and add it as a new column
            ticker = filename.split('.')[0]
            df['ticker'] = ticker

            all_dataframes.append(df)

    # Concatenate all dataframes into one
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    return combined_df


def split_dataset(df, train_ratio=0.6):
    """
    Splits the dataset into training and testing sets based on dates.

    :param df: Pandas DataFrame containing 'date' and 'ticker' columns
    :param train_ratio: The ratio of the dataset to be used as training data
    :return: Two DataFrames, one for training and one for testing
    """
    # Ensure the data is sorted by date
    df_sorted = df.sort_values(by='date')

    # Get unique dates and calculate the split index
    unique_dates = df_sorted['date'].unique()
    train_index = int(len(unique_dates) * train_ratio)

    # Split the dates for training and testing
    train_dates = unique_dates[:train_index]
    test_dates = unique_dates[train_index:]

    # Split the DataFrame into training and testing sets
    train_df = df_sorted[df_sorted['date'].isin(train_dates)].reset_index(drop=True)
    test_df = df_sorted[df_sorted['date'].isin(test_dates)].reset_index(drop=True)

    return train_df, test_df


def one_hot_encode_tickers(df):
    # Ensure 'ticker' column is of type 'category' for efficient encoding
    df['ticker'] = df['ticker'].astype('category')

    # Create one-hot encoded variables for the 'ticker' column
    one_hot_encoded = pd.get_dummies(df['ticker'], prefix='ticker')

    # Drop the original 'ticker' column and concatenate the one-hot encoded columns
    df = pd.concat([df.drop('ticker', axis=1), one_hot_encoded], axis=1)

    return df


def prepare_10stocks_data(filename, start_date, end_date, data_start_date, data_end_date):
    ignore_cols = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'ticker']
    # Read tickers from the provided file
    # 'filename' is expected to be a path to a file containing stock tickers
    print('Loading tickers for the 10 stocks ...')
    ticker = read_tickers(filename)

    # Retrieve stock data for the given date range and tickers
    # 'start_date' and 'end_date' define the time period for the data
    # 'ticker' is a list of stock tickers obtained from the previous step
    print('Loading stock data ...')
    stock = get_stock_data(data_start_date, data_end_date, ticker)

    # Initialize an AlphaFactors object with the retrieved stock data
    # This object is presumably responsible for calculating various financial indicators (factors)
    alpha_calculator = AlphaFactors(stock)

    # Calculate alpha factors (financial indicators) for the data
    print('Calculating the alpha factors ...')
    alpha_calculator.calculate_factors()

    # Make a copy of the DataFrame with the calculated factors
    data = alpha_calculator.df.copy()

    # Preprocess the data by applying various transformations
    # 'ignore_cols' is a list of column names to be excluded from certain processing steps
    print('Preprocessing the alpha factors ...')
    data_processed = preprocess_data(data, ignore_cols)
    data_processed['return'] = data_processed.groupby('ticker')['Close'].shift(-1) / data_processed['Close'] - 1
    data_processed.dropna(subset=['return'], inplace=True)
    data_processed['label'] = (data_processed['return'] > 0).astype(int)

    data_processed = data_processed.loc[(data_processed['date'] >= start_date) & (data_processed['date'] <= end_date)]

    train_df, test_df = split_dataset(data_processed, train_ratio=0.6)

    return train_df, test_df


def prepare_allstocks_data(filename):
    ignore_cols = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'ticker']
    # Combine data from multiple stock CSV files into a single DataFrame
    # 'filename' is expected to contain the path to the folder with CSV files
    print('Loading tickers for all the stocks ...')
    stock = combine_stock_data(filename)

    # Capitalize the first letter of specified column names
    stock.rename(columns={col: col.capitalize() for col in ['open', 'high', 'low', 'close', 'volume']}, inplace=True)

    # Initialize an AlphaFactors object with the combined stock data
    # This object presumably calculates various financial indicators (alpha factors)
    alpha_calculator = AlphaFactors(stock)

    # Calculate alpha factors for the combined stock data
    print('Calculating the alpha factors ...')
    alpha_calculator.calculate_factors()

    # Copy the DataFrame with the calculated alpha factors
    data = alpha_calculator.df.copy()

    # Preprocess the data (e.g., handling missing values, outliers)
    # 'ignore_cols' is a list of column names to exclude from certain processing steps
    print('Preprocessing the alpha factors ...')
    data = preprocess_data(data, ignore_cols)
    data['return'] = data.groupby('ticker')['Close'].shift(-1) / data['Close'] - 1
    data.dropna(subset=['return'], inplace=True)
    data['label'] = (data['return'] > 0).astype(int)

    # Split the data into training and test sets
    # 'train_ratio' defines the proportion of the data to be used for training
    train_df, test_df = split_dataset(data, train_ratio=0.6)

    # Select the first 60 unique dates in the training set
    first_60_dates = train_df['date'].unique()[:60]

    # Remove rows from the training set that correspond to the first 60 dates
    train_df = train_df[~train_df['date'].isin(first_60_dates)].reset_index(drop=True)

    # Return the final training and test DataFrames
    return train_df, test_df


class CustomCSVData(bt.feeds.GenericCSVData):
    params = (
        ('csvformat', 'type1'),  # default CSV format
        # Define common params like date format
        ('dtformat', '%Y-%m-%d'),
        # Add additional parameters for type2 and type3 if needed
    )

    def __init__(self):
        super(CustomCSVData, self).__init__()

        if self.params.csvformat == 'type1':
            self.configure_loader_type1()
        elif self.params.csvformat == 'type2':
            self.configure_loader_type2()
        elif self.params.csvformat == 'type3':
            self.configure_loader_type3()
        else:
            raise ValueError("Unsupported CSV format")

    def configure_loader_type1(self):
        # For CSV type 1, column mapping is already set in params
        self.p.datetime = 0
        self.p.open = 1
        self.p.high = 2
        self.p.low = 3
        self.p.close = 4
        self.p.volume = 5
        self.p.adjclose = 6

    def configure_loader_type2(self):
        # Configure data loading for CSV type 2
        # Assuming 'Unnamed: 0' is the datetime column, map it correctly
        self.p.datetime = 0
        self.p.open = 1
        self.p.high = 3
        self.p.low = 4
        self.p.close = 2
        self.p.volume = 5
        # Additional configurations for other columns as needed

    def configure_loader_type3(self):
        # Configure data loading for CSV type 3
        # Set other params to -1 as they are not used
        self.p.datetime = 0
        self.p.close = 1
        self.p.dtformat = '%Y-%m-%d %H:%M:%S'
        self.p.open = self.p.high = self.p.low = self.p.volume = self.p.adjclose = self.p.openinterest = -1

    def start(self):
        # Load the data from CSV
        dataframe = pd.read_csv(self.p.dataname)

        # Apply sorting based on the CSV format
        if self.p.csvformat == 'type1':
            # For type 1, where 'Date' is the date column
            dataframe['Date'] = pd.to_datetime(dataframe['Date'], format=self.p.dtformat)
            dataframe.sort_values('Date', inplace=True)
        elif self.p.csvformat == 'type2':
            # For type 2, where 'Unnamed: 0' is the date column
            dataframe.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
            dataframe['Date'] = pd.to_datetime(dataframe['Date'], format=self.p.dtformat)
            dataframe.sort_values('Date', inplace=True)
        elif self.p.csvformat == 'type3':
            # Assuming the dataframe is loaded as per your existing code
            dataframe = pd.read_csv(self.p.dataname)
            # Convert 'Date' to datetime
            dataframe['Date'] = pd.to_datetime(dataframe['Date'])
            # Function to adjust date and time
            def adjust_datetime(row):
                if row['Hour_of_Day'] == 24:
                    # Set time to '00:00:00' and move to the next day
                    row['Date'] += timedelta(days=1)
                    return row['Date'].strftime('%Y-%m-%d') + ' 00:00:00'
                else:
                    # Keep the date the same and adjust the hour
                    return row['Date'].strftime('%Y-%m-%d') + f' {row["Hour_of_Day"]:02d}:00:00'
            # Apply the function to each row
            dataframe['Date'] = dataframe.apply(adjust_datetime, axis=1)
            # Convert 'Date' to datetime
            dataframe['Date'] = pd.to_datetime(dataframe['Date'], format=self.p.dtformat)
            # Now, 'Date' is your combined datetime column
            dataframe.sort_values(['Date'], inplace=True)
            # Optionally drop the original 'Date' and 'Hour_of_Day' columns
            dataframe.drop(['Hour_of_Day'], axis=1, inplace=True)

        # Check if dataframe is empty before saving
        if dataframe.empty:
            raise ValueError(f"Dataframe is empty after processing {self.p.dataname}")

        # Save the sorted data to a temporary CSV file
        sorted_csv = 'sorted_temp.csv'

        # Update the dataname to the sorted temp file
        self.p.dataname = sorted_csv

        # Call super() to continue the normal data loading process
        super(CustomCSVData, self).start()


class AlphaFactors:
    def __init__(self, df):
        self.df = df.copy()
        self.df = self.df.sort_values(['ticker', 'date'])
        self.df = self.df.reset_index(drop=True)

    def calculate_factors(self):
        self._kbar_factors()
        self._rolling_factors()

    def _kbar_factors(self):
        df = self.df.copy()
        df['KMID'] = (df['Close'] - df['Open']) / df['Open']
        df['KLEN'] = (df['High'] - df['Low']) / df['Open']
        df['KMID2'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-12)
        df['KUP'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Open']
        df['KUP2'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / (df['High'] - df['Low'] + 1e-12)
        df['KLOW'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Open']
        df['KLOW2'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / (df['High'] - df['Low'] + 1e-12)
        df['KSFT'] = (2 * df['Close'] - df['High'] - df['Low']) / df['Open']
        df['KSFT2'] = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low'] + 1e-12)
        self.df = df.copy()

    def _rolling_factors(self):
        """
        Calculate rolling factors for each ticker.
        """
        windows = [5, 10, 20, 30, 60]
        grouped = self.df.groupby('ticker')
        
        for w in windows:
            # ROC
            self.df[f'ROC{w}'] = grouped['Close'].apply(lambda x: x.pct_change(periods=w)).reset_index(level=0, drop=True)
            
            # MA
            self.df[f'MA{w}'] = grouped['Close'].apply(lambda x: x.rolling(window=w).mean() / x).reset_index(level=0, drop=True)
            
            # STD
            self.df[f'STD{w}'] = grouped['Close'].apply(lambda x: x.rolling(window=w).std() / x).reset_index(level=0, drop=True)

            # BETA - Here we assume linear regression slope as a proxy for BETA
            self.df[f'BETA{w}'] = grouped['Close'].apply(lambda x: x.rolling(window=w).apply(self._linear_regression_slope, raw=False) / x).reset_index(level=0, drop=True)

            # RSQR
            self.df[f'RSQR{w}'] = grouped['Close'].apply(lambda x: x.rolling(window=w).apply(self._rsquared, raw=False)).reset_index(level=0, drop=True)
        
            # Residual for Linear Regression
            self.df[f'RESI{w}'] = grouped['Close'].apply(
                lambda x: x.rolling(window=w).apply(self._linear_regression_residual, raw=False) / x
            ).reset_index(level=0, drop=True)

            # Max High Price
            self.df[f'MAX{w}'] = grouped['High'].apply(lambda x: x.rolling(window=w).max() / x).reset_index(level=0, drop=True)

            # Min Low Price
            self.df[f'MIN{w}'] = grouped['Low'].apply(lambda x: x.rolling(window=w).min() / x).reset_index(level=0, drop=True)

            # Quantile High
            self.df[f'QTLU{w}'] = grouped['Close'].apply(
                lambda x: x.rolling(window=w).quantile(0.8) / x
            ).reset_index(level=0, drop=True)

            # Quantile Low
            self.df[f'QTLD{w}'] = grouped['Close'].apply(
                lambda x: x.rolling(window=w).quantile(0.2) / x
            ).reset_index(level=0, drop=True)

            # Rank
            self.df[f'RANK{w}'] = grouped['Close'].apply(
                lambda x: x.rolling(window=w).apply(lambda y: rankdata(y)[-1]/len(y), raw=False)
            ).reset_index(level=0, drop=True)

            # RSV
            self.df[f'RSV{w}'] = grouped.apply(
                lambda x: (x['Close'] - x['Low'].rolling(window=w).min()) /
                          (x['High'].rolling(window=w).max() - x['Low'].rolling(window=w).min() + 1e-12)
            ).reset_index(level=0, drop=True)

            # IMAX
            self.df[f'IMAX{w}'] = grouped['High'].rolling(window=w).apply(lambda x: w - np.argmax(x) - 1, raw=False).reset_index(level=0, drop=True)

            # IMIN
            self.df[f'IMIN{w}'] = grouped['Low'].rolling(window=w).apply(lambda x: w - np.argmin(x) - 1, raw=False).reset_index(level=0, drop=True)

            # IMXD
            self.df[f'IMXD{w}'] = grouped.apply(
                lambda x: x['High'].rolling(window=w).apply(np.argmax) -
                          x['Low'].rolling(window=w).apply(np.argmin)
            ).reset_index(level=0, drop=True)

            # CORR
            self.df[f'CORR{w}'] = grouped.apply(
                lambda x: x['Close'].rolling(window=w).corr(np.log(x['Volume'] + 1))
            ).reset_index(level=0, drop=True)

            # CORD
            self.df[f'CORD{w}'] = grouped.apply(
                lambda x: x['Close'].pct_change().rolling(window=w).corr(np.log(x['Volume'].pct_change() + 1))
            ).reset_index(level=0, drop=True)

            # CNTP
            self.df[f'CNTP{w}'] = grouped['Close'].apply(
                lambda x: (x > x.shift(1)).rolling(window=w).mean()
            ).reset_index(level=0, drop=True)

            # CNTN
            self.df[f'CNTN{w}'] = grouped['Close'].apply(
                lambda x: (x < x.shift(1)).rolling(window=w).mean()
            ).reset_index(level=0, drop=True)

            # CNTD
            self.df[f'CNTD{w}'] = self.df[f'CNTP{w}'] - self.df[f'CNTN{w}']

            # SUMP
            self.df[f'SUMP{w}'] = grouped['Close'].apply(
                lambda x: x.rolling(window=w).apply(
                    lambda y: np.sum(np.maximum(y - y.shift(1), 0)) / (np.sum(np.abs(y - y.shift(1))) + 1e-12)
                )
            ).reset_index(level=0, drop=True)

            # SUMN
            self.df[f'SUMN{w}'] = 1 - self.df[f'SUMP{w}']

            # SUMD
            self.df[f'SUMD{w}'] = grouped.apply(
                lambda x: (x['Close'].diff().clip(lower=0).rolling(window=w).sum() -
                           x['Close'].diff().clip(upper=0).abs().rolling(window=w).sum()) /
                          x['Close'].diff().abs().rolling(window=w).sum()
            ).reset_index(level=0, drop=True)

            # VMA
            vma_rolling_mean = self.df.groupby('ticker')['Volume'].rolling(window=w).mean().reset_index(level=0, drop=True)
            volume_series = self.df['Volume']
            self.df[f'VMA{w}'] = vma_rolling_mean / (volume_series + 1e-12)

            # VSTD
            vstd_rolling_std = self.df.groupby('ticker')['Volume'].rolling(window=w).std().reset_index(level=0, drop=True)
            volume_series = self.df['Volume']
            self.df[f'VSTD{w}'] = vstd_rolling_std / (volume_series + 1e-12)

            # WVMA
            self.df[f'WVMA{w}'] = grouped.apply(
                lambda x: x['Close'].pct_change().abs().mul(x['Volume']).rolling(window=w).std() /
                          x['Close'].pct_change().abs().mul(x['Volume']).rolling(window=w).mean()
            ).reset_index(level=0, drop=True)

            # VSUMP
            self.df[f'VSUMP{w}'] = grouped.apply(
                lambda x: x['Volume'].diff().clip(lower=0).rolling(window=w).sum() /
                          x['Volume'].diff().abs().rolling(window=w).sum()
            ).reset_index(level=0, drop=True)

            # VSUMN
            self.df[f'VSUMN{w}'] = 1 - self.df[f'VSUMP{w}']

            # VSUMD
            self.df[f'VSUMD{w}'] = grouped.apply(
                lambda x: (x['Volume'].diff().clip(lower=0).rolling(window=w).sum() -
                           x['Volume'].diff().clip(upper=0).abs().rolling(window=w).sum()) /
                          x['Volume'].diff().abs().rolling(window=w).sum()
            ).reset_index(level=0, drop=True)

    @staticmethod
    def _linear_regression_residual(series):
        y = series.values
        x = np.arange(len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        y_pred = slope * x + intercept
        return np.sum((y - y_pred) ** 2)  # Sum of squares of residuals

    # Define more methods for other factor calculations...
    @staticmethod
    def _linear_regression_slope(series):
        # Linear regression slope (proxy for BETA)
        y = series.values
        x = np.arange(len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return slope

    @staticmethod
    def _rsquared(series):
        # R-squared of linear regression
        y = series.values
        x = np.arange(len(y))
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    

class AlphaFactorMetrics:
    def __init__(self, data, return_col='return', tickers_col='ticker', date_col='date'):
        """
        data: pandas DataFrame with alpha factors, returns, tickers, and dates.
        return_col: column name of future returns.
        tickers_col: column name of ticker symbols.
        date_col: column name of dates.
        """
        self.data = data
        self.return_col = return_col
        self.tickers_col = tickers_col
        self.date_col = date_col

    def calculate_ic(self, factor_name):
        """
        Calculate the Information Coefficient (IC) for a given alpha factor.
        """
        grouped = self.data.groupby(self.date_col)
        ic_values = grouped.apply(lambda x: pearsonr(x[factor_name], x[self.return_col])[0])
        return ic_values.mean()

    def calculate_rank_ic(self, factor_name):
        """
        Calculate the Rank Information Coefficient (Rank IC) for a given alpha factor.
        """
        grouped = self.data.groupby(self.date_col)
        rank_ic_values = grouped.apply(lambda x: spearmanr(x[factor_name], x[self.return_col])[0])
        return rank_ic_values.mean()

    def top_n_factors(self, factor_list, n, rank=False):
        """
        Calculate IC or Rank IC for a list of factors and return the top n factors
        along with their metrics, based on the absolute values of their ICs.
        """
        ic_scores = {}
        for factor in factor_list:
            if rank:
                ic_value = self.calculate_rank_ic(factor)
            else:
                ic_value = self.calculate_ic(factor)

            # Store the factor and its IC value if it's not NaN
            if not np.isnan(ic_value):
                ic_scores[factor] = ic_value

        # Sort factors by absolute IC values and select top n
        top_factors = sorted(ic_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:n]

        # Prepare the output as a list of (factor, ic_value) tuples
        return [(factor, ic_scores[factor]) for factor, _ in top_factors]
    
    def top_n_union_factors(self, factor_list, n):
        """
        Calculate the top n factors based on absolute IC and Rank IC values 
        and return the union of these top factors.
        """
        # Get top n factors based on IC
        top_ic_factors = self.top_n_factors(factor_list, n, rank=False)
        top_ic_factors_set = set([factor for factor, _ in top_ic_factors])

        # Get top n factors based on Rank IC
        top_rank_ic_factors = self.top_n_factors(factor_list, n, rank=True)
        top_rank_ic_factors_set = set([factor for factor, _ in top_rank_ic_factors])

        # Union of top IC and Rank IC factors
        union_factors = top_ic_factors_set.union(top_rank_ic_factors_set)

        # Get the IC values for the union factors
        union_factors_with_ic = [(factor, self.calculate_ic(factor)) for factor in union_factors]

        return union_factors_with_ic

# class to define the columns we will provide
class SignalData(bt.feeds.PandasData):
    """
    Define pandas DataFrame structure
    """
    cols = OHLCV + ['predicted']

    # create lines
    lines = tuple(cols)

    # define parameters
    params = {c: -1 for c in cols}
    params.update({'datetime': None})
    params = tuple(params.items())


# define backtesting strategy class
class MLStrategy(bt.Strategy):
    params = dict(
    )
    
    def __init__(self):
        # keep track of open, close prices and predicted value in the series
        self.data_predicted = self.datas[0].predicted
        self.data_open = self.datas[0].open
        self.data_close = self.datas[0].close
        
        # keep track of pending orders/buy price/buy commission
        self.order = None
        self.price = None
        self.comm = None

    # logging function
    def log(self, txt):
        '''Logging function'''
        dt = self.datas[0].datetime.date(0).isoformat()
        print(f'{dt}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # order already submitted/accepted - no action required
            return

        # report executed order
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}'
                )
                self.price = order.executed.price
                self.comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}'
                )


        # report failed order
        elif order.status in [order.Canceled, order.Margin, 
                              order.Rejected]:
            self.log('Order Failed')

        # set no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')

    # We have set cheat_on_open = True.This means that we calculated the signals on day t's close price, 
    # but calculated the number of shares we wanted to buy based on day t+1's open price.
    def next_open(self):
        if not self.position:
            if self.data_predicted == 0:
                # calculate the max number of shares ('all-in')
                size = int(self.broker.getcash() / self.datas[0].open)
                # buy order
                self.log(f'BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, Open: {self.data_open[0]}, Close: {self.data_close[0]}')
                self.buy(size=size)
        else:
            if self.data_predicted == 1:
                # sell order
                self.log(f'SELL CREATED --- Size: {self.position.size}')
                self.sell(size=self.position.size)
    

def get_top_stocks_by_accuracy(model_name, method):
    """
    Reads a CSV file containing stock data, calculates the prediction accuracy 
    for each ticker, and returns the top N tickers sorted by accuracy.

    Parameters:
    file_path (str): Path to the CSV file.
    top_n (int): Number of top tickers to return based on accuracy.

    Returns:
    pandas.DataFrame: A DataFrame containing the top N tickers sorted by accuracy.
    """

    df = pd.read_csv(f"./Factor/{model_name}_{method}.csv")

    # Add a column for correct predictions
    df['correct'] = ((df['return'] > 0) & (df['pred'] == 1)) | ((df['return'] <= 0) & (df['pred'] == 0))

    # Calculate accuracy for each ticker
    accuracy_df = df.groupby('ticker').agg(
        accuracy=('correct', 'mean')
    )

    # Sort by accuracy and get the top N
    top_stocks = accuracy_df.sort_values(by='accuracy', ascending=False)

    return top_stocks


def combine_stock_data(folder_path):
    """
    Reads all CSV files in the specified folder, each representing a stock,
    adds a 'ticker' column with the file name, and concatenates them into a single DataFrame.

    Parameters:
    folder_path (str): The path to the folder containing the CSV files.

    Returns:
    pandas.DataFrame: A combined DataFrame with all stock data and a 'ticker' column.
    """
    all_dataframes = []

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Extract ticker name from filename and add it as a new column
            ticker = filename.split('.')[0]
            df['ticker'] = ticker

            all_dataframes.append(df)

    # Concatenate all dataframes into one
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    return combined_df


def run_backtest(model_name, method, ticker, price_data_dir):
    # Load price data
    price = pd.read_csv(f"{price_data_dir}/{model_name}_{method}.csv")
    price = price.loc[price['ticker'] == ticker]
    price.reset_index(drop=True, inplace=True)

    if not os.path.exists(f'./Stock'):
        os.makedirs(f'./Stock')

    price.to_csv(f"./Stock/{model_name}_{method}_{ticker}.csv", index=False)

    # Data preprocessing
    price = price.rename(columns={'pred': 'predicted', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    price = price.drop(columns=['ticker'])
    price = price[['date', 'predicted', 'open', 'high', 'low', 'close', 'volume']]
    price['date'] = pd.to_datetime(price['date'])
    price = price.set_index('date')

    # Set up backtrader
    data = SignalData(dataname=price)
    cerebro = bt.Cerebro(stdstats = False, cheat_on_open=True)
    cerebro.addstrategy(MLStrategy)
    cerebro.adddata(data, name=ticker)
    cerebro.broker.setcash(1000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    # Run backtest
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    backtest_result = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Process backtest results
    strat = backtest_result[0]
    portfolio_values = strat.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = portfolio_values.get_pf_items()
    strategy_returns = returns.fillna(0)
    strategy_returns.index = strategy_returns.index.tz_localize(None)
    benchmark_returns = price['close'].pct_change().fillna(0)

    # Extend quantstats and generate report
    qs.extend_pandas()
    filename = f"{model_name}_{method}_{ticker}.html"
    report_title = f"Strategy Performance Report by doing {method} with model {model_name} traded on stock {ticker}"
    qs.reports.html(strategy_returns, benchmark_returns, download_filename=filename, title=report_title, output=True)
    

def plot_nan_counts(df, f_cols, start_date, end_date):
    """
    Function to plot the number of NaN values per column in a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    f_cols (list): List of columns to consider for NaN count.
    start_date (str): Start date for filtering the DataFrame.
    end_date (str): End date for filtering the DataFrame.
    """
    # Filter the DataFrame based on the given date range
    filtered_df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]
    filtered_df = filtered_df[f_cols]

    # Calculate the number of NaN values for each column
    nan_counts = filtered_df.isna().sum()

    # Filter out columns with zero NaN values to reduce clutter
    nan_counts = nan_counts[nan_counts > 0]

    # Plot
    plt.figure(figsize=(15, 8))  # Adjust the figure size as needed
    nan_counts.plot(kind='bar')
    plt.title('Number of NaN Values per Column')
    plt.xlabel('Columns')
    plt.ylabel('NaN Count')
    plt.xticks(rotation=45)  # Rotate the column names for better readability
    plt.savefig('nan_count.png')
    plt.show()
    plt.close()


def plot_boxplot(df, f_cols, start_date, end_date):
    """
    Function to plot box plots of specified columns in a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    f_cols (list): List of columns to plot.
    start_date (str): Start date for filtering the DataFrame.
    end_date (str): End date for filtering the DataFrame.
    """
    # Filter the DataFrame based on the given date range
    filtered_df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]
    filtered_df = filtered_df[f_cols]

    # Plotting
    plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
    sns.boxplot(data=filtered_df)
    plt.xticks(rotation=90)  # Rotate the column names for better readability
    plt.title('Box plot of Each Column After Normalization')
    plt.savefig("normalization.png")
    plt.show()
    plt.close()


def plot_label_counts_by_ticker(df, start_date, end_date):
    """
    Function to plot the count of labels 0 and 1 for each ticker.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    start_date (str): Start date for filtering the DataFrame.
    end_date (str): End date for filtering the DataFrame.
    """
    # Filter the DataFrame based on the given date range
    filtered_df = df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # Group by 'ticker' and 'label' and count the occurrences
    label_counts = filtered_df.groupby(['ticker', 'label']).size().unstack()

    # Plot
    label_counts.plot(kind='bar', stacked=True, figsize=(15, 8))
    plt.title('Count of Labels 0 and 1 for Each Ticker')
    plt.xlabel('Ticker')
    plt.ylabel('Count')
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.savefig("label_count.png")
    plt.show()
    plt.close()


def plot_correlation_matrix(df, feature_cols):
    """
    Function to calculate and plot the correlation matrix for the specified features.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    feature_cols (list): List of feature column names for which the correlation matrix will be calculated.
    """
    # Calculate the correlation matrix
    corr_matrix = df[feature_cols].corr()

    # Plot the heatmap
    plt.figure(figsize=(15, 15))
    sns.heatmap(corr_matrix, cmap='coolwarm', cbar=True, square=True, linewidths=.5, annot=False, fmt='.2f')
    plt.title("Correlation Matrix")
    plt.savefig("corr_mat.png")
    plt.show()
    plt.close()