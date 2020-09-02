import numpy as np
import bs4 as bs
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
import pandas as pd
import pandas_datareader.data as web
import os
import pickle
import requests
import seaborn as sns
from sklearn import svm, model_selection, neighbors


style.use('ggplot')


# Extracts the tickers for the s&p 500 from wikipedia table and places them into a list.
def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'id': 'constituents'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.find('td').text.strip()
        if "." in ticker:
            ticker = ticker.replace('.', '-')
            print('ticker replaced to', ticker)
        tickers.append(ticker)

# Pickle is used to convert the python object hierarchy into a byte stream.
    with open('sp500_tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)

    print(tickers)

    return tickers

# Individual functions only for testing purposes, they are incorporated into final function
# save_sp500_tickers()


# Function to iterate through list of tickers and extract company data from yahoo.
# Puts each company data into a csv in a folder called stock_dfs.
# Iterating through all companies to get pricing data takes time (>20 mins) so save locally.
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open('sp500_tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2020, 8, 31)

    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            print(ticker)
            df = web.get_data_yahoo(ticker, start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

# Individual functions only for testing purposes, they are incorporated into final function
# get_data_from_yahoo()


# Compile adjusted close data for all companies into a pandas dataframe and export as a csv
def compile_data():
    with open("sp500_tickers.pickle", 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')

# Individual functions only for testing purposes, they are incorporated into final function
# compile_data()


# The combined csv file is loaded into pandas and df.corr used to create a correlation table.
# Pct change is used to convert from stock prices to stock returns as it os more useful for finance.
# Correlation plot is generated using seaborn.
def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    df.set_index('Date', inplace=True)
    df_corr = df.pct_change().corr()
    print(df_corr.tail())
    sns.heatmap(df_corr, annot=False, cmap='RdYlGn')
    plt.show()

# visualize_data()


# Each model will be on a per company basis
def process_data_for_labels(ticker):
    how_many_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, how_many_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

# Individual functions only for testing purposes, they are incorporated into final function.
# process_data_for_labels('XOM')


# Function to help create labels, may need to alter requirement value
def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)],
                                              ))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', str_vals)

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(0, inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df


# extract_featuresets('XOM')


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)
    clf = neighbors.KNeighborsClassifier()

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    # Buy, sell, hold so 33% 'accuracy' would be the random outcome
    print('Accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('Predicted spread:', predictions)

    return confidence


do_ml('AMZN')




