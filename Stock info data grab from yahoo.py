import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
import mplfinance
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

style.use('ggplot')
start = dt.datetime(2000, 1, 1)
end = dt.datetime(2020, 8, 31)

# collect stock data of Tesla from yahoo, output to csv, read back into pandas
df = web.DataReader('TSLA', 'yahoo', start, end)
df.to_csv('TSLA.csv')
df = pd.read_csv('TSLA.csv', parse_dates=True, index_col=0)

# add a 100 point moving average column
df['Moving Avg'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()

# drop the first 100 rows as there is not enough data to generate moving average
df.dropna(inplace=True)
print(df.tail())

# plot key metrics onto graph overlay
ax1 = plt.subplot2grid((12, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((12, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['Moving Avg'])
ax2.bar(df.index, df['Volume'])

# generate dataframe for candlestick plot with 10 day averaged data (ohlc = open high low close)
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_ohlc.reset_index(inplace=True)
df_volume = df['Volume'].resample('10D').sum()

# need to convert datetime objects to datatime numbers which matplotlib can read
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

# add candlestick plot and volume info to the key metrics plot
ax3 = plt.subplot2grid((12, 1), (7, 0), rowspan=5, colspan=1)
ax4 = plt.subplot2grid((12, 1), (11, 0), rowspan=1, colspan=1, sharex=ax3)
ax3.xaxis_date()

candlestick_ohlc(ax3, df_ohlc.values, width=2, colorup='g')
ax4.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

plt.show()

