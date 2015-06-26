__author__ = 'Brad Hannigan-Daley'

from pandas import DataFrame, Series
import time, csv, os
import pandas as pd
import datetime as dt
import numpy as np
from scipy.stats import linregress


def restrict_time(minuteBars, dailyBars, uniqueDates, firstMinuteIndex, minuteTimes, start_date, end_date):
  '''
  Given minuteBars, ..., minuteTimes as in the output of get_price_data(), returns that data restricted to the interval [start_date, end_date].
  '''
  uniqueDates = np.array(uniqueDates)

  start_date = np.datetime64(start_date)
  end_date = np.datetime64(end_date)

  minute_range = (minuteTimes >= start_date) & (minuteTimes < end_date + np.timedelta64(1,'D'))
  newMinuteBars = minuteBars[minute_range]
  newMinuteTimes = minuteTimes[minute_range]


  new_day_range = (uniqueDates >= start_date) & (uniqueDates <= end_date)
  
  first_day_index = np.where(new_day_range == 1)[0][0]
  newDailyBars = dailyBars[new_day_range]
  newUniqueDates = uniqueDates[new_day_range]

  newFirstMinuteIndex = firstMinuteIndex[new_day_range] - firstMinuteIndex[first_day_index]


  '''print 'newMinuteTimes[0] = ' + str(newMinuteTimes[0])
  print 'newMinuteTimes[-1] = ' + str(newMinuteTimes[-1])
  print 'newUniqueDates[0] = ' + str(newUniqueDates[0])
  print 'newUniqueDates[-1] = ' + str(newUniqueDates[-1])

  print 'first minute of last day: ' + str(newMinuteTimes[newFirstMinuteIndex[-1]])'''



  return newMinuteBars, newDailyBars, newUniqueDates, newFirstMinuteIndex, newMinuteTimes

# PriceStore
def get_price_data(prices_name, dates_to_skip = None):
  '''
  Returns a tuple (minuteBars, dailyBars, uniqueDates, firstMinuteIndex, minuteTimes, rollDates) where:

  - minuteBars is an array of the minute bars associated to prices_name, stripped to trading session hours
  - dailyBars is an array of the daily bars associated to prices_name
  - uniqueDates is an array of the dates of said daily bars, hence len(uniqueDates) = len(dailyBars)
  - firstMinuteIndex is an array, of the same length of dailyBars and uniqueDates, defined by the property that
  minuteBars[firstMinuteIndex[i]] is the first minute bar of the day uniqueDates[i]
  - minuteTimes is an array of the times of the minute bars, hence len(minuteTimes) = len(minuteBars)
  '''
  if os.path.exists(r'C:\backtester\pricedata\preprocessed\%s' % prices_name):
    minutedf = pd.read_csv(
      r'C:\backtester\pricedata\preprocessed\%s\%s_tradingSessionMinuteBars.csv' % (prices_name, prices_name),
      parse_dates=True, index_col=0)[['open','high','low','close','volume']]
    minuteBars = minutedf.values
    dailydf = pd.read_csv(r'C:\backtester\pricedata\preprocessed\%s\%s_dailyBars.csv' % (prices_name, prices_name),
                          parse_dates=True, index_col=0)[['open','high','low','close','volume']]
    dailyBars = dailydf.values
    minuteTimes = np.array(minutedf.index)

    with open(r'C:\backtester\pricedata\preprocessed\%s\%s_uniqueDates.csv' % (prices_name, prices_name), 'rb') as f:
      reader = csv.reader(f)
      uniqueDates = reader.next()
    uniqueDates = map(lambda s: dt.datetime.strptime(s, '%Y-%m-%d').date(), uniqueDates)

    with open(r'C:\backtester\pricedata\preprocessed\%s\%s_firstMinuteIndex.csv' % (prices_name, prices_name), 'rb') as f:
      reader = csv.reader(f)
      firstMinuteIndex = np.array(reader.next(), dtype=np.dtype(np.int32))

  else:
    # generate the arrays and write the CSVs
    df = pd.read_csv(r'C:\backtester\pricedata\minutebars\%s.csv' % prices_name, index_col=0, parse_dates=True,
                     usecols=[0, 'open', 'high', 'low', 'close', 'volume'])
    df = df[['open','high','low','close','volume']]
    os.makedirs(r'C:\backtester\pricedata\preprocessed\%s' % prices_name)
    # strip to trading hours and write to CSV
    df = df.ix[df.index.indexer_between_time(dt.time(9, 30), dt.time(16, 15))]
    df.to_csv(r'C:\backtester\pricedata\preprocessed\%s\%s_tradingSessionMinuteBars.csv' % (prices_name, prices_name))

    minuteTimes = np.array(df.index)

    # generate daily OHLC and write to CSV
    daily_bars_df = minute_bars_to_daily_bars(df)
    daily_bars_df.to_csv(r'C:\backtester\pricedata\preprocessed\%s\%s_dailyBars.csv' % (prices_name, prices_name))
    dailyBars = daily_bars_df.values

    # generate array of unique dates and write to CSV
    uniqueDates = np.unique(daily_bars_df.index.values)

    with open(r'C:\backtester\pricedata\preprocessed\%s\%s_uniqueDates.csv' % (prices_name, prices_name), 'wb') as f:
      csvwriter = csv.writer(f, delimiter=',')
      csvwriter.writerow(uniqueDates)

    dates = list(df.index.date)

    firstMinuteIndex = map(dates.index, uniqueDates)

    with open(r'C:\backtester\pricedata\preprocessed\%s\%s_firstMinuteIndex.csv' % (prices_name, prices_name), 'wb') as f:
      csvwriter = csv.writer(f, delimiter=',')
      csvwriter.writerow(firstMinuteIndex)

    df['datetime'] = df.index
    minuteBars = df.values.transpose()
    minuteBars[-1] = map(lambda t: t.to_datetime(), minuteBars[-1])
    minuteBars = minuteBars.transpose()

  if dates_to_skip is not None:
    date_mask = np.array(map(lambda d: d not in dates_to_skip, uniqueDates))
    dailyBars = dailyBars[date_mask]
    uniqueDates = np.array(uniqueDates)
    uniqueDates = uniqueDates[date_mask]
    firstMinuteIndex = firstMinuteIndex[date_mask]
  return minuteBars, dailyBars, uniqueDates, firstMinuteIndex, minuteTimes

def strip_to_trading_hours(df, trading_start_time, trading_end_time):
  """ take a dataframe of minute bars and strip away all prices outside the given time range. usually 9:30 to 4:15."""
  return 


# resample minute OHLCV to daily OHLCV.
# assume that columns of df are 'open', 'high', 'low', 'close', 'volume' and that it's indexed by Timestamps or datetimes.
def minute_bars_to_daily_bars(df):
  resampling_method = {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}
  ddf = df.resample('1D', how=resampling_method).dropna()
  ddf.index = ddf.index.date
  return ddf

# return an array of all days that have prices in the given df of prices.
def get_days_traded(df, output_filename=None):
  if output_filename is None:
    return np.unique(df.index.date)
  else:
    with open(output_filename, 'w') as f:
      wr = csv.writer(f)
      wr.writerows(np.unique(df.index.date))

# Given a CSV with a single row of dates in format YYYY-MM-DD, return a DatetimeIndex with those values.
def csv_to_datetimeIndex(filename):
  with open(filename) as f:
    reader = csv.reader(f)
    dayslist = reader.next()
    return pd.to_datetime(dayslist)

# Round x to a tick. The parameter direction specifies whether to round up, down, or to nearest tick.
def tickround(x, tick_size, direction=None):
  assert (direction in {None, 'up', 'down'}), 'bad direction for tickround'
  if direction == 'up':
    return round(np.ceil(float(x) / tick_size) * tick_size, 4)
  elif direction == 'down':
    return round(np.floor(float(x) / tick_size) * tick_size, 4)
  return round(round(float(x) / tick_size) * tick_size, 4)

def eod_balances(df, uniqueDates, initialcapital):
  df['exitDate'] = df.exit_time.apply(lambda t: t.date())
  dateindices = Series([find_last(df.exitDate, d) for d in uniqueDates], index=uniqueDates)
  dateindices.fillna(method='pad', inplace=True)
  dateindices.fillna(-1)
  return dateindices.apply(lambda i: df.balance[int(i)] if i >= 0 else initialcapital)  

def returns(bals, initialcapital, cumulative=False):
  if cumulative:
    return ((bals -initialcapital)/initialcapital)
  else:
    bals = pd.concat([Series([initialcapital]), bals])
    return (bals.diff().drop(0,axis=0) / bals.shift().drop(0,axis=0))
  
  
def find_last(lst, elm):
  gen = (len(lst) - 1 - i for i, v in enumerate(reversed(lst)) if v == elm)
  return next(gen, None)  
  
# returns the monthly returns from a given dataframe of trades, with given initial capital
def monthly_returns(df, initialcapital):
  firstmonth = df.ix[0].exit_time.month
  firstyear = df.ix[0].exit_time.year
  lastmonth = df.ix[len(df) - 1].exit_time.month
  lastyear = df.ix[len(df) - 1].exit_time.year

  monthlies = []
  for y, m in month_year_iter(firstmonth, firstyear, lastmonth, lastyear):
    mdf = df[df.exit_time.apply(lambda x: (x.year, x.month)) == (y, m)]
    ntrades = len(mdf)
    if ntrades == 0:  #no trades happened that month, so carry over previous month's balance
      if len(monthlies) > 0:
        #assert((monthlies[-1]['month']+1)%12 == m%12 )
        eombalance = monthlies[-1]['eombalance']  # carry over previous month's balance
      else:
        eombalance = initialcapital
    else:
      eombalance = mdf.balance.tail(1).values[0]  # doesn't work for months where no trades happened!
    #costs = 10*ntrades + sum(mdf.entry_price)
    #pnl = sum(mdf.profit)
    #roi = pnl/costs
    monthlies.append(
      {'year': y, 'month': m, 'ntrades': ntrades, 'eombalance': eombalance})  #'costs':costs, 'pnl':pnl, 'roi':roi})

  mdf = DataFrame(monthlies)
  eombalances = np.insert(mdf.eombalance.values, 0, initialcapital)
  #print 'eombalances:'
  #print eombalances
  eombalancediffs = np.diff(eombalances)
  returns = [eombalancediffs[i] / eombalances[i] for i in xrange(len(eombalancediffs))]
  return returns

def cumulative_monthly_returns(df, initialcapital):
  firstmonth = df.ix[0].exit_time.month
  firstyear = df.ix[0].exit_time.year
  lastmonth = df.ix[len(df) - 1].exit_time.month
  lastyear = df.ix[len(df) - 1].exit_time.year
  eombalances = []
  monthlies = []
  for y, m in month_year_iter(firstmonth, firstyear, lastmonth, lastyear):
    mdf = df[df.exit_time.apply(lambda x: (x.year, x.month)) == (y, m)]
    ntrades = len(mdf)
    if ntrades == 0:  #no trades happened that month, so carry over previous month's balance
      if len(monthlies) > 0:
        #assert((monthlies[-1]['month']+1)%12 == m%12 )
        eombalance = monthlies[-1]['eombalance']  # carry over previous month's balance
      else:
        eombalance = initialcapital
    else:
      eombalance = mdf.balance.tail(1).values[0]
    #costs = 10*ntrades + sum(mdf.entry_price)
    #pnl = sum(mdf.profit)
    #roi = pnl/costs
    monthlies.append(
      {'year': y, 'month': m, 'ntrades': ntrades, 'eombalance': eombalance})
    eombalances.append(eombalance)

  creturns = [(b - initialcapital) / float(initialcapital) for b in eombalances]
  return creturns

# return the Sharpe ratio of the given series of returns
def sharpe_ratio(returns):
  return np.mean(returns) / np.std(returns)

# return the Sortino ratio of the given series of returns
def sortino_ratio(returns):
  downsides = [min(x, 0) for x in returns]
  # downsides = [x for x in returns if x<0]
  return np.mean(returns) / np.std(downsides)

# return the Kestner k-ratio of the given series of cumulative returns
def kratio(creturns, periods_per_year):
  lr = linregress(range(len(creturns)), creturns)
  return (lr[0]/lr[-1]) * np.sqrt(periods_per_year) / len(creturns), lr[0], lr[1]

def total_profit(df):
  return str(sum(df.profit))

# return an iterator returning the pairs (year, month) over the months
def month_year_iter(start_month, start_year, end_month, end_year):
  ym_start = 12 * start_year + start_month - 1
  ym_end = 12 * end_year + end_month - 1
  for ym in xrange(ym_start, ym_end + 1):
    y, m = divmod(ym, 12)
    yield y, m + 1

#slice list l into sublists of size n and return an iterator over those sublists
def chunks(l, n):
  for i in xrange(0, len(l), n):
    yield l[i:i + n]

#return the absolute (dollar) max drawdown of a pandas Series or numpy array.
def max_drawdown(ser):
  max2here = pd.expanding_max(ser)
  dd2here = ser - max2here
  return min(dd2here)

def shift(a, k, val = np.nan):
  """
  Similar to pandas Series.shift(), but it works on numpy arrays and you can specify the value with which to replace
  missing values (NaN by default).
  :param a: array-like
  :param k: integer
  :param val: float or NaN
  :return: a shifted by k with missing values replaced by val.
  """
  if k < 0: return np.append(a[-k:], [val]*(-k))
  elif k > 0: return np.append([val]*k, a[:-k])
  else: return a

def punish_below(xmin, ymin, x):
  """
  xmin is the threshold we want to punish for going below.
  If x is nonpositive, returns 0 (total failure).
  If x >= xmin, returns 1.0 (threshold is satisfied).
  If 0 < x < xmin, return value is y where (x,y) is on the line joining (0,ymin) and (xmin, 1.0).
  (Punish for failing the threshold, proportionally to the failure, with worst punishment ymin.)
  """
  assert xmin > 0
  assert ymin > 0
  if x < 0:
    return 0
  if x >= xmin:
    return 1.0
  return ymin + x*(1.0-ymin)/xmin

def downsample_returns(r):
  return np.prod([1+x for x in r])-1

def annual_autocorr(daily_returns, chunksize):
  '''
  daily_returns is a Series of daily returns indexed by date.
  
  Computes the autocorrelation of the given return series according to specified chunks.
  '''
  autocorrs = {}
  annual_returns = {}
  
  g = daily_returns.groupby(daily_returns.index.year).groups
  years = g.keys()
  for y in years:
    annual_returns[y] = daily_returns.ix[g[y]]
    chunkedreturns = pd.Series(map(downsample_returns, chunks(annual_returns[y].values, chunksize)))
    autocorrs[y] = chunkedreturns.corr(chunkedreturns.shift())
  
  return pd.Series(autocorrs), annual_returns  

  
def drawdown_df(tdf):
  '''
  Given a DataFrame of trades tdf, returns a DataFrame describing all drawdowns that occurred.
  '''
  tdf.index = tdf.exit_time
  peak_datetimes = tdf.index[tdf.balance == tdf.balance.cummax()]
  df = pd.DataFrame(index=peak_datetimes)
  df['drawdown_start'] = df.index
  df['drawdown_end'] = df['drawdown_start'].shift(-1)
  df.loc[df.index[-1], 'drawdown_end'] = tdf.index[-1]
  df['balance_at_dd_start'] = map(lambda d: tdf.balance[d], df.drawdown_start)
  df['trough_datetime'] = map(lambda r: tdf.ix[r[1].drawdown_start : r[1].drawdown_end].balance.argmin(), df.iterrows())
  df['trough_balance'] = map(lambda r: tdf.ix[r[1].drawdown_start : r[1].drawdown_end].balance.min(), df.iterrows())
  df['drawdown_pct'] = 1 - df.trough_balance / df.balance_at_dd_start
  df['drawdown_magnitude'] = df.balance_at_dd_start - df.trough_balance
  df = df[df.drawdown_pct > 0]
  df.index = range(len(df))
  return df

def array_to_drawdown_df(a):
  '''
  Given an array (a) describing an equity curve, returns a DataFrame describing all drawdowns that occurred.
  '''
  ser = pd.Series(a)
  tdf = pd.DataFrame(ser)
  tdf.index = [dt.date.today() + dt.timedelta(i) for i in xrange(len(tdf))]
  tdf.columns = ['balance']
  peak_indices = tdf.index[tdf.balance == tdf.balance.cummax()]
  df = pd.DataFrame(index=peak_indices)
  df['drawdown_start'] = df.index
  df['drawdown_end'] = df['drawdown_start'].shift(-1)
  df.loc[df.index[-1], 'drawdown_end'] = tdf.index[-1]
  df['balance_at_dd_start'] = map(lambda d: tdf.balance[d], df.drawdown_start)
  df['trough_index'] = map(lambda r: tdf.ix[r[1].drawdown_start : r[1].drawdown_end].balance.argmin(), df.iterrows())
  df['trough_balance'] = map(lambda r: tdf.ix[r[1].drawdown_start : r[1].drawdown_end].balance.min(), df.iterrows())
  df['drawdown_pct'] = 1 - df.trough_balance / df.balance_at_dd_start
  df['drawdown_magnitude'] = df.balance_at_dd_start - df.trough_balance
  df = df[df.drawdown_pct > 0]
  df.index = range(len(df))
  return df
  

'''def series_to_max_pct_drawdown(s):
  peak_indices = list(s[s==s.cummax()].index)
  peaks = s[peak_indices[:-1]].values
  if len(peaks) == 0 or peaks[0] == peaks[-1]:
    return 1.0
  troughs = np.array(map(lambda i: min(s[peak_indices[i]: peak_indices[i+1]]), xrange(len(peak_indices)-1)))
  return max((peaks-troughs)/peaks)'''


#return the percentage max drawdown of a pandas Series. (max with respect to DOLLAR amount, not percentage drop.)
def series_to_max_pct_drawdown(ser):
  max2here = pd.expanding_max(ser)
  dd2here = ser - max2here
  dd2herepct = -1.0*dd2here/max2here
  return max(dd2herepct)
  
  for i in xrange(len(peak_indices)-1):
    trough[i] = min(s[peak_indices[i]: peak_indices[i+1]])
      
def array_to_max_pct_drawdown(a):
    df = array_to_drawdown_df(a)
    if len(df>0):
      return max(df.drawdown_pct)
    else: return 0

def series_to_MAR(s, num_years):
  return ((s.iget(-1)/s.iget(0) - 1)**(1.0/num_years))/series_to_max_pct_drawdown(s)