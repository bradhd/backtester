__author__ = 'Brad Hannigan-Daley'
from backtester.numpy_backtester import *
import time
import pandas as pd

class Strategy(object):
  '''
  Abstract single-instrument Strategy class. Not to be itself instantiated!
  initialize with a dictionary of parameters. If initial_capital < 0, take the first open instead.
  '''
  param_names = []
  entry_signals = []
  exit_signals = []
  def __init__(self, instrument, prices_name, price_data, start_date, end_date, initial_capital, **params):
    self.params = params
    assert (sorted(params.keys()) == type(self).param_names), 'bad parameters for %s: %s' % (type(self).param_names, str(sorted(params.keys())))
    self.__dict__.update(params)
    self.prices_name = prices_name
    self.totalCommissions = 0
    self.trades = []
    self.balance = 0
    self.position = 0
    self.bars_long = 0
    self.has_run = False
    self.trades_df = None
    self.wins_df = None
    self.losses_df = None
    self.annotated_name = ''
    self.runtime_start = -1
    self.runtime_end = -1
    self.start_date = start_date
    self.end_date = end_date
    self.minuteBars, self.dailyBars, self.uniqueDates, self.firstMinuteIndex, self.minuteTimes = restrict_time(price_data[0], price_data[1], price_data[2], price_data[3], price_data[4], start_date, end_date)
    self.in_the_red = False # if we ever hit a negative balance, set this to True and stop trading

    self.instrument = instrument
    if initial_capital > 0:
      self.CAPITAL = initial_capital
    else:
      self.CAPITAL = self.minuteBars[0][0] * self.instrument.multiplier
    self.balance = self.CAPITAL
    
  def annotated_trade_string(self):
    param_string = ''
    for p in self.param_names:
      param_string += p + ' ' + str(self.params[p]) + '\n'
    return param_string + '\n' + self.generate_trades(return_trades=True).to_csv() 

  def generate_trades(self, return_trades=False):
    '''
    return a pandas dataframe with the trades resulting from executing this strategy on the given historical minute bars (prices_df) between the given dates (inclusive) with the given initial capital (initial_capital) and multiplier.
    prices_df indexed by Timestamp objects, columns are open, high, low, close, volume. No NaNs.
    columns of the output are ['entry_time','entry_price','exit_time','exit_price','profit','exit_signal','balance']'''
    raise NotImplementedError("Need to implement generate_trades()!")

  # write this Strategy's metrics to a CSV
  def write_metrics_to_csv(self, filepath):
    assert(self.has_run), 'Called write_report() on a Strategy that hasn\'t been run!'
    #if folder is None:
    output_folder = r'C:\backtester\backtester_output\metrics\%s\%s\%s' % (type(self).__name__, self.prices_name, str(self.start_date) + '_' + str(self.end_date))
    #else:
    #  output_folder = folder
    output_filename = output_folder + '\\' + self.get_annotated_name() + '.csv'
    if not os.path.isdir(output_folder):
      os.makedirs(output_folder)

    num_trades = len(self.trades_df)

    win_pct = 100.0 * float(len(self.wins_df)) / len(self.trades_df)
    exitSignalNames = type(self).exit_signals
    exitSignalPcts = {}
    ser = self.trades_df.groupby('exit_signal').profit.count()
    n = float(sum(ser))
    for x in ser.iteritems():
      exitSignalPcts[x[0]] = (100.0*x[1])/n
    exitSignalPcts = [exitSignalPcts.setdefault(x,0) for x in exitSignalNames]
    exitSignalNames = map(lambda s:s + '_pct',exitSignalNames)
    avg_profit_per_winner = round(np.mean(self.wins_df.profit), 2)
    avg_loss_per_loser = round(np.mean(self.losses_df.profit), 2)

    if self.in_the_red or num_trades == 1:
      vals = []
      for p in self.param_names:
        vals.append(self.params[p])
      #for x in self.params.iteritems():
      #  vals.append(x[1])
      vals = vals + [-np.inf, num_trades, -np.inf, win_pct] + exitSignalPcts + [avg_profit_per_winner, avg_loss_per_loser, -np.inf, -np.inf, -np.inf, str(self.start_date), str(self.end_date) ]

      header_filename = output_folder + '\\header.txt'
      if not os.path.isfile(header_filename):
        headers = sorted(self.param_names) + ['total_profit', 'num_trades', 'max_drawdown','win_pct'] + exitSignalNames + ['avg_profit_per_winner','avg_loss_per_loser','sharpe','sortino','kratio','start_date','end_date']
        with open(header_filename,'wb') as f:
          wr = csv.writer(f)
          wr.writerow(headers)

      with open(output_filename, 'wb') as f:
        wr = csv.writer(f)
        wr.writerow(vals)

    else:
      initial_capital_filename = output_folder + '\\initial_capital.txt'
      if not os.path.isfile(initial_capital_filename):
        with open(initial_capital_filename,'w') as f:
          f.write(str(self.CAPITAL))

      total_prof = sum(self.trades_df.profit)
      mmax_drawdown =  max_drawdown(pd.concat([Series(self.CAPITAL), self.trades_df['balance']]))

      eod_bals = eod_balances(self.trades_df, self.uniqueDates, self.CAPITAL)
      ret = returns(eod_bals, self.CAPITAL, cumulative=False)
      cmret = returns(eod_bals, self.CAPITAL, cumulative=True)
      sharpe = sharpe_ratio(ret) * np.sqrt(252)
      sortino = sortino_ratio(ret) * np.sqrt(252)
      k_ratio = kratio(cmret,252)[0]

      headers = []
      vals = []
      for p in self.param_names:
        headers.append(p)
        vals.append(self.params[p])
      vals = vals + [total_prof, num_trades, mmax_drawdown, win_pct] + exitSignalPcts + [avg_profit_per_winner, avg_loss_per_loser, sharpe, sortino, k_ratio, str(self.start_date), str(self.end_date)]

      header_filename = output_folder + '\\header.txt'
      if not os.path.isfile(header_filename):
        headers = headers + ['total_profit', 'num_trades', 'max_drawdown','win_pct'] + exitSignalNames + ['avg_profit_per_winner','avg_loss_per_loser','sharpe','sortino','kratio','start_date','end_date']
        with open(header_filename,'wb') as f:
          wr = csv.writer(f)
          wr.writerow(headers)

    with open(output_filename, 'wb') as f:
      wr = csv.writer(f)
      wr.writerow(vals)

  # return a string that summarizes this Strategy's parameters and metrics
  def write_report(self, print_it = False):
    assert(self.has_run), 'Called write_report() on a Strategy that hasn\'t been run!'
    output_folder = r'C:\backtester\backtester_output\txt\%s\%s\%s' % (type(self).__name__, self.prices_name, str(self.start_date) + '_' + str(self.end_date))
    output_filename = output_folder + '\\' + self.get_annotated_name() + '.txt'
    if not os.path.isdir(output_folder):
      os.makedirs(output_folder)

    report_line_list = []
    report_line_list.append('---------\nPrice Data\n---------')
    report_line_list.append('Instrument: %s' % self.instrument.desc)
    report_line_list.append('Prices name: %s' % self.prices_name)
    report_line_list.append('Start date: %s' % str(self.start_date))
    report_line_list.append('End date: %s' % str(self.end_date))
    report_line_list.append('---------\nParameters\n---------')
    for k in self.params.keys():
        report_line_list.append('%s = %s' % (k, self.params[k]))
    report_line_list.append('---------\nStatistics\n---------')
    report_line_list.append('Initial capital: %s' % str(self.CAPITAL))
    report_line_list.append('Total profit: %s' % str(sum(self.trades_df.profit)))
    report_line_list.append('Final balance: %s' % str(self.balance))
    report_line_list.append('Total commissions: %s' % str(self.totalCommissions))
    report_line_list.append('Number of trades: %d' % len(self.trades_df))
    report_line_list.append('Average number of trades per day: %f' % (len(self.trades_df)/float(len(self.uniqueDates))))
    if 'num_days' in self.trades_df.columns: report_line_list.append('Average # days per trade: %f' % np.mean(self.trades_df.num_days))
    report_line_list.append('Max drawdown: %f' % max_drawdown(self.trades_df['balance']))
    report_line_list.append('Winning percentage: %f' % (100.0 * float(len(self.wins_df)) / len(self.trades_df)))
    ser = self.trades_df.groupby('exit_signal').profit.count()
    n = float(sum(ser))
    for x in ser.iteritems():
      report_line_list.append('%% exits on signal \'%s\': %f' % (x[0],100.0*x[1]/n))
    report_line_list.append('Mean profit per winning trade: %s' % str(round(np.mean(self.wins_df.profit), 2)))
    report_line_list.append('Mean loss per losing trade: %s' % str(round(np.mean(self.losses_df.profit), 2)))
    #report_line_list.append_win_loss(df)
    #if printed: print_win_loss(df)
    eod_bals = eod_balances(self.trades_df, self.uniqueDates, self.CAPITAL)
    ret = returns(eod_bals, self.CAPITAL, cumulative=False)
    cmret = returns(eod_bals, self.CAPITAL, cumulative=True)
    msharpe = sharpe_ratio(ret)
    msortino = sortino_ratio(ret)
    k_ratio = kratio(cmret,252)
    report_line_list.append('Daily Sharpe: %f' % msharpe)
    report_line_list.append('Annualized Sharpe: %f' % (msharpe * np.sqrt(12)))
    report_line_list.append('Daily Sortino: %f' % msortino)
    report_line_list.append('Annualized Sortino: %f' % (msortino * np.sqrt(12)))
    report_line_list.append('Kestner K-ratio: %f' % k_ratio[0])
    report_line_list.append('Slope of regression line of cumulative returns: %f' % k_ratio[1])
    report_line_list.append('Wall time: %s seconds' % str(self.runtime_end - self.runtime_start))

    report_string = '\n'.join(report_line_list)

    with open(output_filename, 'w') as f:
       f.write(report_string)

    if print_it: print(report_string)

  # take commission on a trade.
  def take_commission(self):
    self.balance -= self.instrument.commission
    self.totalCommissions += self.instrument.commission

  def get_annotated_name(self):
    if self.annotated_name == '':
      keys = sorted(self.params.keys())
      paramvals = [self.params[k] for k in keys]
      self.annotated_name = type(self).__name__+'_'+self.prices_name+'-'+'-'.join([str(x) for x in paramvals])
    return self.annotated_name

  def write_trades_to_csv(self):
    assert(self.has_run), 'Called write_trades_to_csv() on a Strategy that hasn\'t been run!'
    output_folder = r'C:\backtester\backtester_output\csv\%s\%s\%s' % (type(self).__name__, self.prices_name, str(self.start_date) + '_' + str(self.end_date))
    output_filename = output_folder + '\\' + self.get_annotated_name() + '.csv'
    if not os.path.isdir(output_folder):
      os.makedirs(output_folder)
    self.trades_df.to_csv(output_filename)
    initial_capital_filename = output_folder + '\\initial_capital.txt'
    if not os.path.isfile(initial_capital_filename):
      with open(initial_capital_filename,'w') as f:
        f.write(str(self.CAPITAL))

  def cleanup(self):
    """
    Construct trades dataframes, set has_run flag, end runtime.
    """
    if len(self.trades) == 0:
      self.trades = [{'position': 0, 'entry_time': self.minuteTimes[0], 'exit_time': self.minuteTimes[-1],
             'entry_price': -1, 'exit_price': -1,
             'profit': -1,
             'entry_signal': 'NO TRADES MADE', 'exit_signal': 'NO TRADES MADE', 'num_days': -1, 'balance': self.balance}]
    self.trades_df = DataFrame(self.trades)[['entry_time', 'entry_price', 'entry_signal', 'exit_time', 'exit_price', 'exit_signal', 'profit', 'balance']]
    self.wins_df = self.trades_df[self.trades_df.profit > 0]
    self.losses_df = self.trades_df[self.trades_df.profit <= 0]
    self.has_run = True
    self.runtime_end = time.time()

  def enter_long(self, entry_price, entry_time, entry_signal):
    """
    Enter a long position at specified price, time and signal.
    """
    assert (
      self.position == 0), 'tried to enter long position on %s with signal %s when we already had a position' % (
      str(entry_time), entry_signal)
    self.take_commission()
    self.entry_price = entry_price
    self.entry_time = entry_time
    self.entry_signal = entry_signal
    self.position = 1
    self.entered_long_today = True


  def exit_long(self, exit_price, exit_time, exit_signal):
    """
    Exit a long position at specified price, time and signal.
    """
    assert (
      self.position == 1), 'tried to exit long position on %s with signal %s when we weren\'t long' % (
      str(exit_time), exit_signal)
    self.take_commission()
    self.exit_price = exit_price
    self.exit_time = exit_time
    self.exit_signal = exit_signal
    self.balance += (self.exit_price - self.entry_price) * self.instrument.multiplier
    self.exited_long_today = True
    '''if self.balance < 0:
      e = Exception('REACHED A NEGATIVE BALANCE: %s' % str(self.balance))
      raise e'''
    self.record_trade()
    self.position = 0

  def enter_short(self, entry_price, entry_time, entry_signal):
    """
    Enter a short position at specified price, time and signal.
    """
    assert (
      self.position == 0), 'tried to enter short position on %s with signal %s when we already had a position' % (
      str(entry_time), entry_signal)
    self.take_commission()
    self.entry_price = entry_price
    self.entry_time = entry_time
    self.entry_signal = entry_signal
    self.position = -1
    self.entered_short_today = True

  def exit_short(self, exit_price, exit_time, exit_signal):
    """
    Exit a long position at specified price, time and signal.
    """
    assert (
      self.position == -1), 'tried to exit short position on %s with signal %s when we weren\'t short' % (
      str(exit_time), exit_signal)
    self.take_commission()
    self.exit_price = exit_price
    self.exit_time = exit_time
    self.exit_signal = exit_signal
    self.balance += (self.entry_price - self.exit_price) * self.instrument.multiplier
    self.exited_short_today = True
    '''if self.balance < 0:
      e = Exception('REACHED A NEGATIVE BALANCE: %s' % str(self.balance))
      raise e'''
    self.record_trade()
    self.position = 0

  def record_trade(self):
    if self.position == 1:
      profit = self.instrument.multiplier * (self.exit_price - self.entry_price) - 2 * self.instrument.commission
    elif self.position == -1:
      profit = self.instrument.multiplier * (self.entry_price - self.exit_price) - 2 * self.instrument.commission
    else:
      raise Exception('recording trade when self.position is neither -1 (short) nor +1 (long)')

    trade = {'position': self.position, 'entry_time': self.entry_time, 'exit_time': self.exit_time,
             'entry_price': self.entry_price, 'exit_price': self.exit_price,
             'profit': profit,
             'entry_signal': self.entry_signal, 'exit_signal': self.exit_signal, 'num_days': (self.exit_time - self.entry_time).astype('timedelta64[D]').item().days, 'balance': self.balance}
    self.trades.append(trade)
    self.position, self.entry_time, self.exit_time, self.entry_price, self.exit_price, self.entry_signal, self.exit_signal = [None] * 7
    self.bars_long = 0


class SimpleLong(Strategy):
  '''
  A simple example of a strategy: go long on the start date, then sell on the end date.
  '''


  def __init__(self, prices_name, price_data, start_date, end_date, initial_capital, **params):
    Strategy.__init__(self, initial_capital, prices_name, params)

  def generate_trades(self, return_trades = False):
    self.balance = self.CAPITAL
    self.enter_long(entry_price=self.minuteBars[0][0], entry_time=self.minuteTimes[0], entry_signal='SimpleLong entry')
    self.exit_long(exit_price=self.minuteBars[-1][3], exit_time=self.minuteTimes[-1], exit_signal='SimpleLong exit')
    self.wins_df = self.trades_df[self.trades_df.profit > 0]
    self.losses_df = self.trades_df[self.trades_df.profit <= 0]
    self.has_run = True
    if return_trades: return self.trades_df
