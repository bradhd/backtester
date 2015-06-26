from backtester.numpy_backtester import *
from backtester.strategies.strategy import Strategy

class ExampleStrategy(Strategy):

  param_names = ['PT','SL','SLACK']
  exit_signals = ['EOD','PT','SL']

  def __init__(self, instrument, prices_name, price_data, start_date, end_date, initial_capital, **params):
    Strategy.__init__(self, instrument, prices_name, price_data, start_date, end_date, initial_capital, **params)


  def generate_trades(self, return_trades=False):
    self.runtime_start = time.time()
    '''
    0 - open
    1 - high
    2 - low
    3 - close
    4 - volume
    '''
    self.balance = self.CAPITAL

    dailyBarsT = np.transpose(self.dailyBars)
    dailyOpens = dailyBarsT[0]
    dailyHighs = dailyBarsT[1]
    dailyLows = dailyBarsT[2]
    dailyCloses = dailyBarsT[3]

    entry_condition = dailyOpens < shift(dailyLows,1,np.nan)*(1+self.SLACK)

    # Use entry_condition as boolean mask -- we don't do anything on days where entry_condition is False.
    
    self.dailyBars = self.dailyBars[entry_condition]
    self.firstMinuteIndex = self.firstMinuteIndex[entry_condition]
    self.uniqueDates = self.uniqueDates[entry_condition]

    # iterate over trading days.
    for i in xrange(len(self.dailyBars)):
      if self.balance <= 0:
          self.in_the_red = True
          break

      if i < len(self.dailyBars) - 1:
        todayMinuteBars = self.minuteBars[self.firstMinuteIndex[i]:self.firstMinuteIndex[i + 1]]
      else:
        todayMinuteBars = self.minuteBars[self.firstMinuteIndex[i]:]

      N = len(todayMinuteBars) # number of bars today
      if N < self.instrument.minimum_traded_minutes_per_day: continue # skip those anomalous days with an insignificant volume of trades

      # we've already filtered out those days on which we don't enter at the open, so... enter at the open and set PT and SL exit prices.
      self.enter_long(entryPrice=dailyOpens[i], entryTime = self.minuteTimes[self.firstMinuteIndex[i]], entrySignal = 'OpenDrive long entry')
      self.ptPrice = tickround((1+self.PT)*(self.entryPrice), self.instrument.tick, direction='up')
      self.slPrice = tickround((1-self.SL)*(self.entryPrice), self.instrument.tick, direction='down')

      # iterate over today's bars.
      for j in xrange(2,N):
        if self.balance <= 0:
          self.in_the_red = True
          break
        if todayMinuteBars[j][2] <= self.slPrice - self.instrument.tick:
          self.exit_long(exitPrice = min(todayMinuteBars[j][0], self.slPrice), exitTime=self.minuteTimes[self.firstMinuteIndex[i] + j], exitSignal='SL')
          break # go to next day

        else: # we don't exit on SL on this bar. do we exit on PT?
          if todayMinuteBars[j][1] >= self.ptPrice + self.instrument.tick:
            self.exit_long(exitPrice = self.ptPrice, exitTime=self.minuteTimes[self.firstMinuteIndex[i] + j], exitSignal='PT')
            break # go to next day

      # now at end of day. If we're long, get out.
      if self.position == 1:
        self.exit_long(exitPrice=todayMinuteBars[-1][3], exitTime=self.minuteTimes[self.firstMinuteIndex[i] + N-1], exitSignal='EOD')
      if self.balance <= 0:
        self.in_the_red = True
        break

    self.cleanup()
    assert(self.trades_df is not None)
    if return_trades: return self.trades_df