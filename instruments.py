import datetime as dt

class Instrument(object):
  '''
  Simple class to encapsulate the data of a tradeable instrument.
  Attributes:
    desc - String describing the instrument.
    symbol - Two-character symbol for the instrument (e.g. ES for S&P e-mini).
    tick - Minimum tick size.
    multiplier - Multiplier. (For equities, as opposed to futures, this would be 1.)
    commission - Commission charged per trade.
    minimum_traded_minutes_per_day - This is a threshold for backtesting. If trades occur on fewer than this many
    minutes of a given day, then it is assumed that that day is an anomaly (holiday, disaster, etc.) and it is skipped.
  '''

  def __init__(self, desc, symbol, tick, multiplier, commission, pit_session_start_time, pit_session_end_time,
               minimum_traded_minutes_per_day, price_names):
    self.desc = desc
    self.symbol = symbol
    self.tick = tick
    self.multiplier = multiplier
    self.commission = commission
    self.pit_session_start_time = pit_session_start_time
    self.pit_session_end_time = pit_session_end_time
    self.minimum_traded_minutes_per_day = minimum_traded_minutes_per_day
    self.price_names = price_names
    
  def __repr__(self):
    return self.desc
    
ES = Instrument(desc = 'E-mini S&P 500 contract', 
                symbol = 'ES', 
                tick = 0.25,
                multiplier = 50,
                commission = 0.75, 
                pit_session_start_time = dt.time(9,30), 
                pit_session_end_time = dt.time(16,15), 
                minimum_traded_minutes_per_day = 200,
                price_names = ['ES_merged_Reuters_and_TickData', 'ES_Reuters', 'ES_TickData'])

ER = Instrument(desc = 'Russell 2000 Index Mini contract',
                symbol = 'ER', 
                tick = 0.1,
                multiplier = 100,
                commission = 0.75, 
                pit_session_start_time = dt.time(9,30), 
                pit_session_end_time = dt.time(16,15), 
                minimum_traded_minutes_per_day = 200,
                price_names = ['ER_merged_Reuters_and_TickData', 'ER_Reuters', 'ER_TickData'])
                
YM = Instrument(desc = 'Dow Mini contract',
                symbol = 'YM', 
                tick = 1,
                multiplier = 5,
                commission = 0.75, 
                pit_session_start_time = dt.time(9,30), 
                pit_session_end_time = dt.time(16,15), 
                minimum_traded_minutes_per_day = 200,
                price_names = ['YM_merged_Reuters_and_TickData'])