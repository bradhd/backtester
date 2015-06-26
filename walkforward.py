import os, itertools, time
import numpy as np
import pandas as pd
import datetime as dt
from backtester.utils import opt, sendemail
from backtester import instruments, strategies
import backtester.utils.sensitivity as sen
import backtester.numpy_backtester as nb
from dateutil.relativedelta import relativedelta


def walkforward(metric, strat, instrument, pricename, pricedata, mins, maxes, rangelengths, insample_period, outsample_period, annual_trades_bound, win_pct_bound, metric_min_quantile, metric_max_quantile, first_date = None, last_date = None):

  pars = strat.param_names
  
  in_delta = relativedelta(months=insample_period, days=-1)
  out_delta = relativedelta(months=outsample_period, days=-1)
  inwindow_starts = []
  inwindow_ends = []
  outwindow_starts = []
  outwindow_ends = []
  
  if first_date is None:
    first_date = pricedata[2][0]
  if last_date is None:
    last_date = pricedata[2][-1]

  this_inwindow_start = first_date
  this_inwindow_end = this_inwindow_start + in_delta
  this_outwindow_start = this_inwindow_end + relativedelta(days=1)
  this_outwindow_end = this_outwindow_start + out_delta

  while True:
      print 'window ends: ', this_inwindow_start, this_inwindow_end, this_outwindow_start, this_outwindow_end
      inwindow_starts.append(this_inwindow_start)
      inwindow_ends.append(this_inwindow_end)
      outwindow_starts.append(this_outwindow_start)
      outwindow_ends.append(this_outwindow_end)
      if this_outwindow_end >= last_date:
        break
      this_inwindow_start += (out_delta + relativedelta(days=1))
      this_inwindow_end += (out_delta + relativedelta(days=1))
      this_outwindow_start += (out_delta + relativedelta(days=1))
      this_outwindow_end += (out_delta + relativedelta(days=1))
      if this_outwindow_end > last_date:
        this_outwindow_end = last_date
  '''
  this_outwindow_end = last_date
  this_outwindow_start = last_date - out_delta
  this_inwindow_end = this_outwindow_start - relativedelta(days=1)
  this_inwindow_start = this_inwindow_end - in_delta

  while this_inwindow_start >= first_date:
    print 'window ends: ', this_inwindow_start, this_inwindow_end, this_outwindow_start, this_outwindow_end
    inwindow_starts.append(this_inwindow_start)
    inwindow_ends.append(this_inwindow_end)
    outwindow_starts.append(this_outwindow_start)
    outwindow_ends.append(this_outwindow_end)

    this_inwindow_start -= (out_delta + relativedelta(days=1))
    this_outwindow_end -= (out_delta + relativedelta(days=1))
    this_outwindow_start -= (out_delta + relativedelta(days=1))
    this_inwindow_end -= (out_delta + relativedelta(days=1))
  '''

  assert len(inwindow_starts) == len(outwindow_ends)

  rows = []
  trades_limit = annual_trades_bound*(insample_period/12.0)
  
  for i in xrange(len(inwindow_starts)):
    print 'computing in_grid from %s to %s...' % (str(inwindow_starts[i]), str(inwindow_ends[i]))
    in_grid, in_grid_filepath = opt.do_grid_search(strat, instrument, pricename, pricedata, mins, maxes, rangelengths, inwindow_starts[i], inwindow_ends[i], emailme=False, return_df = True)
    print 'computing out_grid from %s to %s...' % (str(outwindow_starts[i]), str(outwindow_ends[i]))
    out_grid, out_grid_filepath = opt.do_grid_search(strat, instrument, pricename, pricedata, mins, maxes, rangelengths, outwindow_starts[i], outwindow_ends[i], emailme=False, return_df = True)
    this_row_dict = get_step(metric, pars, in_grid, out_grid, trades_limit, win_pct_bound, metric_max_quantile, in_grid_filepath = in_grid_filepath, outsample_length=outsample_period)
    this_row_dict['in_start'] = inwindow_starts[i]
    this_row_dict['in_end'] = inwindow_ends[i]
    this_row_dict['out_start'] = outwindow_starts[i]
    this_row_dict['out_end'] = outwindow_ends[i]
    rows.append(this_row_dict)
    print this_row_dict

  df = pd.DataFrame(rows)

  pcols = ['argmax_'+p for p in pars]
  newcols = ['in_start','in_end','out_start','out_end'] + pcols + ['pop_size', 'correlation', 'in_max', 'in_num_trades', 'out_result', 'out_result_quantile','outsample_profit'] + ['outsample_max_drawdown','outsample_carr','outsample_win_pct', 'outsample_num_trades']
  df = df[newcols]
  for x in ['mean','min','max']: df.loc[x] = df.describe().ix[x]
  df.loc['sum'] = df.sum()
  folder_path = 'c:\\backtester\\walkforward\\%s' % strat.__name__
  if not os.path.isdir(folder_path):
    os.mkdir(folder_path)
  df.to_csv('c:\\backtester\\walkforward\\%s\\walkforward_(sym_%s)_(in_%s)_(out_%s)_(win_pct_bound_%s)_(met_ub_%s).csv' % (strat.__name__, instrument.symbol, str(insample_period), str(outsample_period), str(win_pct_bound), str(metric_max_quantile)))

def get_step(metric, pars, in_grid, out_grid, trades_bound, win_pct_bound, metric_max_quantile, in_grid_filepath, outsample_length):
  """
  in_grid and out_grid are metrics results of grid search (with the same grid!) on insample and outsample windows respectively.
  Return a dict with keys 'in_start', 'in_end', 'out_start', 'out_end', 'in_max', 'pop_size', 'out_result', and 'argmax_p' for p in pars, where:
    - in_start (resp. out_start) is the start date for the insample (resp. outsample) period; similarly for in_end and out_end.
    - in_max is the best result of the metric on the insample subject to constraints num_trades >= trades_bound and win_pct >= 60
    - argmax_p is the value of the parameter p of the point that gave the best result on the insample
    - pop_size is the number of all parameter values from the top decile wrt the metric that satisfied the above constraints
    - out_result is the metric value obtained from the argmax parameters on the outsample window
  """

  assert(len(in_grid) == len(out_grid)), 'DIFFERENT SEARCH SPACE LENGTHS FOR IN OUT: %d, %d' % (len(in_grid), len(out_grid))

  # reindex out_grid with tuples of step counts. (done for in_grid by attach_sensitivity_metric.)
  print 'reindexing in_grid...'
  in_grid.index = sen.grid_search_index(in_grid, pars)
  print 'reindexing out_grid...'
  out_grid.index = sen.grid_search_index(out_grid, pars)

  metric_ub = in_grid[metric].quantile(metric_max_quantile)

  bounds = {} # tuples (b, sgn) representing {x >= b} if sgn = +1, resp. {x <= b} if sgn = -1

  bounds['win_pct'] = (win_pct_bound, +1)
  bounds['num_trades'] = (trades_bound, +1)

  #print in_grid.head()

  # impose bounds
  for c in bounds.keys():
    if bounds[c][1] == -1:
      in_grid = in_grid[in_grid[c] <= bounds[c][0]]
    elif bounds[c][1] == +1:
      in_grid = in_grid[in_grid[c] >= bounds[c][0]]
    else:
      assert(False), 'bad bounds'

  in_grid = in_grid[(in_grid[metric] <= metric_ub)]

  result = {}
  constrained_index = in_grid.index
  if len(constrained_index) == 0:
    result['pop_size'] = 0
    result['correlation'] = np.nan
    result['in_max'] = np.nan
    for p in pars:
      result['argmax_'+p] = np.nan
    result['out_result'] = np.nan
  else:
    result['pop_size'] = len(constrained_index)
    #print in_grid.head()
    #print '===='
    #print out_grid.ix[constrained_index].head()
    result['correlation'] = in_grid[metric].corr(out_grid.ix[constrained_index][metric])
    z1 = in_grid[metric].mean() + in_grid[metric].std()
    argmax_index = in_grid[metric].argmax()
    result['in_max'] = in_grid[metric].max()
    result['in_num_trades'] = in_grid['num_trades'][argmax_index]
    for p in pars:
      result['argmax_'+p] = in_grid[p][argmax_index]
    print 'argmax_index = %s' % str(argmax_index)
    '''print 'out_grid head: %s' % str(out_grid.head())
    print 'in_grid head: %s' % str(in_grid.head())'''
    result['out_result'] = out_grid[metric][argmax_index]
    result['out_result_quantile'] = sorted(out_grid[metric]).index(out_grid[metric][argmax_index])/(1.0*len(out_grid))
    result['out_max_with_constraints'] = out_grid.ix[constrained_index][metric].max()
    result['outsample_profit'] = out_grid['total_profit'][argmax_index]
    result['outsample_max_drawdown'] = out_grid['max_drawdown'][argmax_index]
    result['outsample_carr'] = (result['outsample_profit']/(outsample_length/12.0))/(-1*result['outsample_max_drawdown'])
    result['outsample_win_pct'] = out_grid['win_pct'][argmax_index]
    result['outsample_num_trades'] = out_grid['num_trades'][argmax_index]
    
  return result