#

import sys
import os
import time
PROJECT_DIR = '/Users/ershov/Desktop/Projects/Trading_Bot/option_skew_project/'
sys.path.append(PROJECT_DIR)

from pandas_datareader.data import Options
import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format
import numpy as np

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from file_handler import file_handler
import pandas_datareader as pdr

# -----------------------------------------------------------------------------
# import symbols
# -----------------------------------------------------------------------------
symbols = (pd.read_csv(PROJECT_DIR+'data/symbols.csv', header=None, index_col=False).rename(columns={0:'symbols'}))

# ------------------------------------------------------------------------------
# define conv. fn.
# ------------------------------------------------------------------------------
def cprint(df):
    print('-'*50)
    print(df.sample(5))
    print()
    print(df.info())
    print()

def random_wait():
    """fn: randomly choose a wait time based on probability"""
    wait_times = [0.2, 0.5, 1, 2]
    probs = [0.3, 0.4, 0.2, 0.1 ]
    choice = np.random.choice(wait_times, size=1, p=probs)
    return choice

# ------------------------------------------------------------------------------
# init file handler
# ------------------------------------------------------------------------------

fh = file_handler(PROJECT_DIR)

# ------------------------------------------------------------------------------
# run aggregation func
# ------------------------------------------------------------------------------
errors = []
dfs_dict = {}

for sym in tqdm(symbols.symbols.values):
    print('-'*50)
    print('downloading {} ...'.format(sym))
    try:
        tmp_df = Options(sym, 'yahoo').get_all_data()
        dfs_dict[sym] = tmp_df
    except Exception as e:
        errors.append(sym)
        print('{} error: {}'.format(sym, e))
        continue
    else:
        print('{} complete'.format(sym))
        print()
        time.sleep(random_wait())

# ------------------------------------------------------------------------------
# concat dfs drop unnecessary columns
# ------------------------------------------------------------------------------
data = (pd.concat(list(dfs_dict.values())).drop(['JSON'], axis=1))
error_series = pd.Series(errors)

cprint(data)
print(error_series)

# ------------------------------------------------------------------------------
# save data
# ------------------------------------------------------------------------------

fh.save_data(error_series, format='csv', resolution='date', errors=True)
try:
    fh.save_data(data, format='parquet')
except Exception as e:
    print(e)

fh.save_data(data, format='h5')
