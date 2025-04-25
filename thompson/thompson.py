""" This function performs the multi-armed bandit (mab) problem by Thompson Sampling, UCB-Upper confidence Bound, and randomized sampling

   import thompson as mab

    df  = mab.example_data()
	out = mab.thompson(df, <optional>)
	out = mab.UCB(df, <optional>)
	out = mab.UCB_random(df, <optional>)
	fig = mab.plot(out, <optional>)

    df : [pd.DataFrame], Contains samples[rows] x features[columns]

    Returns
    -------
    Dictionary containing keys with results and others to make the plot.
"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : thompson.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : Jan. 2019
#--------------------------------------------------------------------------

#%% Libraries
import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import requests
import logging

logger = logging.getLogger(__name__)

#%% Plot
def plot(out, width=15, height=10, verbose='info'):
    '''
    Parameters
    ----------
    out : [dict], Output from thompson, ucb or usb_random.
    width : [Int],  Width of the figure. default is 30.
    height : [Int],  Width of the figure. default is 10.

    Returns
    -------
    None.

    '''
    set_logger(verbose)
    logger.info('Making plot')
    if out['methodtype']=='thompson':
        makefig_thompson(out, width=width, height=height)
    elif out['methodtype']=='UCB':
        makefig_UCB(out, width=width, height=width)
    elif out['methodtype']=='UCB_random':
        makefig_UCB_random(out, width=width, height=height)

#%% Thompson sampling method
def thompson(df, verbose='info'):
    """

    Parameters
    ----------
    df : [pd.DataFrame], Contains samples[rows] x features[columns]
    
    verbose : [str, int], default is 'info' or 20
        Set the verbose messages using string or integer values.
        * [0, 60, None, 'silent', 'off', 'no']: No message.
        * [10, 'debug']: Messages from debug level and higher.
        * [20, 'info']: Messages from info level and higher.
        * [30, 'warning']: Messages from warning level and higher.
        * [50, 'critical', 'error']: Messages from critical level and higher.

    Returns
    -------
    Dictionary containing keys with results and others to make the plot.

    """
    N=df.shape[0]
    d=df.shape[1]
    cols_selected = []
    numbers_of_rewards_1 = [0] * d
    numbers_of_rewards_0 = [0] * d
    total_reward = 0
    logger.info('Compute multi-armed bandit')
    # Run over the rows
    for n in range(0, N):
        col = 0
        max_random = 0

        # Run over the columns
        for i in range(0, d):
            random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
            if random_beta > max_random:
                max_random = random_beta
                col = i

        cols_selected.append(col)
        reward = df.values[n, col]
        if reward == 1:
            numbers_of_rewards_1[col] = numbers_of_rewards_1[col] + 1
        else:
            numbers_of_rewards_0[col] = numbers_of_rewards_0[col] + 1
        total_reward = total_reward + reward

    # Output results
    out=dict()
    out['columns']=df.columns.values
    out['total_reward']=total_reward
    out['cols_selected']=cols_selected
    out['cols_rewards_1']=numbers_of_rewards_1
    out['cols_rewards_0']=numbers_of_rewards_0
    out['methodtype']='thompson'
    return out

#%% Random sampling method
def UCB_random(df, verbose='info'):
    '''

    Parameters
    ----------
    df : [pd.DataFrame], Contains samples[rows] x features[columns]
    
    verbose : [str, int], default is 'info' or 20
        Set the verbose messages using string or integer values.
        * [0, 60, None, 'silent', 'off', 'no']: No message.
        * [10, 'debug']: Messages from debug level and higher.
        * [20, 'info']: Messages from info level and higher.
        * [30, 'warning']: Messages from warning level and higher.
        * [50, 'critical', 'error']: Messages from critical level and higher.

    Returns
    -------
    Dictionary containing keys with results and others to make the plot.

    '''
    set_logger(verbose)
    logger.info('Create USB Random')
    cols_selected = []
    total_reward = 0
    N=df.shape[0]
    d=df.shape[1]

    # 1. For each row, randomly pick a column.
    # 2. Check whether this was a real one. if yes, it gets the reward
    # 3. Sum up the reward for the column
    for n in range(0, N):
        col = random.randrange(d)
        cols_selected.append(col)
        reward = df.values[n, col]
        total_reward = total_reward + reward
    
    # Output results
    out=dict()
    out['columns']=df.columns.values
    out['total_reward']=total_reward
    out['cols_selected']=cols_selected
    out['cols_rewards_1']=None
    out['cols_rewards_0']=None
    out['methodtype']='UCB_random'
    return(out)

#%% Upper Confidence Bound Algorithm
def UCB(df, verbose='info'):
    '''

    Parameters
    ----------
    df : [pd.DataFrame], Contains samples[rows] x features[columns]
    
    verbose : [str, int], default is 'info' or 20
        Set the verbose messages using string or integer values.
        * [0, 60, None, 'silent', 'off', 'no']: No message.
        * [10, 'debug']: Messages from debug level and higher.
        * [20, 'info']: Messages from info level and higher.
        * [30, 'warning']: Messages from warning level and higher.
        * [50, 'critical', 'error']: Messages from critical level and higher.

    Returns
    -------
    Dictionary containing keys with results and others to make the plot.

    '''
    set_logger(verbose)
    logger.info('Compute UCB-Upper confidence Bound.')
    N=df.shape[0]
    d=df.shape[1]
    cols_selected = []
    # At start, each column has no reward yet. Start at 0.
    num_selections = [0] * d
    sum_rewards = [0] * d
    total_reward = 0

    # Run over all rows (rounds)
    for n in range(0, N):
        col = 0
        max_upper_bound = 0
        # Run over all columns (samples)
        for i in range(0, d):
            if (num_selections[i] > 0):
                average_reward = sum_rewards[i] / num_selections[i]
                delta_i = math.sqrt(3/2 * math.log(n + 1) / num_selections[i])
                upper_bound = average_reward + delta_i
            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                col = i
        cols_selected.append(col)
        num_selections[col] = num_selections[col] + 1
        reward = df.values[n, col]
        sum_rewards[col] = sum_rewards[col] + reward
        total_reward = total_reward + reward

    # Output results
    out=dict()
    out['columns']=df.columns.values
    out['total_reward']=total_reward
    out['cols_selected']=cols_selected
    out['sum_rewards']=sum_rewards
    out['num_selections']=num_selections
    out['methodtype']='UCB'

    return(out)

#%% Make figure
def makefig_thompson(out, width=15, height=10):
    columns = out['columns']
    cols_selected = out['cols_selected']
    numbers_of_rewards_1 = out['cols_rewards_1']
    logreward = list(map(lambda x: math.log(x+1), numbers_of_rewards_1))
    # lognum = list(map(lambda x: math.log(x), num_selections))
    
    # Visualising the results - Histogram
    getcounts=np.unique(cols_selected, return_counts=True)
    idx=np.arange(0,len(getcounts[1]))

    # Make figure
    [fig, (ax1,ax2)]=plt.subplots(1, 2, figsize=(width*2,height))
    barwidth = 0.7
    ind = np.arange(len(columns))
    
    ax1.bar(ind, logreward, barwidth)    
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Log(reward)')
    ax1.set_title('Thompson')
    ax1.set_xticks(ind)   
    ax1.set_xticklabels(columns)
    ax1.grid(True)

    ax2.plot(cols_selected,'.')
    ax2.set_title('The selected sample over each round.')
    ax2.set_xlabel('Rounds')
    ax2.set_ylabel('Selected sample')
    ax2.grid(True)
    ax2.axes.set_yticks(idx)
    ax2.axes.set_yticklabels(columns[getcounts[0]])
    
    plt.show()

#%% Figure Thompson
def makefig_UCB(out, width=15, height=10):
    columns=out['columns']
    num_selections=out['num_selections']
    sum_rewards=out['sum_rewards']
    cols_selected = out['cols_selected']

    lognum = list(map(lambda x: math.log(x), num_selections))
    logreward = list(map(lambda x: math.log(x), sum_rewards))

    # Visualising the results - Histogram
    getcounts=np.unique(cols_selected, return_counts=True)
    idx=np.arange(0,len(getcounts[1]))

    ind = np.arange(len(columns))
    barwidth = 0.35

    # fig, ax = plt.subplots(figsize=(width*2,height))
    [fig, (ax1,ax2)]=plt.subplots(1, 2, figsize=(width*2,height))
    rects1 = ax1.bar(ind - barwidth/2, lognum, barwidth)
    rects2 = ax1.bar(ind + barwidth/2, logreward , barwidth)    
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Log(reward)')
    ax1.set_title('UCB')
    ax1.set_xticks(ind)
    ax1.set_xticklabels(columns[ind])
    ax1.legend((rects1[0], rects2[0]), ('Number of times sampled', 'Reward'))
    ax1.grid(True)

    ax2.plot(cols_selected,'.')
    ax2.set_title('The selected sample over each round.')
    ax2.set_xlabel('Rounds')
    ax2.set_ylabel('Selected sample')
    ax2.grid(True)
    ax2.axes.set_yticks(idx)
    ax2.axes.set_yticklabels(columns[getcounts[0]])
    
    plt.show()

#%% Make figure
def makefig_UCB_random(out, width=15, height=10):
    columns=out['columns']
    cols_selected=out['cols_selected']
    
    # Visualising the results - Histogram
#    idx=np.arange(0,df.shape[1])
    colwidth = 0.35       # the width of the bars: can also be len(x) sequence
    getcounts=np.unique(cols_selected, return_counts=True)
    idx=np.arange(0,len(getcounts[1]))

    [fig, (ax1,ax2)]=plt.subplots(1, 2, figsize=(width*2,height))
    # plt.title('UCB randomized results')
    logger.debug(getcounts)
    
    ax1.set_title('Histogram of Sample selections')
    ax1.bar(idx,getcounts[1],colwidth)
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Number of times each sample was selected')
    ax1.set_title('Randomized results')
    ax1.grid(True)
    ax1.axes.set_xticks(idx)
    ax1.axes.set_xticklabels(columns[getcounts[0]])
    
    ax2.plot(cols_selected,'.')
    ax2.set_title('The selected sample over each round.')
    ax2.set_xlabel('Rounds')
    ax2.set_ylabel('Selected sample')
    ax2.grid(True)
    ax2.axes.set_yticks(idx)
    ax2.axes.set_yticklabels(columns[getcounts[0]])
    
    plt.show()


# %% Import example dataset from github.
def import_example(data='ads', url=None, sep=',', verbose='info'):
    """Import example dataset from github source.

    Import one of the few datasets from github source or specify your own download url link.

    Parameters
    ----------
    data : str
        Name of datasets: 'ads'
    url : str
        url link to to dataset.
    verbose : [str, int], default is 'info' or 20
        Set the verbose messages using string or integer values.
        * [0, 60, None, 'silent', 'off', 'no']: No message.
        * [10, 'debug']: Messages from debug level and higher.
        * [20, 'info']: Messages from info level and higher.
        * [30, 'warning']: Messages from warning level and higher.
        * [50, 'critical', 'error']: Messages from critical level and higher.

    Returns
    -------
    pd.DataFrame()
        Dataset containing mixed features.

    """
    set_logger(verbose)
    if url is None:
        if data=='ads':
            url='https://erdogant.github.io/datasets/ads_data.zip'
        else:
            logger.critical('Oops! Example data not found! Try to get it at: www.github.com/erdogant/thompson')
            return None
    else:
        data = wget.filename_from_url(url)

    if url is None:
        logger.warning('Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    filename = os.path.basename(urlparse(url).path)
    PATH_TO_DATA = os.path.join(curpath, filename)
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        logger.info('Downloading [%s] dataset from github source..' %(data))
        wget(url, PATH_TO_DATA)

    # Import local dataset
    logger.info('Import dataset [%s]' %(data))
    # Return
    df=pd.read_csv(PATH_TO_DATA, sep=',')
    return df


# %% Download files from github source
def wget(url, writepath):
    r = requests.get(url, stream=True)
    with open(writepath, "wb") as fd:
        for chunk in r.iter_content(chunk_size=1024):
            fd.write(chunk)

# %%
def convert_verbose_to_new(verbose):
    """Convert old verbosity to the new."""
    # In case the new verbosity is used, convert to the old one.
    if verbose is None: verbose=0
    if not isinstance(verbose, str) and verbose<10:
        status_map = {
            'None': 'silent',
            0: 'silent',
            6: 'silent',
            1: 'critical',
            2: 'warning',
            3: 'info',
            4: 'debug',
            5: 'debug'}
        if verbose>=2: print('[thompson] WARNING use the standardized verbose status. The status [1-6] will be deprecated in future versions.')
        return status_map.get(verbose, 0)
    else:
        return verbose

def get_logger():
    return logger.getEffectiveLevel()


def set_logger(verbose: [str, int] = 'info'):
    """Set the logger for verbosity messages.

    Parameters
    ----------
    verbose : [str, int], default is 'info' or 20
        Set the verbose messages using string or integer values.
        * [0, 60, None, 'silent', 'off', 'no']: No message.
        * [10, 'debug']: Messages from debug level and higher.
        * [20, 'info']: Messages from info level and higher.
        * [30, 'warning']: Messages from warning level and higher.
        * [50, 'critical']: Messages from critical level and higher.

    Returns
    -------
    None.

    > # Set the logger to warning
    > set_logger(verbose='warning')
    > # Test with different messages
    > logger.debug("Hello debug")
    > logger.info("Hello info")
    > logger.warning("Hello warning")
    > logger.critical("Hello critical")

    """
    # Convert verbose to new
    verbose = convert_verbose_to_new(verbose)
    # Set 0 and None as no messages.
    if (verbose==0) or (verbose is None):
        verbose=60
    # Convert str to levels
    if isinstance(verbose, str):
        levels = {'silent': 60,
                  'off': 60,
                  'no': 60,
                  'debug': 10,
                  'info': 20,
                  'warning': 30,
                  'error': 50,
                  'critical': 50}
        verbose = levels[verbose]

    # Show examples
    logger.setLevel(verbose)


def disable_tqdm():
    """Set the logger for verbosity messages."""
    return (True if (logger.getEffectiveLevel()>=30) else False)


def check_logger(verbose: [str, int] = 'info'):
    """Check the logger."""
    set_logger(verbose)
    logger.debug('DEBUG')
    logger.info('INFO')
    logger.warning('WARNING')
    logger.critical('CRITICAL')