"""This module implements multi-armed bandit algorithms including Thompson Sampling, 
UCB (Upper Confidence Bound), and randomized sampling.

"""
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

#%% Plot style helper
def _apply_plot_style():
    """Apply a clean, modern style for all bandit plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
    })


# Shared qualitative palette so every arm keeps the same color across all plots
_ARM_PALETTE = [
    '#4C72B0',  # blue
    '#DD8452',  # orange
    '#55A868',  # green
    '#C44E52',  # red
    '#8172B3',  # purple
    '#937860',  # brown
    '#DA8BC3',  # pink
    '#8C8C8C',  # gray
    '#CCB974',  # yellow
    '#64B5CD',  # cyan
]


def _arm_colors(n_arms, indices=None):
    """Return colors for arms using a fixed shared palette.

    Parameters
    ----------
    n_arms : int
        Total number of arms.
    indices : array-like or None
        Specific arm indices to color. If None, colors for 0..n_arms-1 are returned.

    Returns
    -------
    list of RGBA or hex colors
    """
    palette = _ARM_PALETTE
    if indices is None:
        return [palette[i % len(palette)] for i in range(n_arms)]
    return [palette[int(i) % len(palette)] for i in indices]

#%% Plot
def plot(out, width=15, height=10, verbose='info'):
    """Plot the results of the multi-armed bandit algorithm.

    Creates visualizations showing the performance of the selected algorithm.
    The type of plot depends on the method used (Thompson, UCB, or randomized).

    Parameters
    ----------
    out : dict
        Output from thompson, UCB, or UCB_random containing the results to plot.
    width : int, optional
        Width of the figure in inches. Default is 15.
    height : int, optional
        Height of the figure in inches. Default is 10.
    verbose : str or int, optional
        Set the verbose messages using string or integer values:
        * [0, 60, None, 'silent', 'off', 'no']: No message
        * [10, 'debug']: Messages from debug level and higher
        * [20, 'info']: Messages from info level and higher
        * [30, 'warning']: Messages from warning level and higher
        * [50, 'critical', 'error']: Messages from critical level and higher
        Default is 'info'

    Returns
    -------
    None
        The function displays the plot directly and returns None.

    Examples
    --------
    >>> import thompson as th
    >>> df = th.import_example()
    >>> # Plot Thompson sampling results
    >>> out_tps = th.thompson(df)
    >>> th.plot(out_tps)
    >>> # Plot UCB results
    >>> out_ucb = th.UCB(df)
    >>> th.plot(out_ucb)
    >>> # Plot randomized results
    >>> out_ran = th.UCB_random(df)
    >>> th.plot(out_ran)
    """
    set_logger(verbose)
    logger.info('Making plot')
    if out['methodtype']=='thompson':
        makefig_thompson(out, width=width, height=height)
    elif out['methodtype']=='UCB':
        makefig_UCB(out, width=width, height=height)
    elif out['methodtype']=='UCB_random':
        makefig_UCB_random(out, width=width, height=height)

#%% Thompson sampling method
def thompson(df, verbose='info'):
    """Perform Thompson sampling on the multi-armed bandit problem.

    Thompson sampling is a Bayesian approach to the multi-armed bandit problem.
    It maintains a probability distribution over the expected rewards of each arm
    and samples from these distributions to select the next arm to pull.

    Parameters
    ----------
    df : pd.DataFrame
        Contains samples[rows] x features[columns]. Each row represents a trial,
        and each column represents an arm of the bandit. Values should be 0 or 1,
        where 1 indicates a successful trial.
    verbose : str or int, optional
        Set the verbose messages using string or integer values:
        * [0, 60, None, 'silent', 'off', 'no']: No message
        * [10, 'debug']: Messages from debug level and higher
        * [20, 'info']: Messages from info level and higher
        * [30, 'warning']: Messages from warning level and higher
        * [50, 'critical', 'error']: Messages from critical level and higher
        Default is 'info'

    Returns
    -------
    dict
        Dictionary containing:
        - columns: Names of the columns (arms)
        - total_reward: Total rewards obtained
        - cols_selected: Vector describing which arm was selected for each trial
        - cols_rewards_1: Number of successful trials per arm
        - cols_rewards_0: Number of unsuccessful trials per arm
        - methodtype: 'thompson'

    Examples
    --------
    >>> import thompson as th
    >>> df = th.import_example()
    >>> out = th.thompson(df)
    >>> print(f"Total reward: {out['total_reward']}")
    >>> print(f"Best performing arm: {out['columns'][np.argmax(out['cols_rewards_1'])]}")
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
    """Perform randomized sampling on the multi-armed bandit problem.

    This method randomly selects arms without considering their past performance.
    It serves as a baseline for comparing the performance of more sophisticated
    algorithms like Thompson sampling and UCB.

    Parameters
    ----------
    df : pd.DataFrame
        Contains samples[rows] x features[columns]. Each row represents a trial,
        and each column represents an arm of the bandit. Values should be 0 or 1,
        where 1 indicates a successful trial.
    verbose : str or int, optional
        Set the verbose messages using string or integer values:
        * [0, 60, None, 'silent', 'off', 'no']: No message
        * [10, 'debug']: Messages from debug level and higher
        * [20, 'info']: Messages from info level and higher
        * [30, 'warning']: Messages from warning level and higher
        * [50, 'critical', 'error']: Messages from critical level and higher
        Default is 'info'

    Returns
    -------
    dict
        Dictionary containing:
        - columns: Names of the columns (arms)
        - total_reward: Total rewards obtained
        - cols_selected: Vector describing which arm was selected for each trial
        - methodtype: 'UCB_random'

    Examples
    --------
    >>> import thompson as th
    >>> df = th.import_example()
    >>> out = th.UCB_random(df)
    >>> print(f"Total reward: {out['total_reward']}")
    >>> print(f"Number of trials: {len(out['cols_selected'])}")
    """
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
    """Perform Upper Confidence Bound (UCB) algorithm on the multi-armed bandit problem.

    UCB is a deterministic algorithm that selects arms based on their estimated rewards
    and the uncertainty in those estimates. It balances exploration and exploitation
    by selecting arms with high upper confidence bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Contains samples[rows] x features[columns]. Each row represents a trial,
        and each column represents an arm of the bandit. Values should be 0 or 1,
        where 1 indicates a successful trial.
    verbose : str or int, optional
        Set the verbose messages using string or integer values:
        * [0, 60, None, 'silent', 'off', 'no']: No message
        * [10, 'debug']: Messages from debug level and higher
        * [20, 'info']: Messages from info level and higher
        * [30, 'warning']: Messages from warning level and higher
        * [50, 'critical', 'error']: Messages from critical level and higher
        Default is 'info'

    Returns
    -------
    dict
        Dictionary containing:
        - columns: Names of the columns (arms)
        - total_reward: Total rewards obtained
        - cols_selected: Vector describing which arm was selected for each trial
        - sum_rewards: Sum of rewards obtained per arm
        - num_selections: Number of times each arm was selected
        - methodtype: 'UCB'

    Examples
    --------
    >>> import thompson as th
    >>> df = th.import_example()
    >>> out = th.UCB(df)
    >>> print(f"Total reward: {out['total_reward']}")
    >>> print(f"Most selected arm: {out['columns'][np.argmax(out['num_selections'])]}")
    """
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

#%% Make figure – Thompson
def makefig_thompson(out, width=15, height=10, order='vertical'):
    """Create attractive visualizations for Thompson sampling results."""
    _apply_plot_style()

    columns = out['columns']
    cols_selected = np.asarray(out['cols_selected'])
    numbers_of_rewards_1 = np.asarray(out['cols_rewards_1'])
    logreward = np.log1p(numbers_of_rewards_1)

    getcounts = np.unique(cols_selected, return_counts=True)
    idx = np.arange(len(getcounts[1]))
    ind = np.arange(len(columns))

    # Consistent arm colors (same palette used by all plot functions)
    bar_colors = _arm_colors(len(columns))
    point_colors = _arm_colors(len(columns), cols_selected)

    if order == 'horizontal':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width * 1.6, height))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height * 1.3))

    # --- Bar plot of log-rewards ---
    bars = ax1.bar(ind, logreward, width=0.7, color=bar_colors, edgecolor='white', linewidth=0.8, alpha=0.9, zorder=3)
    ax1.set_xlabel('Arms (Features)', fontweight='medium')
    ax1.set_ylabel('Log(1 + Reward)', fontweight='medium')
    ax1.set_xticks(ind)
    ax1.set_xticklabels(columns, rotation=45 if len(columns) > 8 else 0, ha='right' if len(columns) > 8 else 'center')
    ax1.set_ylim(bottom=0)
    # Subtle value labels on tall bars
    for bar, val in zip(bars, logreward):
        if val > 0.05 * logreward.max():
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * logreward.max(), f'{val:.1f}', ha='center', va='bottom', fontsize=8, color='#333333')

    # --- Selection trajectory ---
    ax2.scatter(np.arange(len(cols_selected)), cols_selected, c=point_colors, s=30, alpha=0.55, edgecolors='none', zorder=3)
    ax2.set_xlabel('Round', fontweight='medium')
    ax2.set_ylabel('Selected Arm', fontweight='medium')
    ax2.set_yticks(idx)
    ax2.set_yticklabels(columns[getcounts[0]])
    ax2.set_xlim(-0.5, len(cols_selected) - 0.5)

    fig.suptitle(f'Thompson Sampling  |  Total Reward = {out["total_reward"]}', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    plt.show()

#%% Figure – UCB
def makefig_UCB(out, width=15, height=10, order='vertical'):
    """Create attractive visualizations for Upper Confidence Bound results."""
    _apply_plot_style()

    columns = out['columns']
    num_selections = np.asarray(out['num_selections'], dtype=float)
    sum_rewards = np.asarray(out['sum_rewards'], dtype=float)
    cols_selected = np.asarray(out['cols_selected'])

    # Avoid log(0)
    lognum = np.log1p(num_selections)
    logreward = np.log1p(sum_rewards)

    getcounts = np.unique(cols_selected, return_counts=True)
    idx = np.arange(len(getcounts[1]))
    ind = np.arange(len(columns))
    barwidth = 0.38

    # Consistent arm colors (same palette used by all plot functions)
    bar_colors = _arm_colors(len(columns))
    point_colors = _arm_colors(len(columns), cols_selected)

    if order == 'horizontal':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width * 1.6, height))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height * 1.3))

    # --- Grouped bars (same arm color for both metrics, different hatch/alpha) ---
    rects1 = ax1.bar(ind - barwidth / 2, lognum, barwidth,
                     color=bar_colors, edgecolor='white', linewidth=0.7,
                     label='Times sampled (log)', alpha=0.85, zorder=3)
    rects2 = ax1.bar(ind + barwidth / 2, logreward, barwidth,
                     color=bar_colors, edgecolor='white', linewidth=0.7,
                     label='Reward (log)', alpha=0.55, hatch='///', zorder=3)

    ax1.set_xlabel('Arms (Features)', fontweight='medium')
    ax1.set_ylabel('Log(1 + Value)', fontweight='medium')
    ax1.set_xticks(ind)
    ax1.set_xticklabels(columns, rotation=45 if len(columns) > 8 else 0,
                        ha='right' if len(columns) > 8 else 'center')
    ax1.legend(frameon=True, fancybox=True, shadow=False, loc='upper right')
    ax1.set_ylim(bottom=0)

    # --- Selection trajectory ---
    ax2.scatter(np.arange(len(cols_selected)), cols_selected, c=point_colors, s=30, alpha=0.55, edgecolors='none', zorder=3)
    ax2.set_xlabel('Round', fontweight='medium')
    ax2.set_ylabel('Selected Arm', fontweight='medium')
    ax2.set_yticks(idx)
    ax2.set_yticklabels(columns[getcounts[0]])
    ax2.set_xlim(-0.5, len(cols_selected) - 0.5)

    fig.suptitle(f'Upper Confidence Bound (UCB)  |  Total Reward = {out["total_reward"]}',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    plt.show()

#%% Make figure – Randomized
def makefig_UCB_random(out, width=15, height=10, order='vertical'):
    """Create attractive visualizations for randomized baseline results."""
    _apply_plot_style()

    columns = out['columns']
    cols_selected = np.asarray(out['cols_selected'])

    getcounts = np.unique(cols_selected, return_counts=True)
    idx = np.arange(len(getcounts[1]))
    counts = getcounts[1]
    arm_indices = getcounts[0]  # the actual arm indices that appear

    # Consistent arm colors (same palette used by all plot functions)
    bar_colors = _arm_colors(len(columns), arm_indices)
    point_colors = _arm_colors(len(columns), cols_selected)

    if order == 'horizontal':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width * 1.6, height))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height * 1.3))

    # --- Histogram of selections ---
    bars = ax1.bar(idx, counts, width=0.65, color=bar_colors, edgecolor='white',
                   linewidth=0.8, alpha=0.9, zorder=3)
    ax1.set_xlabel('Arms (Features)', fontweight='medium')
    ax1.set_ylabel('Number of times selected', fontweight='medium')
    ax1.set_xticks(idx)
    ax1.set_xticklabels(columns[getcounts[0]], rotation=45 if len(idx) > 8 else 0,
                        ha='right' if len(idx) > 8 else 'center')
    ax1.set_ylim(bottom=0)
    for bar, val in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * counts.max(),
                 str(val), ha='center', va='bottom', fontsize=8, color='#333333')

    # --- Selection trajectory ---
    ax2.scatter(np.arange(len(cols_selected)), cols_selected, c=point_colors, s=30, alpha=0.55, edgecolors='none', zorder=3)
    ax2.set_xlabel('Round', fontweight='medium')
    ax2.set_ylabel('Selected Arm', fontweight='medium')
    ax2.set_yticks(idx)
    ax2.set_yticklabels(columns[getcounts[0]])
    ax2.set_xlim(-0.5, len(cols_selected) - 0.5)

    fig.suptitle(f'Randomized Baseline  |  Total Reward = {out["total_reward"]}',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
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
