import logging
from thompson.thompson import (
	import_example,
    thompson,
	UCB,
	UCB_random,
	plot,
    wget,
    check_logger,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '1.0.0'

# Setup root logger
_logger = logging.getLogger('thompson')
_log_handler = logging.StreamHandler()
_fmt = '[{asctime}] [{name}] [{levelname}] {msg}'
_formatter = logging.Formatter(fmt=_fmt, style='{', datefmt='%d-%m-%Y %H:%M:%S')
_log_handler.setFormatter(_formatter)
_log_handler.setLevel(logging.DEBUG)
_logger.addHandler(_log_handler)
_logger.propagate = False


# module level doc-string
__doc__ = """thompson - Multi-armed bandit algorithms for optimal resource allocation
========================================================================================

The thompson package implements three algorithms for solving the multi-armed bandit problem:

1. Thompson Sampling: A Bayesian approach that maintains probability distributions
   over the expected rewards of each arm and samples from these distributions to
   select the next arm to pull.

2. Upper Confidence Bound (UCB): A deterministic algorithm that selects arms based
   on their estimated rewards and the uncertainty in those estimates.

3. Randomized Sampling: A baseline method that randomly selects arms without
   considering their past performance.

The multi-armed bandit problem is a classic reinforcement learning problem that
exemplifies the exploration-exploitation tradeoff dilemma. In this problem, a
fixed limited set of resources must be allocated between competing choices in a
way that maximizes expected gain, when each choice's properties are only partially
known at the time of allocation.

Installation
-----------
```bash
pip install thompson
```

Basic Usage
----------
>>> import thompson as th
>>> # Load example data
>>> df = th.import_example()
>>> # Apply Thompson sampling
>>> out = th.thompson(df)
>>> # Plot results
>>> th.plot(out)

>>> # Use UCB algorithm
>>> out = th.UCB(df)
>>> th.plot(out)

>>> # Use randomized sampling
>>> out = th.UCB_random(df)
>>> th.plot(out)

Features
--------
- Thompson Sampling implementation
- Upper Confidence Bound (UCB) algorithm
- Randomized sampling baseline
- Visualization tools for results
- Example datasets included
- Comprehensive logging system

Documentation
------------
For more information, see:
- Documentation: https://erdogant.github.io/thompson/
- GitHub: https://github.com/erdogant/thompson
- PyPI: https://pypi.org/project/thompson/

License
-------
MIT License
"""
