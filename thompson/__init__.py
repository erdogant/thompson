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
__doc__ = """
thompson - The multi-armed bandit by Thompson Sampling, UCB-Upper confidence Bound, and randomized sampling.
=====================================================================

**thompson** 
See README.md file for more information.

"""
