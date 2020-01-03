from thompson.thompson import (
	example_data,
    thompson,
	UCB,
	UCB_random,
	plot,
)

__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
#__version__ = '0.1.0'

# Automatic version control
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


# module level doc-string
__doc__ = """
thompson - The multi-armed bandit by Thompson Sampling, UCB-Upper confidence Bound, and randomized sampling.
=====================================================================

**thompson** 
See README.md file for more information.

"""
