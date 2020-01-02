from kaplanmeier.kaplanmeier import (
	thompson,
    randomUCB,
	UCB,
	makefig_UCB,
	makefig_thompson,
	makefig,
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
thompson - thompson is an Python package for reinforcement learning by the Thompson Sampling/ Random and the real UCB-Upper confidence Bound
=====================================================================

**thompson** 
See README.md file for more information.

"""
