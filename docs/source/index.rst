thompson's documentation!
==============================

|python| |pypi| |docs| |stars| |LOC| |downloads_month| |downloads_total| |license| |forks| |open issues| |project status| |colab| |DOI| |repo-size| |donate|

``thompson`` is a Python package that implements three algorithms for solving the multi-armed bandit problem:

1. **Thompson Sampling**: A Bayesian approach that maintains probability distributions over the expected rewards of each arm and samples from these distributions to select the next arm to pull.

2. **Upper Confidence Bound (UCB)**: A deterministic algorithm that selects arms based on their estimated rewards and the uncertainty in those estimates.

3. **Randomized Sampling**: A baseline method that randomly selects arms without considering their past performance.

The multi-armed bandit problem is a classic reinforcement learning problem that exemplifies the exploration-exploitation tradeoff dilemma. In this problem, a fixed limited set of resources must be allocated between competing choices in a way that maximizes expected gain, when each choice's properties are only partially known at the time of allocation.

-----------------------------------

.. note::
	**Your ❤️ is important to keep maintaining this package.** You can `support <https://erdogant.github.io/thompson/pages/html/Documentation.html>`_ in various ways, have a look at the `sponsor page <https://erdogant.github.io/thompson/pages/html/Documentation.html>`_.
	Report bugs, issues and feature extensions at `github <https://github.com/erdogant/thompson/>`_ page.

	.. code-block:: console

	   pip install thompson

-----------------------------------

Features
========

- Thompson Sampling implementation with Bayesian updates
- Upper Confidence Bound (UCB) algorithm with confidence intervals
- Randomized sampling baseline for comparison
- Visualization tools for results analysis
- Example datasets included
- Comprehensive logging system
- Detailed documentation and examples

Content
=======

.. toctree::
   :maxdepth: 1
   :caption: Background
   
   Background

.. toctree::
   :maxdepth: 1
   :caption: Installation
   
   Installation

.. toctree::
  :maxdepth: 1
  :caption: Tutorials

  Tutorials

.. toctree::
  :maxdepth: 1
  :caption: Examples

  Examples

.. toctree::
  :maxdepth: 1
  :caption: Documentation
  
  Documentation
  Coding quality
  thompson.thompson

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |python| image:: https://img.shields.io/pypi/pyversions/thompson.svg
    :alt: |Python
    :target: https://erdogant.github.io/thompson/

.. |pypi| image:: https://img.shields.io/pypi/v/thompson.svg
    :alt: |Python Version
    :target: https://pypi.org/project/thompson/

.. |docs| image:: https://img.shields.io/badge/Sphinx-Docs-blue.svg
    :alt: Sphinx documentation
    :target: https://erdogant.github.io/thompson/

.. |stars| image:: https://img.shields.io/github/stars/erdogant/thompson
    :alt: Stars
    :target: https://img.shields.io/github/stars/erdogant/thompson

.. |LOC| image:: https://sloc.xyz/github/erdogant/thompson/?category=code
    :alt: lines of code
    :target: https://github.com/erdogant/thompson

.. |downloads_month| image:: https://static.pepy.tech/personalized-badge/thompson?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month
    :alt: Downloads per month
    :target: https://pepy.tech/project/thompson

.. |downloads_total| image:: https://static.pepy.tech/personalized-badge/thompson?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
    :alt: Downloads in total
    :target: https://pepy.tech/project/thompson

.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :alt: License
    :target: https://github.com/erdogant/thompson/blob/master/LICENSE

.. |forks| image:: https://img.shields.io/github/forks/erdogant/thompson.svg
    :alt: Github Forks
    :target: https://github.com/erdogant/thompson/network

.. |open issues| image:: https://img.shields.io/github/issues/erdogant/thompson.svg
    :alt: Open Issues
    :target: https://github.com/erdogant/thompson/issues

.. |project status| image:: http://www.repostatus.org/badges/latest/active.svg
    :alt: Project Status
    :target: http://www.repostatus.org/#active

.. |medium| image:: https://img.shields.io/badge/Medium-Blog-green.svg
    :alt: Medium Blog
    :target: https://erdogant.github.io/thompson/pages/html/Documentation.html#medium-blog

.. |donate| image:: https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors
    :alt: donate
    :target: https://erdogant.github.io/thompson/pages/html/Documentation.html#

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Colab example
    :target: https://erdogant.github.io/thompson/pages/html/Documentation.html#colab-notebook

.. |DOI| image:: https://zenodo.org/badge/231458137.svg
    :alt: Cite
    :target: https://zenodo.org/badge/latestdoi/231458137

.. |repo-size| image:: https://img.shields.io/github/repo-size/erdogant/thompson
    :alt: repo-size
    :target: https://img.shields.io/github/repo-size/erdogant/thompson

.. include:: add_bottom.add
