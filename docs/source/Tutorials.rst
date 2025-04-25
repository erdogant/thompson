Tutorials
#########

This page provides tutorials for using the thompson package to solve multi-armed bandit problems.

Basic Usage
============

The input for ``thompson`` is a ``pd.DataFrame`` with rows as samples and columns as features (arms). 
For demonstration purposes, we will load a dataset with **ads**, containing 10000 samples and 10 ads.

.. code:: python

	# Import library
	import thompson as th

	# Load example data
	df = th.import_example()

	# Show data
	print(df)

	#	      Ad 1  Ad 2  Ad 3  Ad 4  Ad 5  Ad 6  Ad 7  Ad 8  Ad 9  Ad 10
	# 	0        1     0     0     0     1     0     0     0     1      0
	# 	1        0     0     0     0     0     0     0     0     1      0
	# 	2        0     0     0     0     0     0     0     0     0      0
	# 	3        0     1     0     0     0     0     0     1     0      0
	# 	4        0     0     0     0     0     0     0     0     0      0
	# 	   ...   ...   ...   ...   ...   ...   ...   ...   ...    ...
	# 	9995     0     0     1     0     0     0     0     1     0      0
	# 	9996     0     0     0     0     0     0     0     0     0      0
	# 	9997     0     0     0     0     0     0     0     0     0      0
	# 	9998     1     0     0     0     0     0     0     1     0      0
	# 	9999     0     1     0     0     0     0     0     0     0      0

Thompson Sampling
==================

Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. Here's how to use it:

.. code:: python

	# Apply Thompson sampling
	results = th.thompson(df)

	# Plot the results
	th.plot(results)

The output of ``thompson`` :func:`thompson.thompson` is a dictionary with the following keys:

* ``columns``: Names of the columns (arms)
* ``total_reward``: Total rewards obtained
* ``cols_selected``: Vector describing which arm was selected for each trial
* ``cols_rewards_0``: Number of unsuccessful trials per arm
* ``cols_rewards_1``: Number of successful trials per arm
* ``methodtype``: Method that was used ('thompson')

Upper Confidence Bound (UCB)
==============================

The UCB algorithm balances exploration and exploitation using confidence bounds:

.. code:: python

	# Apply UCB algorithm
	results = th.UCB(df)

	# Plot the results
	th.plot(results)

The output of ``UCB`` :func:`thompson.UCB` includes:

* ``columns``: Names of the columns (arms)
* ``total_reward``: Total rewards obtained
* ``cols_selected``: Vector describing which arm was selected for each trial
* ``sum_rewards``: Sum of rewards obtained per arm
* ``num_selections``: Number of times each arm was selected
* ``methodtype``: Method that was used ('UCB')

Randomized Sampling
====================

For comparison, you can use randomized sampling as a baseline:

.. code:: python

	# Apply randomized sampling
	results = th.UCB_random(df)

	# Plot the results
	th.plot(results)

The output of ``UCB_random`` :func:`thompson.UCB_random` includes:

* ``columns``: Names of the columns (arms)
* ``total_reward``: Total rewards obtained
* ``cols_selected``: Vector describing which arm was selected for each trial
* ``methodtype``: Method that was used ('UCB_random')

Visualization
===============

All three methods support visualization of results:

.. code:: python

	# Plot results with custom figure size
	th.plot(results, width=15, height=10)

The plot function creates visualizations showing:
- The performance of each arm
- The selection pattern over time
- Comparison between different methods

.. include:: add_bottom.add