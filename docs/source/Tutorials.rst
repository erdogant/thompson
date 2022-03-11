Input
###########################

The input for ``thompson`` is a ``pd.DataFrame`` with rows as samples and columns as features. 
For demonstration purposes we will load a dataset with **ads**, with 10000 samples and 10 ads.

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


	# Compute which add is best using multi-armed bandits.
	results = th.thompson(df)


Output
###########################

The output of ``thompson`` :func:`thompson.thompson` is a dictionary with the following keys:

	* columns	 : Name of the columns
	* total_reward	 : Total rewards
	* cols_selected	 : Vector that decribes for each sample the linked feature
	* cols_rewards_0 : The rewards per feature 
	* cols_rewards_1 : The rewards per feature
	* methodtype	 : Method that is used


.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>
