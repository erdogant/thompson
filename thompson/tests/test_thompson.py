import unittest
import numpy as np
import pandas as pd
import thompson as th

class TestThompson(unittest.TestCase):
    """Test suite for the thompson package.
    
    This test suite verifies the functionality of the multi-armed bandit algorithms
    implemented in the thompson package, including Thompson Sampling, UCB (Upper Confidence Bound),
    and randomized sampling.
    
    Examples
    --------
    >>> import thompson as th
    >>> df = th.import_example()
    >>> out = th.thompson(df)
    >>> print(out['total_reward'])
    """

    def setUp(self):
        """Set up test fixtures.
        
        This method is called before each test method. It loads the example dataset
        that will be used across all test cases.
        
        Examples
        --------
        >>> self.df = th.import_example()
        >>> print(self.df.shape)
        (10000, 10)
        """
        self.df = th.import_example()
        
    def test_import_example(self):
        """Test that example data is loaded correctly.
        
        This test verifies that:
        - The loaded data is a pandas DataFrame
        - The data has the correct shape (10000 samples, 10 features)
        - All columns are of type int64
        - All values are either 0 or 1
        
        Examples
        --------
        >>> df = th.import_example()
        >>> assert isinstance(df, pd.DataFrame)
        >>> assert df.shape == (10000, 10)
        >>> assert all(df.dtypes == 'int64')
        >>> assert all(df.isin([0, 1]).all())
        """
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertEqual(self.df.shape, (10000, 10))
        self.assertTrue(all(self.df.dtypes == 'int64'))
        self.assertTrue(all(self.df.isin([0, 1]).all()))
        
    def test_thompson(self):
        """Test Thompson sampling algorithm.
        
        This test verifies that:
        - The output is a dictionary with all required keys
        - The method type is 'thompson'
        - The output arrays have the correct lengths
        - All values are of the correct type
        
        Examples
        --------
        >>> out = th.thompson(df)
        >>> print(out.keys())
        dict_keys(['columns', 'total_reward', 'cols_selected', 'cols_rewards_1', 
                  'cols_rewards_0', 'methodtype'])
        >>> print(out['methodtype'])
        'thompson'
        """
        out = th.thompson(self.df)
        
        # Check output structure
        self.assertIsInstance(out, dict)
        self.assertIn('columns', out)
        self.assertIn('total_reward', out)
        self.assertIn('cols_selected', out)
        self.assertIn('cols_rewards_1', out)
        self.assertIn('cols_rewards_0', out)
        self.assertIn('methodtype', out)
        
        # Check output values
        self.assertEqual(out['methodtype'], 'thompson')
        self.assertEqual(len(out['cols_selected']), len(self.df))
        self.assertEqual(len(out['cols_rewards_1']), len(self.df.columns))
        self.assertEqual(len(out['cols_rewards_0']), len(self.df.columns))
        self.assertTrue(all(isinstance(x, int) for x in out['cols_selected']))
        self.assertTrue(all(isinstance(x, int) for x in out['cols_rewards_1']))
        self.assertTrue(all(isinstance(x, int) for x in out['cols_rewards_0']))
        
    def test_ucb(self):
        """Test UCB (Upper Confidence Bound) algorithm.
        
        This test verifies that:
        - The output is a dictionary with all required keys
        - The method type is 'UCB'
        - The output arrays have the correct lengths
        - All values are of the correct type (including numpy integer types)
        
        Examples
        --------
        >>> out = th.UCB(df)
        >>> print(out.keys())
        dict_keys(['columns', 'total_reward', 'cols_selected', 'sum_rewards', 
                  'num_selections', 'methodtype'])
        >>> print(out['methodtype'])
        'UCB'
        """
        out = th.UCB(self.df)
        
        # Check output structure
        self.assertIsInstance(out, dict)
        self.assertIn('columns', out)
        self.assertIn('total_reward', out)
        self.assertIn('cols_selected', out)
        self.assertIn('sum_rewards', out)
        self.assertIn('num_selections', out)
        self.assertIn('methodtype', out)
        
        # Check output values
        self.assertEqual(out['methodtype'], 'UCB')
        self.assertEqual(len(out['cols_selected']), len(self.df))
        self.assertEqual(len(out['sum_rewards']), len(self.df.columns))
        self.assertEqual(len(out['num_selections']), len(self.df.columns))
        self.assertTrue(all(isinstance(x, int) for x in out['cols_selected']))
        self.assertTrue(all(isinstance(x, (int, np.integer)) for x in out['sum_rewards']))
        self.assertTrue(all(isinstance(x, (int, np.integer)) for x in out['num_selections']))
        
    def test_ucb_random(self):
        """Test randomized sampling algorithm.
        
        This test verifies that:
        - The output is a dictionary with all required keys
        - The method type is 'UCB_random'
        - The output arrays have the correct lengths
        - All values are of the correct type
        
        Examples
        --------
        >>> out = th.UCB_random(df)
        >>> print(out.keys())
        dict_keys(['columns', 'total_reward', 'cols_selected', 'methodtype'])
        >>> print(out['methodtype'])
        'UCB_random'
        """
        out = th.UCB_random(self.df)
        
        # Check output structure
        self.assertIsInstance(out, dict)
        self.assertIn('columns', out)
        self.assertIn('total_reward', out)
        self.assertIn('cols_selected', out)
        self.assertIn('methodtype', out)
        
        # Check output values
        self.assertEqual(out['methodtype'], 'UCB_random')
        self.assertEqual(len(out['cols_selected']), len(self.df))
        self.assertTrue(all(isinstance(x, int) for x in out['cols_selected']))
        
    def test_plot(self):
        """Test plotting functionality.
        
        This test verifies that:
        - The plot function works with all three methods (Thompson, UCB, randomized)
        - The plot function returns None (as it displays the plot directly)
        
        Examples
        --------
        >>> out_tps = th.thompson(df)
        >>> fig_tps = th.plot(out_tps)  # Displays plot
        >>> assert fig_tps is None
        """
        # Test with Thompson output
        out_tps = th.thompson(self.df)
        fig_tps = th.plot(out_tps)
        self.assertIsNone(fig_tps)  # plot() returns None
        
        # Test with UCB output
        out_ucb = th.UCB(self.df)
        fig_ucb = th.plot(out_ucb)
        self.assertIsNone(fig_ucb)
        
        # Test with randomized output
        out_ran = th.UCB_random(self.df)
        fig_ran = th.plot(out_ran)
        self.assertIsNone(fig_ran)

if __name__ == '__main__':
    unittest.main()
