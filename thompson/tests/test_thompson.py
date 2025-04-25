import unittest
import numpy as np
import pandas as pd
import thompson as th

class TestThompson(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.df = th.import_example()
        
    def test_import_example(self):
        """Test that example data is loaded correctly"""
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertEqual(self.df.shape, (10000, 10))
        self.assertTrue(all(self.df.dtypes == 'int64'))
        self.assertTrue(all(self.df.isin([0, 1]).all()))
        
    def test_thompson(self):
        """Test Thompson sampling algorithm"""
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
        """Test UCB algorithm"""
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
        """Test randomized sampling"""
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
        """Test plotting functionality"""
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
