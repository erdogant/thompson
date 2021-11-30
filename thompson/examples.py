import numpy as np
import thompson as mab

# Load data
df  = mab.import_example()

# Compute multi-armed bandit using method and make plot

# Thompson
out_tps = mab.thompson(df)
fig = mab.plot(out_tps)

# UCB-Upper confidence Bound,
out_ucb = mab.UCB(df)
fig = mab.plot(out_ucb)

# Randomized
out_ran = mab.UCB_random(df)
fig = mab.plot(out_ran)