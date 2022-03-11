import numpy as np
import thompson as th

# Load data
df = th.import_example()

# Compute multi-armed bandit using method and make plot
out_tps = th.thompson(df)

# Plot
fig = th.plot(out_tps)

# UCB-Upper confidence Bound,
out_ucb = th.UCB(df)
fig = th.plot(out_ucb)

# Randomized
out_ran = th.UCB_random(df)
fig = th.plot(out_ran)

