---
name: thompson
description: Use this skill when working with thompson, a Python library for multi-armed bandit algorithms (Thompson Sampling, UCB, randomized baseline). Trigger on thompson, Thompson sampling, UCB bandit, multi-armed bandit, ad optimization, exploration-exploitation, or plotting bandit results.
---

# thompson

Python library for multi-armed bandit algorithms. Solves the exploration–exploitation tradeoff by allocating trials across competing arms (e.g. ads, treatments) to maximize total reward when each arm’s success rate is only partially known.

## When to use this skill

- Running Thompson Sampling, UCB, or a random baseline on binary reward data
- Comparing bandit algorithms on the same dataset
- Loading the built-in ads example or plotting results
- Debugging input shape, reward encoding, or plot output

## Core capabilities

| Function | Purpose |
| -------- | ------- |
| `thompson(df)` | Bayesian Thompson Sampling (Beta posteriors) |
| `UCB(df)` | Upper Confidence Bound (deterministic) |
| `UCB_random(df)` | Uniform random baseline |
| `plot(out)` | Visualize selections and rewards for any of the three outputs |
| `import_example()` | Load the built-in ads dataset (10k × 10 binary) |

All algorithms expect a pandas DataFrame of shape `(n_trials, n_arms)` with binary values (0/1). Each row is one round; each column is one arm.

## Installation

```bash
pip install thompson
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `requests`.

Import:

```python
import thompson as th
```

## Usage

```python
import thompson as th

df = th.import_example()          # or your own 0/1 DataFrame
out = th.thompson(df)             # or th.UCB(df) / th.UCB_random(df)
print(out["total_reward"])
th.plot(out)
```

## API summary

### `thompson(df, verbose='info')`

Samples from Beta(successes+1, failures+1) for each arm and pulls the arm with the highest sample. Returns a dict with keys:

- `columns`, `total_reward`, `cols_selected`, `cols_rewards_1`, `cols_rewards_0`, `methodtype='thompson'`

### `UCB(df, verbose='info')`

Selects the arm with highest average reward + exploration bonus \(\sqrt{(3/2)\log(n)/n_i}\). Returns:

- `columns`, `total_reward`, `cols_selected`, `sum_rewards`, `num_selections`, `methodtype='UCB'`

### `UCB_random(df, verbose='info')`

Picks a random arm each round. Returns:

- `columns`, `total_reward`, `cols_selected`, `methodtype='UCB_random'`

### `plot(out, width=15, height=10, verbose='info')`

Dispatches to the correct figure for `out['methodtype']`. Displays plots; returns `None`.

### `import_example(data='ads', url=None, sep=',', verbose='info')`

Downloads and caches the ads dataset (10000 rows × 10 binary columns) under the package `data/` folder.

## Workflows

### 1. Compare algorithms on the example dataset

1. Load data: `df = th.import_example()`
2. Run each method: `out_tps = th.thompson(df)`, `out_ucb = th.UCB(df)`, `out_ran = th.UCB_random(df)`
3. Compare `total_reward` and inspect `cols_selected`
4. Plot each: `th.plot(out_tps)`, etc.

### 2. Use your own data

1. Build a DataFrame of shape `(n_rounds, n_arms)` with values in `{0, 1}`
2. Column names become arm labels in plots
3. Call `th.thompson(df)` (or UCB / UCB_random)
4. Read `out['total_reward']` and per-arm reward counts / selection counts

### 3. Visualize results

Pass any algorithm output dict to `th.plot(out)`. Plots show log-scaled rewards (or selection counts) and the arm-selection trajectory over rounds.

## Examples

Executable examples live under `examples/`:

- `examples/thompson_sampling.py` — full Thompson Sampling run + plot
- `examples/ucb.py` — UCB run + plot
- `examples/random_baseline.py` — randomized baseline + plot

## Best practices

- Input must be binary (0/1). Non-binary values are treated as rewards but the Beta model and UCB formulas assume Bernoulli rewards.
- Prefer Thompson Sampling or UCB over the random baseline for higher total reward.
- Thompson Sampling is stochastic (uses `random.betavariate`); re-runs give different selections. UCB is deterministic given the same data order.
- Large `n_trials` is fine; complexity is \(O(n \times d)\) per algorithm.
- Set `verbose` to `'silent'` or `0` to suppress log messages.

## Troubleshooting

See `references/troubleshooting.md` for installation, data shape, import, and plotting issues.

## References

Detailed method docs:

- Thompson Sampling: `references/thompson.md`
- UCB: `references/ucb.md`
- Randomized baseline: `references/ucb_random.md`
- Plotting: `references/plot.md`
- Data loading: `references/import_example.md`
