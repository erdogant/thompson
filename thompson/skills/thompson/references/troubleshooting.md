# Troubleshooting

Use this guide for common issues with the **thompson** package.

## Installation

```bash
python -m pip install -U thompson
python -c "import thompson as th; print(th.__version__)"
```

If the import fails, confirm the active environment and that `numpy`, `pandas`, and `matplotlib` are installed.

## ImportError / ModuleNotFoundError

1. Verify the package is installed in the same interpreter you are using.
2. Restart the session after installing.
3. Check version: `python -c "import thompson; print(thompson.__version__)"`

## Data issues

### Wrong shape or non-binary values

Algorithms expect a DataFrame of shape `(n_trials, n_arms)` with 0/1 entries.

```python
print(df.shape)
print(df.dtypes)
print(df.isin([0, 1]).all().all())
```

Non-binary values will still be summed as rewards, but Thompson Sampling’s Beta model and UCB’s Bernoulli assumptions no longer hold.

### Empty or single-column DataFrame

With zero rows or zero columns the loops produce empty lists or trivial results. Ensure at least one trial and one arm.

## Algorithm behavior

### Different results on each run (Thompson Sampling)

Expected: `thompson()` uses `random.betavariate` and is stochastic. UCB is deterministic for a fixed data order.

### Low total reward vs random baseline

On some datasets early random exploration can temporarily look similar; with enough trials Thompson and UCB typically accumulate higher reward. Compare `out['total_reward']` across methods on the same `df`.

## Plotting

### Plot returns None / no figure appears

`plot()` calls `plt.show()` and returns `None`. In headless environments set a non-interactive backend (e.g. `matplotlib.use('Agg')`) before importing, or save the current figure yourself after the internal plot functions run. The public API does not return a figure object.

### Wrong plot type

`plot` dispatches on `out['methodtype']`. Passing a dict without that key, or with an unknown value, will not draw a figure.

## Logging

Suppress messages:

```python
th.thompson(df, verbose='silent')
# or
th.set_logger('silent')
```

## Reproducibility

- Thompson Sampling: set `random.seed(...)` before calling if you need a fixed sequence (the library does not expose a `random_state` argument).
- UCB: fully deterministic given the same DataFrame order.
- Record library version and Python version when reporting issues.

## Debugging checklist

1. Reproduce with `df = th.import_example()`.
2. Print `out.keys()` and `out['methodtype']`.
3. Check `len(out['cols_selected']) == len(df)`.
4. Compare `total_reward` across the three methods.
