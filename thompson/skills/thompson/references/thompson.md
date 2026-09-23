# thompson.thompson

## Overview

Bayesian multi-armed bandit via Thompson Sampling. Maintains a Beta posterior per arm and, on each round, samples one value from each posterior, then selects the arm with the highest sample.

## Signature

```python
thompson.thompson(df, verbose='info')
```

## Parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `df` | `pd.DataFrame` | required | Shape `(n_trials, n_arms)`. Values should be 0 or 1 (Bernoulli reward). |
| `verbose` | `str` or `int` | `'info'` | Logging level. Use `'silent'` / `0` to suppress messages. |

## Returns

`dict` with:

| Key | Type | Description |
| --- | ---- | ----------- |
| `columns` | array-like | Arm names (DataFrame columns) |
| `total_reward` | int/float | Sum of observed rewards |
| `cols_selected` | list[int] | Arm index chosen on each trial |
| `cols_rewards_1` | list[int] | Success count per arm |
| `cols_rewards_0` | list[int] | Failure count per arm |
| `methodtype` | str | Always `'thompson'` |

## Usage

```python
import thompson as th

df = th.import_example()
out = th.thompson(df)
print(out["total_reward"])
print(out["columns"][out["cols_rewards_1"].index(max(out["cols_rewards_1"]))])
th.plot(out)
```

## Important details

- Prior is Beta(1, 1) (uniform). After \(s\) successes and \(f\) failures the posterior is Beta(\(s+1\), \(f+1\)).
- Selection is stochastic; different runs can choose different sequences.
- Complexity \(O(n \times d)\).

## Common pitfalls

- Passing non-binary data still runs but the Beta model is no longer correct for the reward distribution.
- Empty DataFrame or zero columns will raise or produce empty results.

## Related

- `thompson.UCB` — deterministic alternative
- `thompson.UCB_random` — random baseline
- `thompson.plot` — visualize `out`
