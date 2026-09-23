# thompson.UCB

## Overview

Upper Confidence Bound algorithm. On each round selects the arm that maximizes average reward plus an exploration bonus \(\sqrt{(3/2)\log(n+1)/n_i}\). Arms never pulled receive an effectively infinite bound so they are tried first.

## Signature

```python
thompson.UCB(df, verbose='info')
```

## Parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `df` | `pd.DataFrame` | required | Shape `(n_trials, n_arms)`, binary 0/1 rewards. |
| `verbose` | `str` or `int` | `'info'` | Logging level. |

## Returns

`dict` with:

| Key | Type | Description |
| --- | ---- | ----------- |
| `columns` | array-like | Arm names |
| `total_reward` | int/float | Sum of rewards |
| `cols_selected` | list[int] | Arm chosen each round |
| `sum_rewards` | list | Cumulative reward per arm |
| `num_selections` | list | Times each arm was selected |
| `methodtype` | str | Always `'UCB'` |

## Usage

```python
import thompson as th

df = th.import_example()
out = th.UCB(df)
print(out["total_reward"])
print(out["num_selections"])
th.plot(out)
```

## Important details

- Deterministic given fixed data order.
- Exploration term uses \(\sqrt{(3/2)\log(n+1)/n_i}\).
- Complexity \(O(n \times d)\).

## Related

- `thompson.thompson` — Bayesian alternative
- `thompson.UCB_random` — random baseline
- `thompson.plot`
