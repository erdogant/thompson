# thompson.UCB_random

## Overview

Uniform random baseline. Each round picks an arm uniformly at random, ignoring past rewards. Useful for comparing against Thompson Sampling and UCB.

## Signature

```python
thompson.UCB_random(df, verbose='info')
```

## Parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `df` | `pd.DataFrame` | required | Shape `(n_trials, n_arms)`, binary rewards. |
| `verbose` | `str` or `int` | `'info'` | Logging level. |

## Returns

`dict` with:

| Key | Type | Description |
| --- | ---- | ----------- |
| `columns` | array-like | Arm names |
| `total_reward` | int/float | Sum of rewards |
| `cols_selected` | list[int] | Arm chosen each round |
| `cols_rewards_1` | None | Not tracked |
| `cols_rewards_0` | None | Not tracked |
| `methodtype` | str | Always `'UCB_random'` |

## Usage

```python
import thompson as th

df = th.import_example()
out = th.UCB_random(df)
print(out["total_reward"])
th.plot(out)
```

## Related

- `thompson.thompson`, `thompson.UCB`, `thompson.plot`
