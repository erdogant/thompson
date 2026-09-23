# thompson.plot

## Overview

Visualizes the result dict from `thompson`, `UCB`, or `UCB_random`. Shows log-scaled rewards (or selection counts) and the arm-selection trajectory over rounds. Displays the figure; returns `None`.

## Signature

```python
thompson.plot(out, width=15, height=10, verbose='info')
```

## Parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `out` | `dict` | required | Output from one of the three algorithms (`methodtype` required). |
| `width` | int | 15 | Figure width in inches. |
| `height` | int | 10 | Figure height in inches. |
| `verbose` | `str` or `int` | `'info'` | Logging level. |

## Returns

`None` (plots are shown with `plt.show()`).

## Usage

```python
import thompson as th

df = th.import_example()
out = th.thompson(df)
th.plot(out)

th.plot(th.UCB(df))
th.plot(th.UCB_random(df))
```

## Notes

- Dispatch is based on `out['methodtype']` (`'thompson'`, `'UCB'`, `'UCB_random'`).
- Arm colors are consistent across plots via a shared palette.
- Requires a graphical backend (or headless Agg if only saving is needed; the library itself always calls `plt.show()`).
