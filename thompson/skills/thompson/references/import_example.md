# thompson.import_example

## Overview

Loads the built-in ads multi-armed bandit dataset (or a custom URL). Caches the file under the package `data/` directory.

## Signature

```python
thompson.import_example(data='ads', url=None, sep=',', verbose='info')
```

## Parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `data` | str | `'ads'` | Dataset name. Currently only `'ads'` is supported when `url` is None. |
| `url` | str or None | None | Optional direct download URL. |
| `sep` | str | `','` | CSV separator (passed to `pd.read_csv`). |
| `verbose` | `str` or `int` | `'info'` | Logging level. |

## Returns

`pd.DataFrame` — for the ads dataset: shape `(10000, 10)`, dtype int64, values in `{0, 1}`.

## Usage

```python
import thompson as th

df = th.import_example()
print(df.shape)          # (10000, 10)
print(df.head())

# Custom URL (optional)
# df = th.import_example(url="https://example.com/my_bandit.csv")
```

## Notes

- First call downloads from `https://erdogant.github.io/datasets/ads_data.zip` (or the given URL) and caches the file.
- Requires network access on first download.
- Column names become arm labels in subsequent plots.
