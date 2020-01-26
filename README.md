# Multi-armed bandit

[![Python](https://img.shields.io/pypi/pyversions/thompson)](https://img.shields.io/pypi/pyversions/thompson)
[![PyPI Version](https://img.shields.io/pypi/v/thompson)](https://pypi.org/project/thompson/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/thompson/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/thompson)](https://pepy.tech/project/thompson)
[![Donate](https://img.shields.io/badge/donate-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)

* Thompson is Python package to evaluate the multi-armed bandit problem. In addition to thompson, Upper Confidence Bound (UCB) algorithm, and randomized results are also implemented.
* In probability theory, the multi-armed bandit problem is a problem in which a fixed limited set of resources must be allocated between competing (alternative) choices in a way that maximizes their expected gain, when each choice's properties are only partially known at the time of allocation, and may become better understood as time passes or by allocating resources to the choice. This is a classic reinforcement learning problem that exemplifies the exploration-exploitation tradeoff dilemma <a href="https://en.wikipedia.org/wiki/Multi-armed_bandit">wikipedia</a>. 
* In the problem, each machine provides a random reward from a probability distribution specific to that machine. The objective of the gambler is to maximize the sum of rewards earned through a sequence of lever pulls. The crucial tradeoff the gambler faces at each trial is between "exploitation" of the machine that has the highest expected payoff and "exploration" to get more information about the expected payoffs of the other machines. The trade-off between exploration and exploitation is also faced in machine learning. In practice, multi-armed bandits have been used to model problems such as managing research projects in a large organization like a science foundation or a pharmaceutical company <a href="https://en.wikipedia.org/wiki/Multi-armed_bandit">wikipedia</a>.


## Contents
- [Installation](#-installation)
- [Requirements](#-Requirements)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

## Installation
* Install thompson from PyPI (recommended). thompson is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* Distributed under the MIT license.

## Requirements
```python
pip install matplotlib numpy pandas
```

## Quick Start
```
pip install thompson
```

* Alternatively, install thompson from the GitHub source:
```bash
git clone https://github.com/erdogant/thompson.git
cd thompson
python setup.py install
```  

## Import thompson package
```python
import thompson as mab
```

## Load example data:
```python
df  = mab.example_data()
```

## Compute multi-armed bandit using thompson
```python
out = mab.thompson(df)
fig = mab.plot(out)
```
<p align="center">
  <img src="https://github.com/erdogant/thompson/blob/master/docs/figs/fig_thompson.png" width="900" />
</p>

## Compute multi-armed bandit using UCB-Upper confidence Bound
```python
out = mab.UCB(df)
fig = mab.plot(out)
```
<p align="center">
  <img src="https://github.com/erdogant/thompson/blob/master/docs/figs/fig_ucb.png" width="900" />
</p>

## Compute multi-armed bandit using randomized data
```python
out = mab.UCB_random(df)
fig = mab.plot(out)
```
<p align="center">
  <img src="https://github.com/erdogant/thompson/blob/master/docs/figs/fig_ucb_random.png" width="900" />
</p>


####  df looks like this:
```
      Ad 1  Ad 2  Ad 3  Ad 4  Ad 5  Ad 6  Ad 7  Ad 8  Ad 9  Ad 10
0        1     0     0     0     1     0     0     0     1      0
1        0     0     0     0     0     0     0     0     1      0
2        0     0     0     0     0     0     0     0     0      0
3        0     1     0     0     0     0     0     1     0      0
4        0     0     0     0     0     0     0     0     0      0
   ...   ...   ...   ...   ...   ...   ...   ...   ...    ...
9995     0     0     1     0     0     0     0     1     0      0
9996     0     0     0     0     0     0     0     0     0      0
9997     0     0     0     0     0     0     0     0     0      0
9998     1     0     0     0     0     0     0     1     0      0
9999     0     1     0     0     0     0     0     0     0      0

[10000 rows x 10 columns]
```

### Citation
Please cite thompson in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2019thompson,
  title={thompson},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/thompson}},
}
```

### References
* https://en.wikipedia.org/wiki/Multi-armed_bandit
   
### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

### Contribute
* All kinds of contributions are welcome!

### Licence
See [LICENSE](LICENSE) for details.

### Donation
* This package is created and maintained in my free time. If this package is usefull, you can show your <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">gratitude</a> :) Thanks!
