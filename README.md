# Multi-armed bandit

[![Python](https://img.shields.io/pypi/pyversions/thompson)](https://img.shields.io/pypi/pyversions/thompson)
[![PyPI Version](https://img.shields.io/pypi/v/thompson)](https://pypi.org/project/thompson/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/thompson/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/thompson/month)](https://pepy.tech/project/thompson/month)
[![Downloads](https://pepy.tech/badge/thompson)](https://pepy.tech/project/thompson)
[![DOI](https://zenodo.org/badge/231458137.svg)](https://zenodo.org/badge/latestdoi/231458137)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/thompson/)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

* ```Thompson``` is Python package to evaluate the multi-armed bandit problem. In addition to thompson, Upper Confidence Bound (UCB) algorithm, and randomized results are also implemented.
* In probability theory, the multi-armed bandit problem is a problem in which a fixed limited set of resources must be allocated between competing (alternative) choices in a way that maximizes their expected gain, when each choice's properties are only partially known at the time of allocation, and may become better understood as time passes or by allocating resources to the choice. This is a classic reinforcement learning problem that exemplifies the exploration-exploitation tradeoff dilemma <a href="https://en.wikipedia.org/wiki/Multi-armed_bandit">wikipedia</a>. 
* In the problem, each machine provides a random reward from a probability distribution specific to that machine. The objective of the gambler is to maximize the sum of rewards earned through a sequence of lever pulls. The crucial tradeoff the gambler faces at each trial is between "exploitation" of the machine that has the highest expected payoff and "exploration" to get more information about the expected payoffs of the other machines. The trade-off between exploration and exploitation is also faced in machine learning. In practice, multi-armed bandits have been used to model problems such as managing research projects in a large organization like a science foundation or a pharmaceutical company <a href="https://en.wikipedia.org/wiki/Multi-armed_bandit">wikipedia</a>.

# 
**⭐️ Star this repo if you like it ⭐️**
#

#### Install thompson from PyPI

```bash
pip install thompson
```

#### Import thompson package

```python
import thompson as th
```
# 


### [Documentation pages](https://erdogant.github.io/thompson/)

On the [documentation pages](https://erdogant.github.io/thompson/) you can find detailed information about the working of the ``thompson`` with examples. 

<hr> 

### Examples

# 

* [Example: Compute multi-armed bandit using Thompson](https://erdogant.github.io/thompson/pages/html/Examples.html#)

<p align="left">
  <a href="https://erdogant.github.io/thompson/pages/html/Examples.html#">
  <img src="https://github.com/erdogant/thompson/blob/master/docs/figs/fig_thompson.png" width="900" />
  </a>
</p>


# 


* [Example: Compute multi-armed bandit using UCB-Upper confidence Bound](https://erdogant.github.io/thompson/pages/html/Examples.html#ucb-upper-confidence-bound)

<p align="left">
  <a href="https://erdogant.github.io/thompson/pages/html/Examples.html#ucb-upper-confidence-bound">
  <img src="https://github.com/erdogant/thompson/blob/master/docs/figs/fig_ucb.png" width="900" />
  </a>
</p>


# 


* [Example: Compute multi-armed bandit using randomized data](https://erdogant.github.io/thompson/pages/html/Examples.html#randomized-data)

<p align="left">
  <a href="https://erdogant.github.io/thompson/pages/html/Examples.html#randomized-data">
  <img src="https://github.com/erdogant/thompson/blob/master/docs/figs/fig_ucb_random.png" width="900" />
  </a>
</p>


<hr>

### References
* https://en.wikipedia.org/wiki/Multi-armed_bandit
   
### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

### Contribute
* All kinds of contributions are welcome!
* If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated :)

