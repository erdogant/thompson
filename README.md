# Multi-armed bandit

[![Python](https://img.shields.io/pypi/pyversions/thompson)](https://img.shields.io/pypi/pyversions/thompson)
[![PyPI Version](https://img.shields.io/pypi/v/thompson)](https://pypi.org/project/thompson/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/thompson/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/thompson/month)](https://pepy.tech/project/thompson/month)
[![Downloads](https://pepy.tech/badge/thompson)](https://pepy.tech/project/thompson)
[![DOI](https://img.shields.io/badge/DOI-zenodo-black)](https://zenodo.org/badge/latestdoi/231458137)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/thompson/)

* ```Thompson``` is Python package to evaluate the multi-armed bandit problem. In addition to thompson, Upper Confidence Bound (UCB) algorithm, and randomized results are also implemented.
The thompson package implements three algorithms for solving the multi-armed bandit problem:

1. Thompson Sampling: A Bayesian approach that maintains probability distributions
   over the expected rewards of each arm and samples from these distributions to
   select the next arm to pull.

2. Upper Confidence Bound (UCB): A deterministic algorithm that selects arms based
   on their estimated rewards and the uncertainty in those estimates.

3. Randomized Sampling: A baseline method that randomly selects arms without
   considering their past performance.

The multi-armed bandit problem is a classic reinforcement learning problem that
exemplifies the exploration-exploitation tradeoff dilemma. In this problem, a
fixed limited set of resources must be allocated between competing choices in a
way that maximizes expected gain, when each choice's properties are only partially
known at the time of allocation.
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


##### Skill Installation

Developing with Agentic Skills with this library is possible.
The skills are bundled inside the `thompson` package. It is automatically available when you install Thompson from PyPI.
First install the library as depicted above. Then you can install the skill locally or globally for the harness you want:

```bash
thompson install skill --auto           # detect + install locally
thompson install skill --auto --global  # detect + install globally (~/)
thompson install skill --global         # install claude globally (default harness)
thompson install skill --harness opencode --global   # install opencode globally
thompson install skill --harness claude              # unchanged original behaviour
```

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

* [Example: Compute multi-armed bandit using UCB-Upper confidence Bound](https://erdogant.github.io/thompson/pages/html/Examples.html#ucb-upper-confidence-bound)
<p align="left">
  <a href="https://erdogant.github.io/thompson/pages/html/Examples.html#ucb-upper-confidence-bound">
    <img src="https://github.com/erdogant/thompson/blob/master/docs/figs/fig_ucb.png" width="900" />
  </a>
</p>

* [Example: Compute multi-armed bandit using randomized data](https://erdogant.github.io/thompson/pages/html/Examples.html#randomized-data)
<p align="left">
  <a href="https://erdogant.github.io/thompson/pages/html/Examples.html#randomized-data">
    <img src="https://github.com/erdogant/thompson/blob/master/docs/figs/fig_ucb_random.png" width="900" />
  </a>
</p>


### Maintainer
* Erdogan Taskesen, GitHub: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* Yes! This library is entirely **free**, but it runs on coffee! :) Feel free to support with a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a>.

[![Buy me a coffee](https://img.buymeacoffee.com/button-api/?text=Buy+me+a+coffee&emoji=&slug=erdogant&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://www.buymeacoffee.com/erdogant)
