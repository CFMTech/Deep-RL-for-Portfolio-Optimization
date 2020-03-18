# Deep RL for Portfolio Optimization

This repository accompanies our [arXiv preprint](https://arxiv.org/abs/2003.06497) "Deep
Deterministic Portfolio Optimization" where we explore deep reinforcement learning methods
to solve portfolio optimization problems. More precisely, we consider three tractable cost
models for which the optimal or approximately optimal solutions are well known in the
literature. We adapt the Deep Deterministic Policy Gradient (DDPG) algorithm to each of
these problems to retrieve the corresponding solutions.


## Getting started

### Prerequisites

- Python 3.6 or greater.
- PyTorch 1.0.1
- Seaborn, Scipy, tensorboardX

### Installation

Clone this repository:
```
git clone https://github.com/CFMTech/Deep-RL-for-Portfolio-Optimization.git
cd Deep-RL-for-Portfolio-Optimization
```

Create an environment:
```
conda env create -f ./environment.yml
```

Activate it:
```
conda activate deep_rl_for_portfolio_optimization
```

Install the corresponding IPython kernel:
```
python -m ipykernel install --name deep_rl_for_portfolio_optimization --user
```


## Tutorial

The file "summary.ipynb" presents a clear pipeline of how to define an environment and an
agent, in addition to training and evaluation. The ".py" files have their content
documented too, class methods and functions are described in terms of their objective, the
input and output variables along with their types. So for example if one wants to use
uniform sampling instead of prioritized sampling, the constructor of class ```agent ```
has parameter ```memory_type ``` that can be set to either ```uniform``` or
```prioritized```, and this is well explained in the documentation of the constructor.

You can visualize the evolution of some variables during training like the positions,
actions, signal values, actor and critic losses, and reward with tensorboardX just by
running the command ```tensorboard --logdir runs_directory/``` where "runs_directory/" can
be set in ```tensordir``` parameter of method ```train``` within class ```agent```. Then
open the URL http://localhost:6006
