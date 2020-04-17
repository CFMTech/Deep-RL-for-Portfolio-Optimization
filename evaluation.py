import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# local imports
from agent import Agent, optimal_max_pos_vec
from models import Actor


def test_models(
    path_weights, env, fc1_units=16, fc2_units=8, random_state=1024, n_episodes=100
):
    """
    Description
    ---------------
    Evaluate saved models on a number of environments simulated with a fixed random state.
    The evaluation is the average score over the total number of test episodes.

    Parameters
    ---------------
    path_weights    : String, path to the saved training weights.
    env : Object of class Environment, the evaluation environment (It should be the same
          as the training environment).
    fc1_units       : Int, number of nodes in the first layer of the network.
    fc2_units       : Int, number of nodes in the second layer of the network.
    random_state : Int, a fixed random state to evaluate all models on the same episodes.
    n_episodes      : Int, the total number of evaluation episodes.

    Returns
    ---------------
    scores          : Dict, - keys   : training iteration of the saved model.
                            - values : average score over the evaluation episodes.
    scores_episodes : Dict, - keys   : training iteration of the saved model.
                            - values : Dict, - keys   : index of the evaluation episode.
                                             - values : score over the evaluation episode.
    scores_cumsum   : Dict, - keys   : training iteration of the saved model.
                            - values : Dict,
                                - keys   : index of the evaluation episode.
                                - values : cumulative sum of the reward at each time step
                                           per episode.
    pnls            : Dict, - keys   : training iteration of the saved model.
                            - values : Dict, - keys   : index of the evaluation episode.
                                             - values : pnl per episode.
    positions       : Dict, - keys   : training iteration of the saved model.
                            - values : Dict, - keys   : index of the evaluation episode.
                                             - values : positions per episode.
    """

    models_names = os.listdir(path_weights)
    if ".ipynb_checkpoints" in models_names:
        models_names.remove(".ipynb_checkpoints")

    agent = Agent()
    rng = np.random.RandomState(random_state)
    random_states = rng.randint(0, int(1e6), size=n_episodes)
    scores = {}
    scores_episodes = {}
    scores_cumsum = {}
    positions = {}
    pnls = {}
    for model_name in models_names:
        # print(model_name)
        state_dict = torch.load(path_weights + model_name)
        model = Actor(env.state_size, fc1_units=fc1_units, fc2_units=fc2_units)
        model.load_state_dict(state_dict)
        score, score_episode, score_cumsum, pnl, position = env.test(
            agent, model, total_episodes=n_episodes, random_states=random_states
        )
        scores[int(model_name[5:][:-4])] = score
        scores_episodes[int(model_name[5:][:-4])] = score_episode
        scores_cumsum[int(model_name[5:][:-4])] = score_cumsum
        positions[int(model_name[5:][:-4])] = position
        pnls[int(model_name[5:][:-4])] = pnl
        # print('Average score : %.2f' % score)
        # print('\n')

    return scores, scores_episodes, scores_cumsum, pnls, positions


def plot_bars(scores):
    """
    Description
    ---------------
    Bar plot of the evaluation score across the models. Note that the optimal model is indexed by 0

    Parameters
    ---------------
    scores : First return of test_models function (see its doc for more details)

    Returns
    ---------------
    """

    scores = dict(sorted(scores.items()))

    _ = plt.figure(figsize=(30, 10))
    plt.bar(range(len(scores)), list(scores.values()), align="center")
    plt.xticks(range(len(scores)), list(scores.keys()))
    plt.title("Overview of the perfomance of each model", fontsize=20)
    plt.show()


def plot_min_max(env, models_keys, scores_episodes):
    """
    Description
    ---------------
    For each model indexed by models_keys and the optimal model, plot the min score signal
    and max score signals along with their respective scores. Note that the optimal model
    is indexed by 0

    Parameters
    ---------------
    env : Object of class Environment, the evaluation environment (It should be the same
          as the training environment).
    models_keys     : List of 5 elements containing the indices of the models to use.
    scores_episodes : Second return of test_models function (see its doc for more
                      details).

    Returns
    ---------------
    """

    _ = plt.figure(figsize=(20, 10))
    models_keys = [-1] + models_keys
    for i, model_key in enumerate(models_keys):
        score_episode = scores_episodes[model_key]
        random_state_min, random_state_max = (
            min(score_episode, key=score_episode.get),
            max(score_episode, key=score_episode.get),
        )

        env.reset(random_state=random_state_min)
        signal_min = env.signal

        env.reset(random_state=random_state_max)
        signal_max = env.signal

        plt.subplot(2, 3, i + 1)
        plt.plot(
            range(len(signal_min)),
            signal_min,
            label="min %.2f" % score_episode[random_state_min],
        )
        plt.plot(
            range(len(signal_max)),
            signal_max,
            label="max %.2f" % score_episode[random_state_max],
        )
        plt.legend()
        plt.title("model %d" % model_key)

    plt.show()


def plot_hist(model_key, scores_episodes):
    """
    Description
    ---------------
    Plot histogram of scores across the evaluation episodes of model indexed by model_key
    and compare it with the optimal model

    Parameters
    ---------------
    model_key       : Int, the index of the considered model.
    scores_episodes : Second return of test_models function (see its doc for more
                      details).

    Returns
    ---------------
    """

    _ = plt.figure(figsize=(15, 4))
    plt.subplot(121)
    sns.distplot(list(scores_episodes[model_key].values()))
    plt.title("score distribution of model %d" % model_key)

    plt.subplot(122)
    sns.distplot(list(scores_episodes[-1].values()))
    plt.title("score distribution of optimal model")

    plt.show()


def optimal_f(p, pi, lambd=0.5, psi=0.3, cost="trade_l2"):
    """
    Description
    --------------
    Optimal solution for cost models with l2, l1 or no trading costs.

    Parameters
    --------------
    p     : Float, the next signal value.
    pi    : Float, the current position.
    lambd : Float > 0, Parameter of the cost model.
    psi   : Float > 0, Parameter of our model defining the trading cost.
    cost  : String in ['none', 'trade_l1', 'trade_l2'], cost model.

    Returns
    --------------
    Float, The optimal solution evaluation (which is the next position).
    """

    if cost == "trade_0":
        return p / (2 * lambd) - pi

    elif cost == "trade_l2":
        return p / (2 * (lambd + psi)) + psi * pi / (lambd + psi) - pi

    elif cost == "trade_l1":
        if p <= -psi + 2 * lambd * pi:
            return (p + psi) / (2 * lambd) - pi

        elif -psi + 2 * lambd * pi < p < psi + 2 * lambd * pi:
            return 0

        elif p >= psi + 2 * lambd * pi:
            return (p - psi) / (2 * lambd) - pi


# Vectorizing the optimal solution.
optimal_f_vec = np.vectorize(optimal_f, excluded=set(["pi", "lambd", "psi", "cost"]))


def plot_function(
    path_weights,
    env,
    models_keys,
    fc1_units=16,
    fc2_units=8,
    low=-1,
    high=1,
    step=0.01,
    pi=0.5,
    psi=0.3,
    lambd=0.5,
    thresh=2,
    clip=True,
):
    """
    Description
    ---------------
    Plot the functions of each model along with the optimal function given a position and
    a range for the signal.

    Parameters
    ---------------
    path_weights    : String, path to the saved training weights.
    env : Object of class Environment, the evaluation environment (It should be the same
          as the training environment).
    models_keys     : List of 6 elements containing the indices of the models to use.
    fc1_units       : Int, number of nodes in the first layer of the network.
    fc2_units       : Int, number of nodes in the second layer of the network.
    low             : Float, minimum of the signal range.
    high            : Float, maximum of the signal range.
    step            : Float, step size along the signal range.
    pi              : Float, plot using -pi, 0 and pi.
    psi             : Float or None, parameter of the solution for both second and third
                                     model.
                      None  -> Only plot the myopic solution function.
                      Float -> Also plot the solution function with the given parameter
                               psi.
    lambd           : Float, parameter controlling the slope of the solution outside the
                      band.
    thresh          : Float>0, price threshold to make a trade.
    clip            : Boolean, whether to clip positions beyond maxpos or not.

    Returns
    ---------------
    """

    range_values = np.arange(low, high, step)
    signal_zeros = torch.tensor(
        np.vstack((range_values, np.zeros(len(range_values)))).T, dtype=torch.float
    )
    signal_ones_pos = torch.tensor(
        np.vstack((range_values, pi * np.ones(len(range_values)))).T, dtype=torch.float
    )
    signal_ones_neg = torch.tensor(
        np.vstack((range_values, -pi * np.ones(len(range_values)))).T, dtype=torch.float
    )
    if env.squared_risk:
        optimal1 = optimal_f_vec(
            signal_ones_neg[:, 0].numpy(), -pi, lambd=lambd, psi=psi, cost=env.cost
        )
        optimal2 = optimal_f_vec(
            signal_zeros[:, 0].numpy(), 0, lambd=lambd, psi=psi, cost=env.cost
        )
        optimal3 = optimal_f_vec(
            signal_ones_pos[:, 0].numpy(), pi, lambd=lambd, psi=psi, cost=env.cost
        )

    else:
        optimal1 = optimal_max_pos_vec(
            signal_ones_neg[:, 0].numpy(), -pi, thresh, env.max_pos
        )
        optimal2 = optimal_max_pos_vec(
            signal_zeros[:, 0].numpy(), 0, thresh, env.max_pos
        )
        optimal3 = optimal_max_pos_vec(
            signal_ones_pos[:, 0].numpy(), pi, thresh, env.max_pos
        )

    _ = plt.figure(figsize=(20, 30))
    for i, model_key in enumerate(models_keys):
        state_dict = torch.load(path_weights + "ddpg_" + str(model_key) + ".pth")
        model = Actor(env.state_size, fc1_units=fc1_units, fc2_units=fc2_units)
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            if clip:
                model1 = np.clip(
                    model(signal_ones_neg)[:, 0].data.numpy(),
                    -env.max_pos + pi,
                    env.max_pos + pi,
                )
                model2 = np.clip(
                    model(signal_zeros)[:, 0].data.numpy(), -env.max_pos, env.max_pos
                )
                model3 = np.clip(
                    model(signal_ones_pos)[:, 0].data.numpy(),
                    -env.max_pos - pi,
                    env.max_pos - pi,
                )

            else:
                model1 = model(signal_ones_neg)[:, 0].data.numpy()
                model2 = model(signal_zeros)[:, 0].data.numpy()
                model3 = model(signal_ones_pos)[:, 0].data.numpy()

            plt.subplot(len(models_keys), 3, i * 3 + 1)
            plt.plot(signal_ones_neg[:, 0].numpy(), model1, label="model")
            plt.plot(signal_ones_neg[:, 0].numpy(), optimal1, label="optimal")

            plt.xlim(-4, 4)
            plt.ylim(-4, 4)
            plt.title(r"model : %d, $\pi = -%s$" % (model_key, str(pi)))
            plt.legend()

            plt.subplot(len(models_keys), 3, i * 3 + 2)
            plt.plot(signal_zeros[:, 0].numpy(), model2, label="model")
            plt.plot(signal_zeros[:, 0].numpy(), optimal2, label="optimal")

            plt.xlim(-4, 4)
            plt.ylim(-4, 4)
            plt.title(r"model : %d, $\pi = 0$" % model_key)
            plt.legend()

            plt.subplot(len(models_keys), 3, i * 3 + 3)
            plt.plot(signal_ones_pos[:, 0].numpy(), model3, label="model")
            plt.plot(signal_ones_pos[:, 0].numpy(), optimal3, label="optimal")

            plt.xlim(-4, 4)
            plt.ylim(-4, 4)
            plt.title(r"model : %d, $\pi = %s$" % (model_key, str(pi)))
            plt.legend()

    plt.show()
