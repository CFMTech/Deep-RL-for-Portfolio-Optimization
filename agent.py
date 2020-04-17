import os
from time import sleep
from collections import deque
from collections import namedtuple

import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display

# local imports
from memory import Memory, PrioritizedMemory, Node
from models import Actor, Critic

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "dones")
)

GAMMA = 0.99  # discount factor
TAU_ACTOR = 1e-1  # soft update of the actor target parameters
TAU_CRITIC = 1e-3  # soft update of critic target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY_actor = 0  # L2 weight decay of the actor
WEIGHT_DECAY_critic = 1e-2  # L2 weight decay of the critic
BATCH_SIZE = 64  # minibatch size
BUFFER_SIZE = int(1e6)  # replay buffer size
PRETRAIN = 64  # number of pretraining steps (must be greater than BATCH_SIZE)  #noqa
MAX_STEP = 100  # number of steps in an episode
WEIGHTS = "weights/"  # path to the repository where to save the models' weights
FC1_UNITS_ACTOR = 16  # Number of nodes in first hidden layer
FC2_UNITS_ACTOR = 8  # Number of nodes in second hidden layer
FC1_UNITS_CRITIC = 64  # Number of nodes in first hidden layer of the critic network
FC2_UNITS_CRITIC = 32  # Number of nodes in second hidden layer of the critic network
DECAY_RATE = 0  # Decay rate of the exploration noise
EXPLORE_STOP = 1e-3  # Final exploration probability


def optimal_f(p, pi, lambd=0.5, psi=0.3, cost="trade_l2"):
    """
    Description
    --------------
    Function with the shape of the optimal solution for cost models with 0, l2 and l1
    trading costs.

    Parameters
    --------------
    p     : Float, the next signal value.
    pi    : Float, the current position.
    lambd : Float > 0, Parameter of the cost model.
    psi   : Float > 0, Parameter of our model defining the trading cost.
    cost  : String in ['none', 'trade_l1', 'trade_l2'], cost model.

    Returns
    --------------
    Float, The function evaluation (which is the next trade).
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


def optimal_max_pos(p, pi, thresh, max_pos):
    """
    Description
    --------------
    Function with the shape of the optimal solution for MaxPos cost model with l1 trading
    cost.

    Parameters
    --------------
    p       : Float, the next signal value.
    pi      : Float, the current position.
    thresh  : Float > 0, threshold of the solution in the infinite horizon case.
    max_pos : Float > 0, maximum allowed position.

    Returns
    --------------
    Float, The function evaluation (which is the next trade).
    """

    if abs(p) < thresh:
        return 0
    elif p >= thresh:
        return max_pos - pi
    elif p <= -thresh:
        return -max_pos - pi


# Vectorizing.
optimal_f_vec = np.vectorize(optimal_f, excluded=set(["pi", "lambd", "psi", "cost"]))
optimal_max_pos_vec = np.vectorize(
    optimal_max_pos, excluded=set(["pi", "thresh", "max_pos"])
)


class OUNoise:
    """
    Class of the OU exploration noise.
    """

    def __init__(self, mu=0.0, theta=0.1, sigma=0.1):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu

    def sample(self, truncate=False, max_pos=2, position=0, action=0):
        x = self.state
        if truncate:
            from scipy.stats import truncnorm

            m = -max_pos - position - action - (1 - self.theta) * x
            M = max_pos - position - action - (1 - self.theta) * x
            x_a, x_b = m / self.sigma, M / self.sigma
            X = truncnorm(x_a, x_b, scale=self.sigma)
            dx = self.theta * (self.mu - x) + X.rvs()
            self.state = x + dx
            return self.state

        else:
            dx = self.theta * (self.mu - x) + self.sigma * np.random.randn()
            self.state = x + dx
            return self.state


class Agent:
    def __init__(
        self,
        gamma=GAMMA,
        max_size=BUFFER_SIZE,
        max_step=MAX_STEP,
        memory_type="uniform",
        alpha=0.6,
        beta0=0.4,
        epsilon=1e-8,
        sliding="oldest",
        batch_size=BATCH_SIZE,
        theta=1.0,
        sigma=1.0,
    ):
        """
        Description
        -------------
        Constructor of class Agent

        Parameters & Arguments
        -------------
        gamma       : Float < 1 (typically 0.99), the discount factor.
        max_size    : Int, the maximum size of the memory buffer.
        max_step    : Int, number of steps in an episode.
        memory_type : String in ['uniform', 'prioritized'] type of experience replay to
                      use.
        alpha       : Float in [0, 1], power of prioritization to use (used only with
                      prioritized experience replay).
        beta0       : Float in [0, 1] that gets annealed to 1 during training because of
                      the bias introduced by priority sampling (used only with prioritized
                      experience replay).
        epsilon     : Float > 0 very small, introduced in priority estimation to ensure no
                      transition has 0 priority.
        sliding     : String in ['oldest', 'random'], when the tree gets saturated and a
                      new experience comes up.
                            - 'oldest' : Oldest leaves are the first to be changed.
                            - 'random' : Random leaves are changed.
        batch_size  : Int, the training batch size.
        theta       : Float, Noise parameter.
        sigma       : Float, Noise parameter.
        memory      : Memory object, the memory buffer.
        noise       : OUNoise object, the exploration noise which is an Ornstein-Uhlenbeck
                      process.

        """

        assert 0 <= gamma <= 1, "Discount factor gamma must be in [0, 1]"
        assert memory_type in [
            "uniform",
            "prioritized",
            "per_intervals",
        ], "memory must be in ['uniform', 'prioritized']"
        self.gamma = gamma
        self.max_size = max_size
        self.memory_type = memory_type
        self.epsilon = epsilon

        if memory_type == "uniform":
            self.memory = Memory(max_size=max_size)

        elif memory_type == "prioritized":
            self.memory = PrioritizedMemory(max_size=max_size, sliding=sliding)

        self.max_step = max_step
        self.alpha = alpha
        self.beta0 = beta0
        self.batch_size = batch_size
        self.noise = OUNoise(theta=theta, sigma=sigma)

        # Actor Networks initialized to None
        self.actor_local = None
        self.actor_target = None

        # Critic Networks initialized to None
        self.critic_local = None
        self.critic_target = None

    def reset(self):
        """
        Description
        -------------
        Reset the exploration noise.

        Parameters
        -------------

        Returns
        -------------
        """

        self.noise.reset()

    def step(self, state, action, reward, next_state, done, pretrain=False):
        """
        Description
        -------------
        Save the experience (state, action, reward, next_state, not done) in the replay
        buffer.

        Parameters
        -------------
        state      : 2-tuple of Floats: - state[0]: pi, the current position.
                                        - state[1]: p, the next value of the signal.
        action     : Float, the action taken (which is the next position).
        reward     : Float, the computed reward.
        next_state : 2-tuple of Floats representing the next state.
        done       : Boolean, whether the episode is over or not (I'm not sure if we
                     should only consider time limit as the finishing condition).
        pretrain   : Boolean, whethen we are in a pretraining phase or not.

        Returns
        -------------
        """

        # We use Pytorch tensors for further use in the pipeline.
        state_mb = torch.tensor([state], dtype=torch.float)
        action_mb = torch.tensor([[action]], dtype=torch.float)
        reward_mb = torch.tensor([[reward]], dtype=torch.float)
        next_state_mb = torch.tensor([next_state], dtype=torch.float)
        not_done_mb = torch.tensor([[not done]], dtype=torch.float)

        if self.memory_type == "uniform":
            self.memory.add(
                (state_mb, action_mb, reward_mb, next_state_mb, not_done_mb)
            )

        # During pretraining, the just initialized critic network is likely to output
        # near 0 values, so we will assume the TD error to be equal to the reward.
        elif self.memory_type == "prioritized":
            priority = (
                (abs(reward) + self.epsilon) ** self.alpha
                if pretrain
                else self.memory.highest_priority()
            )
            # Add (transition, leaf) to the buffer.
            self.memory.add(
                (state_mb, action_mb, reward_mb, next_state_mb, not_done_mb), priority
            )

    def act(self, state, noise=True, explore_probability=1, truncate=False, max_pos=2):
        """
        Description
        -------------
        Act in an exploratory fashion by adding the noise.

        Parameters
        -------------
        state               : 2-tuple of Floats:
                              - state[0]: pi, the current position.
                              - state[1]: p, the next value of the signal.
        noise               : Boolean, whether to add exploratory noise or not.
        explore_probability : Float, decaying parameter that controls the noise magnitude.
        truncate            : Boolean, truncate the noise sample such that the position
                              remains between -MaxPos and MaxPos.
        max_pos             : Float > 0, truncate the positions between -MaxPos and
                              MaxPos.

        Returns
        -------------
        Float, the clipped action (trade) to be taken.
        """

        position = state[1]
        state = torch.tensor([state], dtype=torch.float)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).data.numpy()

        self.actor_local.train()
        if noise:
            noise_sample = self.noise.sample(
                truncate=truncate,
                max_pos=max_pos,
                position=position,
                action=float(action),
            )
            action += explore_probability * noise_sample

        return float(action)

    def soft_update(self, local_model, target_model, tau):
        """
        Description
        -------------
        According to https://arxiv.org/abs/1509.02971
        Perform a soft target update of weights theta of the target_network using those
        theta_prime of the local network: theta_prime = tau*theta + (1 - tau)*theta_prime

        Parameters
        -------------
        local_model  : Actor or Critic local network.
        target_model : Actor or Critic target network.
        tau          : 0 < tau < 1

        Returns
        -------------
        Float, the clipped action to be taken (a.k.a the new position).
        """

        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def pretrain(self, env, total_steps=PRETRAIN):
        """
        Description
        -------------
        Pretrain the agent to partially fill the replay buffer.

        Parameters
        -------------
        env : Environment object, it serves as the environment of training for the agent.
        total_steps : Int, number of pretraining steps (must be greater than BATCH_SIZE).

        Returns
        -------------
        """

        env.reset()
        with torch.no_grad():
            for i in range(total_steps):
                state = env.get_state()
                action = self.act(
                    state, truncate=(not env.squared_risk), max_pos=env.max_pos
                )
                reward = env.step(action)
                next_state = env.get_state()
                done = env.done
                self.step(state, action, reward, next_state, done, pretrain=True)
                if done:
                    env.reset()

    def train(
        self,
        env,
        total_episodes=100,
        tau_actor=TAU_ACTOR,
        tau_critic=TAU_CRITIC,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        weight_decay_actor=WEIGHT_DECAY_actor,
        weight_decay_critic=WEIGHT_DECAY_critic,
        total_steps=PRETRAIN,
        weights=WEIGHTS,
        freq=50,
        fc1_units_actor=FC1_UNITS_ACTOR,
        fc2_units_actor=FC2_UNITS_ACTOR,
        fc1_units_critic=FC1_UNITS_CRITIC,
        fc2_units_critic=FC2_UNITS_CRITIC,
        decay_rate=DECAY_RATE,
        explore_stop=EXPLORE_STOP,
        tensordir="runs/",
        learn_freq=50,
        plots=False,
        pi=0.5,
        lambd=None,
        psi=None,
        phi=None,
        thresh=3,
        mile=50,
        progress="tqdm_notebook",
    ):
        """
        Description
        -------------
        Train the agent for a total number of episodes.

        Parameters
        -------------
        env                 : Environment object, it serves as the training environment
                              for the agent.
        total_episodes      : Int, total number of training episodes.
        tau_actor           : 0 < Float < 1, soft update parameter of the actor.
        tau_critic          : 0 < Float < 1, soft update parameter of the critic.
        lr_actor            : Float, learning rate of the actor network.
        lr_critic           : Float, learning rate of the critic network.
        weight_decay_actor  : Float, L2 weight decay of the actor network.
        weight_decay_critic : Float, L2 weight decay of the critic network.
        total_steps         : Int, number of pretraining steps (must be greater than
                              BATCH_SIZE).
        weights             : String, path to the repository where to save the models'
                              weights.
        freq                : Int, number of episodes between two saved models.
        fc1_units_actor     : Int, number of nodes in the first hidden layer of the actor
                              network.
        fc2_units_actor     : Int, number of nodes in the second hidden layer of the actor
                              network.
        fc1_units_critic    : Int, number of nodes in the first hidden layer of the critic
                              network.
        fc2_units_critic    : Int, number of nodes in the second hidden layer of the
                              critic network.
        decay_rate          : Float, the decay rate of exploration noise.
        explore_stop        : Float, the final exploration noise magnitude.
        tensordir           : String, path to write tensorboard scalars.
        learn_freq          : Int, each time (number_steps%learn_freq == 0), we make a
                              training step.
        plots               : Boolean, whether to plot the shape of the function at the
                              end of each episode or not.
        pi                  : Float, only used when plots is True. The plot is done by
                              fixing pi and moving p between -4 and 4.
        lambd               : Float or None, only used when plots is True. The lambda
                              parameter of the function to plot against the models.
                              If None, lambd will be the lambd parameter of the
                              environment env.
        psi                 : Float or None, only used when plots is True. The psi
                              parameter of the function to plot against the models.
                              If None, lambd will be the lambd parameter of the
                              environment env.
        thresh              : Float > 0, threshold of the solution in the infinite horizon
                              case.

        Returns
        -------------
        """

        # Creare folder where to store the Actor weights.
        if not os.path.isdir(weights):
            os.mkdir(weights)

        # Set the summary writer of tensorboard
        writer = SummaryWriter(log_dir=tensordir)

        if plots:
            _ = plt.figure(figsize=(15, 10))
            range_values = np.arange(-4, 4, 0.01)
            signal_zeros = torch.tensor(
                np.vstack((range_values, np.zeros(len(range_values)))).T,
                dtype=torch.float,
            )
            signal_ones_pos = torch.tensor(
                np.vstack((range_values, 0.5 * np.ones(len(range_values)))).T,
                dtype=torch.float,
            )
            signal_ones_neg = torch.tensor(
                np.vstack((range_values, -0.5 * np.ones(len(range_values)))).T,
                dtype=torch.float,
            )
            if psi is None:
                psi = env.psi

            if lambd is None:
                lambd = env.lambd

            if env.squared_risk:
                result1 = optimal_f_vec(
                    signal_ones_neg[:, 0].numpy(),
                    -pi,
                    lambd=lambd,
                    psi=psi,
                    cost=env.cost,
                )
                result2 = optimal_f_vec(
                    signal_zeros[:, 0].numpy(), 0, lambd=lambd, psi=psi, cost=env.cost
                )
                result3 = optimal_f_vec(
                    signal_ones_pos[:, 0].numpy(),
                    pi,
                    lambd=lambd,
                    psi=psi,
                    cost=env.cost,
                )

            else:
                result1 = optimal_max_pos_vec(
                    signal_ones_neg[:, 0].numpy(), -pi, thresh, env.max_pos
                )
                result2 = optimal_max_pos_vec(
                    signal_zeros[:, 0].numpy(), 0, thresh, env.max_pos
                )
                result3 = optimal_max_pos_vec(
                    signal_ones_pos[:, 0].numpy(), pi, thresh, env.max_pos
                )

        # Define Actor local and target networks
        self.actor_local = Actor(
            env.state_size, fc1_units=fc1_units_actor, fc2_units=fc2_units_actor
        )
        self.actor_target = Actor(
            env.state_size, fc1_units=fc1_units_actor, fc2_units=fc2_units_actor
        )

        # Define the optimizer and its learning rate scheduler for the Actor networks
        actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=lr_actor, weight_decay=weight_decay_actor
        )
        actor_lr_scheduler = lr_scheduler.StepLR(
            actor_optimizer, step_size=mile * 100, gamma=0.5
        )

        # Define Actor local and target networks
        self.critic_local = Critic(
            env.state_size, fcs1_units=fc1_units_critic, fc2_units=fc2_units_critic
        )
        self.critic_target = Critic(
            env.state_size, fcs1_units=fc1_units_critic, fc2_units=fc2_units_critic
        )

        # Define the optimizer and its learning rate scheduler for the Critic networks
        critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=lr_critic,
            weight_decay=weight_decay_critic,
        )
        critic_lr_scheduler = lr_scheduler.StepLR(
            critic_optimizer, step_size=mile * 100, gamma=0.5
        )

        # Save the initialized model
        model_file = weights + "ddpg_1" + ".pth"
        torch.save(self.actor_local.state_dict(), model_file)
        # print('\nSaved model to ' + model_file + '\n')

        # Initialize containers to add some useful information about training (useful to
        # visualize with tensorboard)
        mean_rewards = deque(maxlen=10)
        cum_rewards = []
        actor_losses = deque(maxlen=10)
        critic_losses = deque(maxlen=10)

        # Reset counting the nodes of the SumTree when using Prioritized Experience
        # Replay.
        Node.reset_count()
        # Pretraining to partially fill the replay buffer.
        self.pretrain(env, total_steps=total_steps)
        i = 0
        # exploration_probability = 1
        N_train = total_episodes * env.T // learn_freq
        beta = self.beta0
        self.reset()
        n_train = 0

        range_total_episodes = range(total_episodes)
        # setup progress bar
        if progress == "tqdm_notebook":
            from tqdm import tqdm_notebook

            range_total_episodes = tqdm_notebook(list(range_total_episodes))
            progress_bar = range_total_episodes
        elif progress == "tqdm":
            from tqdm import tqdm

            range_total_episodes = tqdm(list(range_total_episodes))
            progress_bar = range_total_episodes
        else:
            progress_bar = None

        for episode in range_total_episodes:
            # start_time = time()
            episode_rewards = []
            env.reset()
            state = env.get_state()
            done = env.done
            train_iter = 0
            # Environment Exploration phase
            while not done:
                explore_probability = explore_stop + (1 - explore_stop) * np.exp(
                    -decay_rate * i
                )
                action = self.act(
                    state,
                    truncate=(not env.squared_risk),
                    max_pos=env.max_pos,
                    explore_probability=explore_probability,
                )
                reward = env.step(action)
                writer.add_scalar("State/signal", state[0], i)
                writer.add_scalar("Signal/position", state[1], i)
                writer.add_scalar("Signal/action", action, i)
                next_state = env.get_state()
                done = env.done
                self.step(state, action, reward, next_state, done)
                state = next_state
                episode_rewards.append(reward)
                i += 1
                train_iter += 1
                if done:
                    self.reset()
                    total_reward = np.sum(episode_rewards)
                    mean_rewards.append(total_reward)
                    if (episode > 0) and (episode % 5 == 0):
                        mean_r = np.mean(mean_rewards)
                        cum_rewards.append(mean_r)
                        writer.add_scalar("Reward & Loss/reward", mean_r, episode)
                        writer.add_scalar(
                            "Reward & Loss/actor_loss", np.mean(actor_losses), episode
                        )
                        writer.add_scalar(
                            "Reward & Loss/critic_loss", np.mean(critic_losses), episode
                        )

                # Learning phase
                if train_iter % learn_freq == 0:
                    n_train += 1
                    if self.memory_type == "uniform":
                        # Sample a batch of experiences :
                        # (state, action, reward, next_state, done)
                        transitions = self.memory.sample(self.batch_size)
                        batch = Transition(*zip(*transitions))
                        states_mb = torch.cat(batch.state)
                        actions_mb = torch.cat(batch.action)
                        rewards_mb = torch.cat(batch.reward)
                        next_states_mb = torch.cat(batch.next_state)
                        dones_mb = torch.cat(batch.dones)

                    elif self.memory_type == "prioritized":
                        # Sample a batch of experiences :
                        # (state, action, reward, next_state, done)
                        transitions, indices = self.memory.sample(self.batch_size)
                        batch = Transition(*zip(*transitions))
                        states_mb = torch.cat(batch.state)
                        actions_mb = torch.cat(batch.action)
                        rewards_mb = torch.cat(batch.reward)
                        next_states_mb = torch.cat(batch.next_state)
                        dones_mb = torch.cat(batch.dones)

                    # Update local Critic network
                    # Use target Actor to compute the next actions to take at the sampled
                    # next states
                    actions_next = self.actor_target(next_states_mb)
                    # Use target Critic to compute the Q values of the sampled
                    # (next_states, actions)
                    Q_targets_next = self.critic_target(next_states_mb, actions_next)
                    Q_targets = rewards_mb + (
                        self.gamma * Q_targets_next * dones_mb
                    )  # Compute target Q values
                    # Compute expected Q values with the local Critic network
                    Q_expected = self.critic_local(states_mb, actions_mb)
                    # Compute the TD errors (needed to update priorities when using
                    # Prioritized replay, and also to compute the loss)
                    td_errors = F.l1_loss(Q_expected, Q_targets, reduction="none")
                    # Update the priorities of experiences in the sampled batch when
                    # Prioritized Experience Replay is used
                    if self.memory_type == "prioritized":
                        # Sum of all priorities.
                        sum_priorities = self.memory.sum_priorities()
                        # Sampling probabilities.
                        probabilities = (
                            self.memory.retrieve_priorities(indices) / sum_priorities
                        ).reshape((-1, 1))
                        # Importance sampling weights.
                        is_weights = torch.tensor(
                            1 / ((self.max_size * probabilities) ** beta),
                            dtype=torch.float,
                        )
                        # Normalize the importance sampling weights.
                        is_weights /= is_weights.max()
                        # Update parameter beta.
                        beta = (1 - self.beta0) * (n_train / N_train) + self.beta0
                        for i_enum, index in enumerate(indices):
                            # Update the priorities of the sampled experiences.
                            self.memory.update(
                                index,
                                (abs(float(td_errors[i_enum].data)) + self.epsilon)
                                ** self.alpha,
                            )

                        # Compute Critic loss function with bias correction.
                        critic_loss = (is_weights * (td_errors ** 2)).mean() / 2

                    elif self.memory_type == "uniform":
                        # Compute Critic loss function.
                        critic_loss = (td_errors ** 2).mean() / 2

                    # Store the current Critic loss value.
                    critic_losses.append(critic_loss.data.item())

                    # Minimize the Critic loss
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    # Clip the gradient to avoid taking huge steps in the gradient update
                    torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 0.1)
                    critic_optimizer.step()
                    critic_lr_scheduler.step()

                    # Update local Actor network
                    # Compute Actor loss which comes from the Off-Policy Deterministic
                    # Policy gradient theorem,
                    # see http://proceedings.mlr.press/v32/silver14.pdf and https://arxiv.org/abs/1509.02971  # noqa
                    actions_pred = self.actor_local(states_mb)
                    actor_loss = -self.critic_local(states_mb, actions_pred).mean()
                    actor_losses.append(actor_loss.data.item())

                    # Minimize the Actor loss
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # Clip the gradient to avoid taking huge steps in the gradient update
                    torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 0.1)
                    actor_optimizer.step()
                    actor_lr_scheduler.step()

                    # Update Critic and Actor target Networks
                    self.soft_update(self.critic_local, self.critic_target, tau_critic)
                    self.soft_update(self.actor_local, self.actor_target, tau_actor)

            # Plot the shape of the function and a function with approximately optimal
            # shape (regarding the cumulative reward) found by a gridsearch over lambd and
            # psi parameters
            if plots:
                plt.clf()
                self.actor_local.eval()
                with torch.no_grad():
                    plt.subplot(2, 3, 1)
                    plt.plot(
                        signal_ones_neg[:, 0].numpy(),
                        self.actor_local(signal_ones_neg)[:, 0].data.numpy(),
                        label="model",
                    )
                    plt.plot(signal_ones_neg[:, 0].numpy(), result1, label="optimal")
                    plt.xlim(-4, 4)
                    plt.ylim(-4, 4)
                    plt.legend()

                    plt.subplot(2, 3, 2)
                    plt.plot(
                        signal_zeros[:, 0].numpy(),
                        self.actor_local(signal_zeros)[:, 0].data.numpy(),
                        label="model",
                    )
                    plt.plot(signal_zeros[:, 0].numpy(), result2, label="optimal")
                    plt.xlim(-4, 4)
                    plt.ylim(-4, 4)
                    plt.legend()

                    plt.subplot(2, 3, 3)
                    plt.plot(
                        signal_ones_pos[:, 0].numpy(),
                        self.actor_local(signal_ones_pos)[:, 0].data.numpy(),
                        label="model",
                    )
                    plt.plot(signal_ones_pos[:, 0].numpy(), result3, label="optimal")
                    plt.xlim(-4, 4)
                    plt.ylim(-4, 4)
                    plt.legend()

                    plt.subplot(2, 3, 4)
                    sns.distplot(states_mb[:, 0])

                display.clear_output(wait=True)
                if progress_bar is not None:
                    display.display(progress_bar)
                display.display(plt.gcf())
                sleep(0.0001)
                self.actor_local.train()

            # Save the Actor network weights after a number of episodes each time
            if (episode % freq) == 0:
                model_file = weights + "ddpg_" + str(episode) + ".pth"
                torch.save(self.actor_local.state_dict(), model_file)
                # print('\nSaved model to ' + model_file + '\n')

        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()
