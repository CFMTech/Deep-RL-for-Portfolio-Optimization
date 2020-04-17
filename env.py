import numpy as np

# local imports
from utils import build_ou_process


class Environment:
    """
    The environment consists of the following:
        - state  : (current_position, next_signal)
        - action : next_position.
        - reward : it depends on our model.
    The signal is built as an OU process.
    """

    def __init__(
        self,
        sigma=0.5,
        theta=1.0,
        T=1000,
        random_state=None,
        lambd=0.5,
        psi=0.5,
        cost="trade_0",
        max_pos=10,
        squared_risk=True,
        penalty="none",
        alpha=10,
        beta=10,
        clip=True,
        noise=False,
        noise_std=10,
        noise_seed=None,
        scale_reward=10,
    ):
        """
        Description
        ---------------
        Constructor of class Environment.

        Parameters & Attributes
        ---------------
        sigma          : Float, parameter of price predictor signal
                         p_t - p_{t-1} = -theta*p_{t-1} + sigma*epsilon_t; (epsilon_t)_t
                         are standard normal random variables
        theta          : Float, parameter of price predictor signal
                         p_t - p_{t-1} = -theta*p_{t-1} + sigma*epsilon_t; (epsilon_t)_t
                         are standard normal random variables
        T              : Float, time horizon.
        random_state   : Int or None:
                         - if None, do not use a random state (useful to simulate
                           different paths each time running the simulation).
                         - if Int, use a random state (useful to compare different
                           experimental results).
        lambd          : Float, penalty term of the position in the reward function.
        psi            : Float, penalty term of the trade magnitude in the reward
                         function.
        cost           : String in ['trade_0', 'trade_l1', 'trade_l2']
                          - 'trade_0'  : no trading cost.
                          - 'trade_l1' : squared trading cost.
                          - 'trade_l2' : linear trading cost.
        max_pos        : Float > 0, maximum allowed position.
        squared_risk   : Boolean, whether to use the squared risk term or not.
        penalty        : String in ['none', 'constant', 'tanh', 'exp'], the type of
                         penalty to add to penalize positions beyond maxpos.
                         (It is advised to use a tanh penalty in the maxpos setting).
        alpha          : Int, a parameter of the penalty function.
        beta           : Int, a parameter of the penalty function.
        clip           : Boolean, whether to clip positions beyond maxpos or not.
        noise          : Boolean, whether to consider noisy returns or returns equal to
                         predictor values.
        noise_std      : Float, standard deviation of the noise added to the returns.
        noise_seed     : Int, see used to produce the additive noise of the returns.
        scale_reward   : Float>0, parameter that scales the rewards.
        signal         : 1D np.array of shape (T,) containing the sampled OU process.
        it             : Int, the current time iteration.
        pi             : Float, the current position.
        p              : Float, the next value of the signal.
        state          : 2-tuple, the current state: (p, pi).
        done           : Boolean, whether the episode is over or not. Initialized to
                         False.
        state_size     : Int, state size.
        action_size    : Int, action size.

        Returns
        ---------------
        """

        self.sigma = sigma
        self.theta = theta
        self.T = T
        self.lambd = lambd
        self.psi = psi
        self.cost = cost
        self.max_pos = max_pos
        self.squared_risk = squared_risk
        self.random_state = random_state
        self.signal = build_ou_process(T, sigma, theta, random_state)
        self.it = 0  # First iteration is 0
        self.pi = 0
        self.p = self.signal[self.it + 1]
        self.state = (self.p, self.pi)
        self.done = False
        self.state_size = len(self.state)
        self.action_size = 1
        self.penalty = penalty
        self.alpha = alpha
        self.beta = beta
        self.clip = clip
        self.scale_reward = scale_reward
        self.noise = noise
        self.noise_std = noise_std
        self.noise_seed = noise_seed
        if noise:
            if noise_seed is None:
                self.noise_array = np.random.normal(0, noise_std, T)

            else:
                rng = np.random.RandomState(noise_seed)
                self.noise_array = rng.normal(0, noise_std, T)

    def reset(self, random_state=None, noise_seed=None):
        """
        Description
        ---------------
        Reset the environment to run a new episode.

        Parameters
        ---------------
        random_state : Int or None:
            - if None, do not use a random state (useful to simulate different paths each
              time running the simulation).
            - if Int, use a random state (useful to compare different experimental
              results).
        noise_seed   : Same as random_state but for the noisy returns instead of the
                       predictor signal.

        Returns
        ---------------
        """

        self.signal = build_ou_process(self.T, self.sigma, self.theta, random_state)
        self.it = 0  # First iteration is 0
        self.pi = 0
        self.p = self.signal[self.it + 1]
        self.state = (self.p, self.pi)
        self.done = False
        if self.noise:
            if noise_seed is None:
                self.noise_array = np.random.normal(0, self.noise_std, self.T)

            else:
                rng = np.random.RandomState(noise_seed)
                self.noise_array = rng.normal(0, self.noise_std, self.T)

    def get_state(self):
        """
        Description
        ---------------
        Get the current state of the environment.

        Parameters
        ---------------

        Returns
        ---------------
        2-tuple representing the current state.
        """

        return self.state

    def step(self, action):
        """
        Description
        ---------------
        Aplly action to the environment to modify the state of the agent and get the
        corresponding reward.

        Parameters
        ---------------
        action : Float, the action to perform (next trade to make).

        Returns
        ---------------
        Float, the reward we get by applying action to the current state.
        """

        assert not self.done, (
            "The episode is over, you cannot take another step! "
            "Please reset the environment."
        )
        pi_next_unclipped = self.pi + action
        if self.clip:
            # Clip the next position between -max_pos and max_pos
            pi_next = np.clip(self.pi + action, -self.max_pos, self.max_pos)

        else:
            pi_next = self.pi + action

        if self.penalty == "none":
            pen = 0

        if self.penalty == "constant":
            pen = self.alpha * max(
                0,
                (self.max_pos - pi_next) / abs(self.max_pos - pi_next),
                (-self.max_pos - pi_next) / abs(-self.max_pos - pi_next),
            )

        elif self.penalty == "tanh":
            pen = self.beta * (
                np.tanh(self.alpha * (abs(pi_next_unclipped) - 5 * self.max_pos / 4))
                + 1
            )

        elif self.penalty == "exp":
            pen = self.beta * np.exp(self.alpha * (abs(pi_next) - self.max_pos))

        if self.cost == "trade_0":
            reward = (
                self.p * pi_next - self.lambd * pi_next ** 2 * self.squared_risk - pen
            ) / self.scale_reward

        elif self.cost == "trade_l1":
            if self.noise:
                reward = (
                    (self.p + self.noise_array[self.it]) * pi_next
                    - self.lambd * pi_next ** 2 * self.squared_risk
                    - self.psi * abs(pi_next - self.pi)
                    - pen
                ) / self.scale_reward

            else:
                reward = (
                    self.p * pi_next
                    - self.lambd * pi_next ** 2 * self.squared_risk
                    - self.psi * abs(pi_next - self.pi)
                    - pen
                ) / self.scale_reward

        elif self.cost == "trade_l2":
            if self.noise:
                reward = (
                    (self.p + self.noise_array[self.it]) * pi_next
                    - self.lambd * pi_next ** 2 * self.squared_risk
                    - self.psi * (pi_next - self.pi) ** 2
                    - pen
                ) / self.scale_reward

            else:
                reward = (
                    self.p * pi_next
                    - self.lambd * pi_next ** 2 * self.squared_risk
                    - self.psi * (pi_next - self.pi) ** 2
                    - pen
                ) / self.scale_reward

        self.pi = pi_next
        self.it += 1
        self.p = self.signal[self.it + 1]
        self.state = (self.p, self.pi)
        self.done = self.it == (len(self.signal) - 2)
        return reward

    def test(
        self, agent, model, total_episodes=10, random_states=None, noise_seeds=None
    ):
        """
        Description
        ---------------
        Test a model on a number of simulated episodes and get the average cumulative
        reward.

        Parameters
        ---------------
        agent          : Agent object, the agent that loads the model.
        model          : Actor object, the actor network.
        total_episodes : Int, number of episodes to test.
        random_states  : None or List of length total_episodes:
            - if None, do not use random state when generating episodes
              (useful to get an idea about the performance of a single model).
            - if List, generate episodes with the values in random_states (useful when
              comparing different models).

        noise_seeds    : None or List of length total_episodes:
                         - if None, do not use a random state when generating the additive
                           noise of the returns
                         - if List, generate noise with seeds in noise_seeds.

        Returns
        ---------------
        2-tuple : - Float, average cumulative reward over the generated episodes.
                  - Dict, cumulative reward per episode (random state).
        """

        scores = {}
        scores_cumsum = {}
        pnls = {}
        positions = {}
        agent.actor_local = model
        if random_states is not None:
            assert total_episodes == len(
                random_states
            ), "random_states should be a list of length total_episodes!"

        cumulative_rewards = []
        cumulative_pnls = []
        for episode in range(total_episodes):
            episode_rewards = []
            episode_pnls = []
            episode_positions = [0]
            random_state = None if random_states is None else random_states[episode]
            noise_seed = None if noise_seeds is None else noise_seeds[episode]
            self.reset(random_state, noise_seed)
            state = self.get_state()
            done = self.done
            while not done:
                action = agent.act(state, noise=False)
                pi_next = np.clip(self.pi + action, -self.max_pos, self.max_pos)
                episode_positions.append(pi_next)
                reward = self.step(action)
                pnl = reward + (self.lambd * self.pi ** 2) * self.squared_risk
                state = self.get_state()
                done = self.done
                episode_rewards.append(reward)
                episode_pnls.append(pnl)
                if done:
                    total_reward = np.sum(episode_rewards)
                    total_pnl = np.sum(episode_pnls)
                    if random_states is not None:
                        scores[random_states[episode]] = total_reward
                        scores_cumsum[random_states[episode]] = np.cumsum(
                            episode_rewards
                        )
                        pnls[random_states[episode]] = total_pnl
                        positions[random_states[episode]] = episode_positions

                    cumulative_rewards.append(total_reward)
                    cumulative_pnls.append(total_pnl)
                    # print('Episode: {}'.format(episode),
                    #      'Total reward: {:.2f}'.format(total_reward))

        return (
            np.mean(cumulative_rewards),
            scores,
            scores_cumsum,
            np.mean(cumulative_pnls),
            positions,
        )

    def apply(self, state, thresh=1, lambd=None, psi=None):
        """
        Description
        ---------------
        Apply solution with a certain band and slope outside the band, otherwise apply the
        myopic solution.

        Parameters
        ---------------
        state      : 2-tuple, the current state.
        thresh     : Float>0, price threshold to make a trade.
        lambd      : Float, slope of the solution in the non-banded region.
        psi        : Float, band width of the solution.

        Returns
        ---------------
        Float, the trade to make in state according to this function.
        """

        p, pi = state
        if lambd is None:
            lambd = self.lambd

        if psi is None:
            psi = self.psi

        if not self.squared_risk:
            if abs(p) < thresh:
                return 0
            elif p >= thresh:
                return self.max_pos - pi
            elif p <= -thresh:
                return -self.max_pos - pi

        else:
            if self.cost == "trade_0":
                return p / (2 * lambd) - pi

            elif self.cost == "trade_l2":
                return (p + 2 * psi * pi) / (2 * (lambd + psi)) - pi

            elif self.cost == "trade_l1":
                if p < -psi + 2 * lambd * pi:
                    return (p + psi) / (2 * lambd) - pi
                elif -psi + 2 * lambd * pi <= p <= psi + 2 * lambd * pi:
                    return 0
                elif p > psi + 2 * lambd * pi:
                    return (p - psi) / (2 * lambd) - pi

    def test_apply(
        self,
        total_episodes=10,
        random_states=None,
        thresh=1,
        lambd=None,
        psi=None,
        noise_seeds=None,
        max_point=6.0,
        n_points=1000,
    ):
        """
        Description
        ---------------
        Test a function with certain slope and band width for each reward model (with and
        without trading cost, and depending on the penalty when trading cost is used).
        When psi and lambd are not provided, use the myopic solution.

        Parameters
        ---------------
        total_episodes : Int, number of episodes to test.
        random_states  : None or List of length total_episodes:
                         - if None, do not use random state when generating episodes
                           (useful to get an idea about the performance of a single
                           model).
                         - if List, generate episodes with the values in random_states
                           (useful when comparing different models).
        lambd          : Float, slope of the solution in the non-banded region.
        psi            : Float, band width of the solution.
        max_point      : Float, the maximum point in the grid [0, max_point]
        n_points       : Int, the number of points in the grid.

        Returns
        ---------------
        5-tuple : - Float, average cumulative reward over the generated episodes.
                  - Dict, cumulative reward per episode (random state).
                  - Dict, cumulative sum of the reward at each time step per episode.
                  - Dict, pnl per episode.
                  - Dict, positions per episode.
        """

        scores = {}
        scores_cumsum = {}
        pnls = {}
        positions = {}
        if random_states is not None:
            assert total_episodes == len(
                random_states
            ), "random_states should be a list of length total_episodes!"

        cumulative_rewards = []
        cumulative_pnls = []
        for episode in range(total_episodes):
            episode_rewards = []
            episode_pnls = []
            episode_positions = [0]
            random_state = None if random_states is None else random_states[episode]
            noise_seed = None if noise_seeds is None else noise_seeds[episode]
            self.reset(random_state, noise_seed)
            state = self.get_state()
            done = self.done
            while not done:
                action = self.apply(state, thresh=thresh, lambd=lambd, psi=psi)
                reward = self.step(action)
                pnl = reward + (self.lambd * self.pi ** 2) * self.squared_risk
                state = self.get_state()
                done = self.done
                episode_rewards.append(reward)
                episode_pnls.append(pnl)
                episode_positions.append(state[1])
                if done:
                    total_reward = np.sum(episode_rewards)
                    total_pnl = np.sum(episode_pnls)
                    if random_states is not None:
                        scores[random_states[episode]] = total_reward
                        scores_cumsum[random_states[episode]] = np.cumsum(
                            episode_rewards
                        )
                        pnls[random_states[episode]] = episode_pnls
                        positions[random_states[episode]] = episode_positions

                    cumulative_rewards.append(total_reward)
                    cumulative_pnls.append(total_pnl)
                    # print('Episode: {}'.format(episode),
                    #       'Total reward: {:.2f}'.format(total_reward))

        return (
            np.mean(cumulative_rewards),
            scores,
            scores_cumsum,
            np.mean(cumulative_pnls),
            positions,
        )
