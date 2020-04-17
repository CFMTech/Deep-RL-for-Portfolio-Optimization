import numpy as np


def build_ou_process(T=100000, theta=0.1, sigma=0.1, random_state=None):
    """
    Description
    ---------------
    Build a discrete OU process signal of length T starting at p_0=0:
    ```
        p_t - p_{t-1} = -theta*p_{t-1} + sigma*epsilon_t;
    ```
    (epsilon_t)_t are standard normal random variables


    Parameters:
    ---------------
    T : Int, length of the signal.
    theta : Float>0, parameter of the OU process.
    sigma : Float>0, parameter of the OU process.
    random_state : None or Int, if Int, generate the same sequence of noise each time.

    Returns
    ---------------
    np.array of shape (T,), the OU signal generated.
    """
    X = np.empty(T)
    t = 0
    x = 0.0
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        normals = rng.normal(0, 1, T)

    else:
        normals = np.random.normal(0, 1, T)

    for t in range(T):
        x += -x * theta + sigma * normals[t]
        X[t] = x
    X /= sigma * np.sqrt(1.0 / 2.0 / theta)
    return X


def get_returns(signal, random_state=None):
    """
    Description
    ---------------
    Compute the returns r_t = p_t + eta_t, where p_t is the signal and eta_t is a Gaussian
    white noise.

    Parameters
    ---------------
    signal : 1D np.array, the signal computed as a sample path of an OU process.
    random_state : Int or None:
        - if None, do not use a random state (useful to simulate different paths each time
          running the simulation).
        - if Int, use a random state (useful to compare different experimental results).

    Returns
    ---------------
    1D np.array containing the returns
    """

    if random_state is not None:
        rng = np.random.RandomState(random_state)
        return signal + rng.normal(size=signal.size)

    else:
        return signal + np.random.normal(size=signal.size)
