import numpy as np


class Mess3HMM:
    """
    3-state HMM with token-conditional transition matrices (Mess3 process).

    The process is defined by three matrices T^(x) where
      T^(x)[i, j] = P(emit token x, transition to state j | current state i).

    This couples emission and transition: emitting token x correlates with
    transitioning toward state x. Each observed token applies a different
    affine map to belief space, b -> normalize(b @ T^(x)), creating an
    iterated function system that produces fractal belief geometry.

    Params:
      s: self-transition strength. P(emit own token, stay in own state).
         Higher = stickier dynamics. Range (0, 1).
      r: asymmetry ratio for off-diagonal coupled transitions.
         r=1: symmetric. r>1: clockwise bias. r<1: counterclockwise bias.
      coupling: fraction of non-self mass going to coupled (emit x, go to x)
         transitions vs uniform leakage. Default 0.8.
    """

    def __init__(self, s: float = 0.7, r: float = 1.5, coupling: float = 0.8, seed: int = 0):
        self.s = float(s)
        self.r = float(r)
        self.coupling = float(coupling)
        self.rng = np.random.default_rng(seed)

        # Build token-conditional transition matrices T^(x)
        # T_x[x][i, j] = P(emit x, transition to j | state i)
        self.T_x = [np.zeros((3, 3)) for _ in range(3)]

        for z in range(3):
            cw = (z + 1) % 3
            ccw = (z + 2) % 3
            remaining = 1.0 - self.s

            # Self-loop: emit own token, stay in own state
            self.T_x[z][z, z] = self.s

            # Coupled transitions: emit token x, go to state x
            coupled_mass = remaining * self.coupling
            r_cw = self.r / (1.0 + self.r)
            r_ccw = 1.0 / (1.0 + self.r)
            self.T_x[cw][z, cw] = coupled_mass * r_cw
            self.T_x[ccw][z, ccw] = coupled_mass * r_ccw

            # Uniform leakage over remaining 6 (token, next_state) pairs
            leak_mass = remaining * (1.0 - self.coupling)
            for x in range(3):
                for zp in range(3):
                    if (x == z and zp == z):
                        continue
                    if (x == cw and zp == cw):
                        continue
                    if (x == ccw and zp == ccw):
                        continue
                    self.T_x[x][z, zp] += leak_mass / 6.0

        # Marginal transition matrix T[i,j] = sum_x T^(x)[i,j]
        self.T = sum(self.T_x)

        # Marginal emission P(x | z) = sum_j T^(x)[z, j]
        self.E = np.array([self.T_x[x].sum(axis=1) for x in range(3)]).T  # (3, 3)

        # Stationary distribution (left eigenvector of T with eigenvalue 1)
        eigvals, eigvecs = np.linalg.eig(self.T.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi = np.real(eigvecs[:, idx])
        pi = np.maximum(pi, 0.0)
        if pi.sum() == 0:
            pi = np.ones(3)
        self.pi = pi / pi.sum()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_sequence(self, length: int):
        """
        Sample one trajectory of observations and hidden states.

        Returns (z, y) where:
          z: (length,) int array of hidden states
          y: (length,) int array of observed tokens
        """
        z = np.zeros(length + 1, dtype=int)
        y = np.zeros(length, dtype=int)

        z[0] = self.rng.choice(3, p=self.pi)

        for t in range(length):
            # Joint distribution over (token, next_state) from current state z[t]
            # Flatten the 3x3 table [T_x[0][z[t],:], T_x[1][z[t],:], T_x[2][z[t],:]]
            joint = np.array([self.T_x[tok][z[t]] for tok in range(3)])  # (3, 3)
            flat = joint.ravel()  # (9,)
            idx = self.rng.choice(9, p=flat)
            y[t] = idx // 3
            z[t + 1] = idx % 3

        return z[:length], y

    # ------------------------------------------------------------------
    # Forward filtering: beliefs via token-conditional matrices
    # ------------------------------------------------------------------

    def forward_beliefs(self, y):
        """
        Compute filtered beliefs alpha_t = P(z_t | y_0, ..., y_{t-1}).

        The update rule is: alpha_{t+1} = normalize(alpha_t @ T^(y_t))

        Args:
          y: (length,) array of observed tokens in {0, 1, 2}.

        Returns:
          beliefs: (length, 3) array, one belief per timestep after each observation.
        """
        length = len(y)
        # alpha[0] is the prior, alpha[t] is the belief after seeing y[0:t]
        alpha = np.zeros((length + 1, 3))
        alpha[0] = self.pi.copy()

        for t in range(length):
            alpha[t + 1] = alpha[t] @ self.T_x[y[t]]
            s = alpha[t + 1].sum()
            if s > 0:
                alpha[t + 1] /= s
            else:
                alpha[t + 1] = alpha[t]  # degenerate fallback

        return alpha[1:]  # (length, 3): belief after each observation


def generate_beliefs_batch(
    hmm: Mess3HMM,
    batch_size: int = 3000,
    seq_len: int = 64,
    seed: int = 7,
):
    """
    Generate many belief trajectories for visualization/analysis.

    Returns:
      beliefs: np.ndarray of shape (batch_size * seq_len, 3)
    """
    rng = np.random.default_rng(seed)
    all_beliefs = []

    for _ in range(batch_size):
        hmm.rng = np.random.default_rng(rng.integers(2**31 - 1))
        _z, y = hmm.sample_sequence(seq_len)
        alpha = hmm.forward_beliefs(y)
        all_beliefs.append(alpha)

    all_beliefs = np.stack(all_beliefs, axis=0)  # (batch_size, seq_len, 3)
    return all_beliefs.reshape(-1, 3)
