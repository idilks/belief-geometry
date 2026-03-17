import numpy as np


class Mess3HMM:
    """
    A simple 3-state HMM loosely inspired by the mess3 process.

    Params:
      x: how 'mixed' the off-diagonal transitions are (0 = no mixing, larger = more mess).
      a: how likely you are to switch vs stay (0 = always stay, 1 = always switch).
    """

    def __init__(self, x: float = 0.15, a: float = 0.6, seed: int = 0):
        self.x = float(x)
        self.a = float(a)
        self.rng = np.random.default_rng(seed)

        # ---- Hidden-state transitions P(z_{t+1} | z_t) ----
        # Base transition structure: prefer staying, with some "messy" chance of jumping.
        #
        # We'll make a 3x3 matrix T where:
        #   T[i,i]   = 1 - a           (probability of staying)
        #   T[i,j≠i] = a * w_ij / Z_i  (probability of jumping, normalized)
        #
        # and w_ij depends on x to control how uneven the off-diagonals are.
        W = np.array([
            [0.0, 1.0, 1.0 + x],   # from state 0, prefer 2 slightly more than 1
            [1.0 + x, 0.0, 1.0],   # from state 1, prefer 0 slightly more
            [1.0, 1.0 + x, 0.0],   # from state 2, prefer 1 slightly more
        ])
        T = np.zeros((3, 3))
        for i in range(3):
            stay = 1.0 - self.a
            move_mass = self.a
            off = W[i].copy()
            off[i] = 0.0
            if off.sum() == 0:
                # purely self-persistent
                T[i, i] = 1.0
                continue
            off = off / off.sum()  # normalize off-diagonal preferences
            T[i, i] = stay
            T[i] += move_mass * off
        self.T = T  # shape (3, 3)

        # ---- Initial distribution over hidden states ----
        # Use stationary distribution of T (left eigenvector with eigenvalue 1).
        eigvals, eigvecs = np.linalg.eig(self.T.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi = np.real(eigvecs[:, idx])
        pi = np.maximum(pi, 0.0)
        if pi.sum() == 0:
            pi = np.array([1.0, 1.0, 1.0])
        pi = pi / pi.sum()
        self.pi = pi  # shape (3,)

        # ---- Emission matrix P(y | z) ----
        # Simple discrete emissions:
        #   - each hidden state i mostly emits observation i
        #   - with small chance of "confusion" proportional to x
        base = 0.8   # mass on "correct" symbol
        noise = 0.2  # total mass to spread on others
        E = np.zeros((3, 3))
        for z in range(3):
            E[z, z] = base
            others = [j for j in range(3) if j != z]
            # make noise slightly asymmetric using x
            # (not important; just for flavor)
            if x > 0:
                weights = np.array([1.0, 1.0 + x])
                weights = weights / weights.sum()
            else:
                weights = np.array([0.5, 0.5])
            for w, j in zip(weights, others):
                E[z, j] = noise * w
        self.E = E  # shape (3, 3)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_sequence(self, T: int):
        """Sample one trajectory (z_1:T, y_1:T) from the HMM."""
        z = np.zeros(T, dtype=int)
        y = np.zeros(T, dtype=int)

        # z_1 ~ pi
        z[0] = self.rng.choice(3, p=self.pi)
        y[0] = self.rng.choice(3, p=self.E[z[0]])

        for t in range(1, T):
            z[t] = self.rng.choice(3, p=self.T[z[t - 1]])
            y[t] = self.rng.choice(3, p=self.E[z[t]])

        return z, y

    # ------------------------------------------------------------------
    # Forward algorithm to compute beliefs p(z_t | y_{1:t})
    # ------------------------------------------------------------------

    def forward_beliefs(self, y):
        """
        Compute filtered beliefs alpha_t(i) = p(z_t = i | y_1:t) for a single sequence.

        Args:
          y: array of shape (T,) with discrete observations in {0,1,2}.

        Returns:
          beliefs: array of shape (T, 3), each row sums to 1.
        """
        T_len = len(y)
        alpha = np.zeros((T_len, 3))

        # t = 1
        # unnormalized alpha_1(i) ∝ pi(i) * P(y1 | z1=i)
        alpha[0] = self.pi * self.E[:, y[0]]
        alpha[0] /= alpha[0].sum()

        # t > 1
        for t in range(1, T_len):
            # prediction: p(z_t = j | y_1:t-1) = sum_i alpha_{t-1}(i) T[i,j]
            pred = alpha[t - 1] @ self.T
            # update with emission
            alpha[t] = pred * self.E[:, y[t]]
            s = alpha[t].sum()
            if s <= 0:
                # degenerate; fall back to prediction
                alpha[t] = pred
                s = alpha[t].sum()
            alpha[t] /= s
        return alpha


def generate_beliefs_batch(
    hmm: Mess3HMM,
    batch_size: int = 3000,
    seq_len: int = 64,
    seed: int = 7,
):
    """
    Rough analogue of generate_beliefs(process, BATCH_SIZE, SEQ_LEN, SEED)
    but for our stand-alone Mess3HMM.

    Returns:
      beliefs: np.ndarray of shape (batch_size * seq_len, 3)
    """
    rng = np.random.default_rng(seed)

    all_beliefs = []

    for _ in range(batch_size):
        # sample one sequence
        z, y = hmm.sample_sequence(seq_len)
        # (optional) re-seed internal RNG for variety
        hmm.rng = np.random.default_rng(rng.integers(2**31 - 1))

        # compute filtered beliefs for this sequence
        alpha = hmm.forward_beliefs(y)  # shape (seq_len, 3)
        all_beliefs.append(alpha)

    all_beliefs = np.stack(all_beliefs, axis=0)  # (batch_size, seq_len, 3)
    flat = all_beliefs.reshape(-1, 3)            # (batch_size * seq_len, 3)
    return flat