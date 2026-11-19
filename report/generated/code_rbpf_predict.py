def predict(self, u: np.ndarray, dt: float) -> None:
    u = np.asarray(u, dtype=float)
    P = self.P_bias
    S = dt * dt * P + self.sigma_pos**2 * np.eye(2)
    # 1. sample positions with the bias marginalised out
    chol = np.linalg.cholesky(S)
    b_hat = self.particles[:, 2:]
    mean = self.particles[:, :2] + dt * (u - b_hat)
    new_pos = mean + self.rng.standard_normal((self.n, 2)) @ chol.T
    # 2. the increment is a linear measurement of b_{k-1}
    eta = new_pos - self.particles[:, :2] - dt * u
    L = -dt * P @ np.linalg.inv(S)
    b_hat = b_hat + (eta + dt * b_hat) @ L.T
    P = P - L @ S @ L.T
    # 3. bias random walk
    P = P + self.sigma_bias**2 * np.eye(2)
    self.particles = np.column_stack([new_pos, b_hat])
    self.P_bias = 0.5 * (P + P.T)
