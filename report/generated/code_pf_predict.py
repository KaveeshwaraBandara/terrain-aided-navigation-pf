def predict(self, u: np.ndarray, dt: float) -> None:
    n = self.n
    bias = self.particles[:, 2:4]  # b_{k-1}, read before the bias is perturbed below
    self.particles[:, :2] += dt * (np.asarray(u, dtype=float) - bias) + self.rng.normal(
        0.0, self.sigma_pos, size=(n, 2)
    )
    self.particles[:, 2:4] += self.rng.normal(0.0, self.sigma_bias, size=(n, 2))
    if self.hb_on:
        self.particles[:, 4] += self.rng.normal(0.0, self.hb_rw, size=n)
