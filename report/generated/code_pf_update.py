def update(self, z: float) -> None:
    z = float(z)
    px, py = self.particles[:, 0], self.particles[:, 1]
    h = self.dem.height(px, py)
    if self.hb_on:
        h = h + self.particles[:, 4]  # predicted measurement includes the offset state
    loglik = self.log_likelihood(z - h)

    w_prev = self.weights
    z_hat = float(w_prev @ h)
    s_pred = float(w_prev @ (h - z_hat) ** 2) + self.meas_sigma**2
    self._last_nis = (z - z_hat) ** 2 / s_pred

    new = self.log_weights + np.where(self.dem.contains(px, py), loglik, -np.inf)
    evidence = logsumexp(new)
    if not np.isfinite(evidence):
        new = self.log_weights + loglik
        evidence = logsumexp(new)
    self._last_log_evidence = float(evidence)
    self.log_weights = new - evidence  # keep sum w_i = 1 in log form
