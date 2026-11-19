def step(self, u: np.ndarray, dt: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    if self.monitor_on:
        self._propagate_dr(u, dt)
    self.predict(u, dt)
    self.update(z)
    alarm = self.monitor_on and self._check_monitor()
    if alarm:
        self._respread()

    ess = effective_sample_size(self.weights)
    resampled = ess < self.ess_threshold * self.n
    if resampled:
        idx = self.resample_fn(self.weights, self.rng)
        self.particles = self.particles[idx]
        self.log_weights = np.full(self.n, -np.log(self.n))
        if self.roughening > 0:
            self.roughen()

    self._k += 1
    self.history["ess"].append(float(ess))
    self.history["resampled"].append(bool(resampled))
    self.history["log_evidence"].append(self._last_log_evidence)
    self.history["nis"].append(self._last_nis)
    self.history["alarms"].append(bool(alarm))
    if self.hb_on:
        self.history["height_bias"].append(float(self.weights @ self.particles[:, 4]))
    if self.snapshot_every and self._k % self.snapshot_every == 0:
        self.snapshots.append((self._k, self.particles[:, :2].copy()))
    return self.estimate()
