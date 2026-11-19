def update(self, z: float) -> None:
    px, py = self.x[0], self.x[1]
    h = float(self.dem.height(px, py))
    gx, gy = self.dem.gradient(px, py)
    H = np.array([[float(gx), float(gy), 0.0, 0.0]])

    S = float((H @ self.P @ H.T)[0, 0]) + self.R
    K = (self.P @ H.T) / S  # (4, 1)
    self.x = self.x + K[:, 0] * (float(z) - h)

    A = np.eye(4) - K @ H
    self.P = A @ self.P @ A.T + self.R * (K @ K.T)
    self.P = 0.5 * (self.P + self.P.T)  # remove round-off asymmetry
