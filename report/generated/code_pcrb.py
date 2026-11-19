def pcrb(
    expected_slope_outer: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: float, dt: float
) -> np.ndarray:
    F = transition(dt)
    J = np.linalg.inv(P0)
    out = np.empty((len(expected_slope_outer), 4, 4))
    for k, G in enumerate(expected_slope_outer):
        J = np.linalg.inv(Q + F @ np.linalg.solve(J, F.T))
        J[:2, :2] += G / R
        out[k] = np.linalg.inv(J)
    return out
