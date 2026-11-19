def systematic(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(w)
    pointers = (rng.random() + np.arange(n)) / n
    return _invert(_cdf(w), pointers)
