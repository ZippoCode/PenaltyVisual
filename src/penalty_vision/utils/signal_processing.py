import numpy as np
from scipy.signal import savgol_filter


def smooth_signal(signal: np.ndarray, window_length: int = 7, poly_order: int = 2) -> np.ndarray:
    if len(signal) < window_length:
        window_length = len(signal) if len(signal) % 2 == 1 else len(signal) - 1

    if window_length < poly_order + 1:
        poly_order = window_length - 1 if window_length > 1 else 1

    if window_length < 2:
        return signal

    smoothed = savgol_filter(signal, window_length=window_length, polyorder=poly_order)
    return smoothed
