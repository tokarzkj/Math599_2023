import numpy as np


def dft_transform(signal) -> np.array:
    """
    Use the DFT summation technique to transform a signal vector into frequency vector
    :param signal: N length vector containing samples of a signal
    :return: An N length vector of frequency samples
    """
    i = complex(0, 1)
    N = len(signal)

    dft_samples = np.empty((N, 1), dtype=np.complex_)

    for n in range(0, N):
        sample_summation = np.float64(0)
        for k in range(0, N):
            signal_sample = signal[k]
            sample_summation += signal_sample * np.exp((-2 * np.pi * i * k * n) / N)
        dft_samples[n] = sample_summation

    return dft_samples


def shifted_dft_transform(signal, z):
    shifted_signal = np.roll(signal, z)
    dft = dft_transform(shifted_signal)

    i = complex(0, 1)
    N = len(signal)
    for k in range(0, N):
        x_hat = dft[k]
        dft[k] = np.exp((-2 * np.pi * i * z)/N) * x_hat

    return dft


def inverse_dft_transform(frequency) -> np.array:
    """
    Uses the inverse DFT summation technique to recover the signal from a frequency.
    Currently subject to some rounding errors when comparing to the original signal values.
    :param frequency: N length vector containing samples of a frequency
    :return: An N length vector of signal samples
    """
    i = complex(0, 1)
    N = len(frequency)
    signal_samples = np.empty((N, 1), dtype=np.float_)

    for k in range(0, N):
        frequency_summation = np.float64(0)
        for n in range(0, N):
            frequency_sample = frequency[n]
            frequency_summation += frequency_sample * np.exp((2 * np.pi * i * k * n) / N)
        signal_samples[k] = np.float64((1 / N) * frequency_summation)

    return signal_samples


def modulated_signal(signal, l):
    dft = dft_transform(signal)

    i = complex(0, 1)
    N = len(signal)
    for k in range(0, N):
        x_hat = dft[k]
        dft[k] = np.exp((2 * np.pi * i * l * k) / N) * x_hat

    return dft