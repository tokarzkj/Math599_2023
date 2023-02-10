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


def conjugate_signal_transform(signal):
    """
    Uses the FT summation technique to recover the signal from a frequency. This method will calculate the conjugate
    of your signal. Currently subject to some rounding errors when comparing to the original signal values.
    :param signal: N length vector containing samples of a frequency
    :return: An N length vector of signal samples
    """
    conjugate_signal = np.conjugate(signal)
    return dft_transform(conjugate_signal)
