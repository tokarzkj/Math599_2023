import numpy as np


def cosine_samples(f, N) -> np.array:
    """
    Calculates N-samples of a specific cosine signal
    :param f: An integer
    :param N: A non-zero integer representing how many samples to take
    :return: N length vector containing samples of a cosine signal
    """
    signal_vector_results = np.empty((N, 1))
    for n in range(0, N):
        signal_vector_results[n] = np.cos((2 * np.pi * f * n)/N)

    return signal_vector_results


def sin_samples(f, N) -> np.array:
    """
    Calculates N samples of a specific sin signal
    :param f: An integer
    :param N: A non-zero integer representing how many samples to take
    :return: N length vector containing samples of a sin signal
    """
    signal_vector_results = np.empty((N, 1))
    for n in range(0, N):
        signal_vector_results[n] = np.sin(2 * np.pi * f * n)

    return signal_vector_results


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
            frequency_summation += frequency_sample * np.exp((2 * np.pi * i * k * n)/N)
        signal_samples[k] = np.float64((1/N) * frequency_summation)

    return signal_samples


if __name__ == '__main__':
    f = 1
    N = 8
    cos_signal = cosine_samples(f, N)
    sin_signal = sin_samples(f, N)

    cos_frequency = dft_transform(cos_signal)
    sin_frequency = dft_transform(sin_signal)

    idft_cos_signal = inverse_dft_transform(cos_frequency)
    idft_sin_signal = inverse_dft_transform(sin_frequency)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
