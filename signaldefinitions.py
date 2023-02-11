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
        signal_vector_results[n] = np.cos((2 * np.pi * f * n) / N)

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
        signal_vector_results[n] = np.sin((2 * np.pi * f * n) / N)

    return signal_vector_results


def box_signal_samples(N, M, u=0) -> np.array:
    samples = np.empty(N)
    start = u
    end = M + u

    for n in range(start, end):
        samples[n] = 1

    return samples


def imaginary_sample(f, N) -> np.array:
    """
    Calculates N-samples of a specific cosine signal
    :param f: An integer
    :param N: A non-zero integer representing how many samples to take
    :return: N length vector containing samples of a cosine signal
    """
    signal_vector_results = np.empty((N, 1))
    for n in range(0, N):
        signal_vector_results[n] = np.exp((2 * np.pi * f * n))

    return signal_vector_results


def kronecker_delta(k, N):
    samples = np.zeros(N)
    samples[k] = 1
    return samples


def constance_sequence(magnitude, N):
    return np.full(N, magnitude)
