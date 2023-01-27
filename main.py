import numpy as np


def cosine_samples(f, N):
    signal_vector_results = np.empty(N)
    for n in range(0, N):
        signal_vector_results[n] = np.cos((2 * np.pi * f * n)/N)

    return signal_vector_results


def sin_samples(f, N) -> np.array:
    signal_vector_results = np.empty(N)
    for n in range(0, N):
        signal_vector_results[n] = np.sin(2 * np.pi * f * n)

    return signal_vector_results


def perform_dft_transform(signal) -> np.array:
    i = complex(0, 1)
    N = len(signal)

    dft_samples = np.empty((N, 1), dtype=np.complex_)

    for n in range(0, N):
        sample_summation = 0
        for k in range(0, N):
            signal_sample = signal[k]
            sample_summation += signal_sample * np.exp((-2 * np.pi * i * k * n) / N)
        dft_samples[n] = sample_summation

    return dft_samples


if __name__ == '__main__':
    f = 1
    N = 8
    cos_signal = cosine_samples(f, N)
    sin_signal = sin_samples(f, N)

    cos_frequency = perform_dft_transform(cos_signal)
    sin_frequency = perform_dft_transform(sin_signal)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
