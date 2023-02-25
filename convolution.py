import numpy
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import signaldefinitions as sd


def convolution_dft(signal, mask, shift):
    i = complex(0, 1)
    N = len(signal)

    dft_samples = np.empty(N, dtype=np.complex_)

    shifted_mask = np.roll(mask, shift)
    for n in range(0, N):
        sample_summation = np.float64(0)
        for k in range(0, N):
            signal_sample = signal[k]
            mask_sample = shifted_mask[k]
            sample_summation += signal_sample * mask_sample * np.exp((-2 * np.pi * i * k * n) / N)
        dft_samples[n] = sample_summation

    return dft_samples


def convolution_command():
    print("Please select a non-negative integer N")
    N = int(input())

    print("Please select an integer f:")
    f = int(input())

    print("Please select a shift l")
    l = -1*int(input())

    print("Please select a sigma for the Gaussian mask")
    sigma = int(input())

    sin_signal = sd.sin_samples(f, N)
    mask = signal.windows.gaussian(N, sigma)
    dft = convolution_dft(sin_signal, mask, l)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.5, hspace=0.75)
    fig.set_figheight(9)
    fig.set_figwidth(11)

    x = list(range(N))

    ax1.set_ylabel('Amplitude')
    ax1.set_xlabel('Sample')
    ax1.set_title('Sin Signal')
    ax1.stem(x, sin_signal)

    ax2.set_ylabel('Amplitude')
    ax2.set_xlabel('Sample')
    ax2.set_title('Mask Signal')
    ax2.stem(x, mask)

    ax3.set_ylabel('Amplitude')
    ax3.set_xlabel('Sample')
    ax3.set_title('Shifted Mask Signal')
    ax3.stem(x, numpy.roll(mask, l))

    ax4.set_ylabel('Magnitude')
    ax4.set_xlabel('Sample')
    ax4.set_title('Convoluted DFT')
    ax4.stem(x, dft)
    plt.show()




