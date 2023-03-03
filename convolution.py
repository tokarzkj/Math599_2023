import numpy
import numpy as np
from numpy import real, imag
from scipy import signal
import matplotlib.pyplot as plt

import dft
import signaldefinitions as sd


def convolution_dft(sig, mask, shift):
    i = complex(0, 1)
    N = len(sig)

    dft_samples = np.empty(N, dtype=np.complex_)

    shifted_mask = np.roll(mask, shift)
    for n in range(0, N):
        sample_summation = np.float64(0)
        for k in range(0, N):
            signal_sample = sig[k]
            mask_sample = shifted_mask[k]
            sample_summation += signal_sample * mask_sample * np.exp((-2 * np.pi * i * k * n) / N)
        dft_samples[n] = sample_summation

    return dft_samples


def inverse_convolution_dft(dft, mask, shift):
    i = complex(0, 1)
    N = len(dft)

    signal_samples = np.empty(N)
    shifted_mask = np.roll(mask, shift)

    for n in range(0, N):
        sample_summation = np.float64(0)
        for k in range(0, N):
            dft_sample = dft[k]
            sample_summation += dft_sample * np.exp(2*np.pi*i*n*k/N)
        signal_samples[n] = np.float64((1/N) * sample_summation)
    return signal_samples


def shifted_convoluted_dft(sig, sig_shift, mask, shift):
    shifted_signal = np.roll(sig, sig_shift)
    shifted_dft = convolution_dft(shifted_signal, mask, shift)

    i = complex(0, 1)
    N = len(sig)
    for k in range(0, N):
        x_hat = shifted_dft[k]
        shifted_dft[k] = np.exp((-2 * np.pi * i * sig_shift) / N) * x_hat

    return shifted_dft

def prove_convolution_dft(sig, mask, shift):
    signal_dft = dft.dft_transform(sig)
    mask_dft = dft.dft_transform(np.roll(mask, shift))
    n = len(signal_dft)

    dft_samples = np.empty(n, dtype=np.complex_)
    for i in range(0, n):
        dft_samples[i] = signal_dft[i] * mask_dft[i]

    return dft_samples

def convolution_command():
    print("Please select a non-negative integer N")
    N = int(input())

    print("Please select an integer f:")
    f = int(input())

    print("Please select a shift l for the measurement")
    l = -1*int(input())

    print("Please select a sigma for the Gaussian mask")
    sigma = int(input())

    sin_signal = sd.sin_samples(f, N)
    mask = signal.windows.gaussian(N, sigma)
    convoluted_dft = convolution_dft(sin_signal, mask, l)
    component_multiplication_dft = prove_convolution_dft(sin_signal, mask, l)
    idft = inverse_convolution_dft(convoluted_dft, mask, l)

    x = list(range(N))

    graph_convolution(x, component_multiplication_dft, convoluted_dft, l, mask, sin_signal, idft)

    print("Would you like to display a property? Select Shift, Reverse")
    subcommand = input()

    if subcommand == "Shift":
        print("Please select a shift for a shifted signal")
        sig_shift = int(input())

        graph_convolution_shift(x, sin_signal, mask, convoluted_dft, l, sig_shift)
    elif subcommand == "Reverse":
        graph_reverse_signal(sin_signal, mask, l, convoluted_dft, x)

    plt.show()


def graph_reverse_signal(sin_signal, mask, shift, convoluted_dft, x):
    reversed_signal = dft.time_reverse_array(sin_signal)
    reversed_signal_dft = convolution_dft(reversed_signal, mask, shift)
    reversed_convoluted_dft = dft.time_reverse_array(convoluted_dft)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 2)
    fig.suptitle("Reversal Property")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.5, hspace=0.75)
    fig.set_figheight(9)
    fig.set_figwidth(11)

    ax1[0].set_ylabel('Magnitude')
    ax1[0].set_xlabel('Sample')
    ax1[0].set_title('Signal')
    ax1[0].stem(x, sin_signal)

    ax1[1].set_ylabel('Magnitude')
    ax1[1].set_xlabel('Sample')
    ax1[1].set_title('Reversed Signal')
    ax1[1].stem(x, reversed_signal)

    ax2[0].set_ylabel('Magnitude')
    ax2[0].set_xlabel('Sample')
    ax2[0].set_title('Re(DFT)')
    ax2[0].stem(x, [real(r) for r in convoluted_dft])

    ax2[1].set_ylabel('Magnitude')
    ax2[1].set_xlabel('Sample')
    ax2[1].set_title('Im(DFT)')
    ax2[1].stem(x, [imag(i) for i in convoluted_dft])

    ax3[0].set_ylabel('Magnitude')
    ax3[0].set_xlabel('Sample')
    ax3[0].set_title('Re(Reversed DFT)')
    ax3[0].stem(x, [real(r) for r in reversed_convoluted_dft])

    ax3[1].set_ylabel('Magnitude')
    ax3[1].set_xlabel('Sample')
    ax3[1].set_title('Im(Reversed DFT)')
    ax3[1].stem(x, [imag(i) for i in reversed_convoluted_dft])

    ax4[0].set_ylabel('Magnitude')
    ax4[0].set_xlabel('Sample')
    ax4[0].set_title('Re(Reversed Signal DFT)')
    ax4[0].stem(x, [real(r) for r in reversed_signal_dft])

    ax4[1].set_ylabel('Magnitude')
    ax4[1].set_xlabel('Sample')
    ax4[1].set_title('Im(Reversed Signal DFT)')
    ax4[1].stem(x, [imag(i) for i in reversed_signal_dft])


def graph_convolution(x, component_multiplication_dft, convoluted_dft, shift, mask, sin_signal, idft):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 2)
    fig.suptitle("Measurement Convolution Setup")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.5, hspace=0.75)
    fig.set_figheight(9)
    fig.set_figwidth(11)

    ax1[0].set_ylabel('Amplitude')
    ax1[0].set_xlabel('Sample')
    ax1[0].set_title('Sin Signal')
    ax1[0].stem(x, sin_signal)

    ax2[0].set_ylabel('Amplitude')
    ax2[0].set_xlabel('Sample')
    ax2[0].set_title('Mask Signal')
    ax2[0].stem(x, mask)
    ax2[1].set_ylabel('Amplitude')
    ax2[1].set_xlabel('Sample')
    ax2[1].set_title('Shifted Mask Signal')
    ax2[1].stem(x, numpy.roll(mask, shift))

    ax3[0].set_ylabel('Magnitude')
    ax3[0].set_xlabel('Sample')
    ax3[0].set_title('Re(Convoluted DFT)')
    ax3[0].stem(x, [real(r) for r in convoluted_dft])
    ax3[1].set_ylabel('Magnitude')
    ax3[1].set_xlabel('Sample')
    ax3[1].set_title('Im(Convoluted DFT)')
    ax3[1].stem(x, [imag(i) for i in convoluted_dft])

    ax4[0].set_ylabel('Magnitude')
    ax4[0].set_xlabel('Sample')
    ax4[0].set_title('Re(Component Multiplication DFT)')
    ax4[0].stem(x, [real(r) for r in component_multiplication_dft])
    ax4[1].set_ylabel('Magnitude')
    ax4[1].set_xlabel('Sample')
    ax4[1].set_title('Im(Component Multiplication DFT)')
    ax4[1].stem(x, [imag(i) for i in component_multiplication_dft])

    ax5[0].set_ylabel('Amplitude')
    ax5[0].set_xlabel('Sample')
    ax5[0].set_title('IDFT Sin Signal')
    ax5[0].stem(x, idft)


def graph_convolution_shift(x, sin_signal, mask, convoluted_dft, shift, sig_shift):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 2)
    fig.suptitle("Shifted Signal Property")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.5, hspace=0.75)
    fig.set_figheight(9)
    fig.set_figwidth(11)

    shifted_signal = np.roll(sin_signal, sig_shift)
    shifted_dft = shifted_convoluted_dft(sin_signal, sig_shift, mask, shift)

    ax1[0].set_ylabel('Amplitude')
    ax1[0].set_xlabel('Sample')
    ax1[0].set_title('Signal')
    ax1[0].stem(x, sin_signal)

    ax1[1].set_ylabel('Amplitude')
    ax1[1].set_xlabel('Sample')
    ax1[1].set_title('Shifted Signal')
    ax1[1].stem(x, shifted_signal)

    ax2[0].set_ylabel('Magnitude')
    ax2[0].set_xlabel('Sample')
    ax2[0].set_title('Re(Shifted DFT)')
    ax2[0].stem(x, [real(r) for r in convoluted_dft])

    ax2[1].set_ylabel('Magnitude')
    ax2[1].set_xlabel('Sample')
    ax2[1].set_title('Im(Shifted DFT)')
    ax2[1].stem(x, [imag(i) for i in convoluted_dft])

    ax3[0].set_ylabel('Magnitude')
    ax3[0].set_xlabel('Sample')
    ax3[0].set_title('Re(Shifted Signal DFT)')
    ax3[0].stem(x, [real(r) for r in shifted_dft])

    ax3[1].set_ylabel('Magnitude')
    ax3[1].set_xlabel('Sample')
    ax3[1].set_title('Im(Shifted Signal DFT)')
    ax3[1].stem(x, [imag(i) for i in shifted_dft])