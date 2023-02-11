import numpy as np
import matplotlib.pyplot as plt
from numpy import real, imag
import signaldefinitions as sd
import dft


def DftGraph():
    print("Please select an integer f:")
    f = int(input())
    print("Please select a non-negative integer N")
    N = int(input())
    print("Please select a lower boundary u for the box signal")
    u = int(input())
    print("Please select an upper boundary M for the box signal (Must be less than N)")
    M = int(input())

    while M >= N:
        print("Please select an upper boundary M for the box signal (Must be less than N)")
        M = int(input())

    cos_signal = sd.cosine_samples(f, N)
    sin_signal = sd.sin_samples(f, N)
    simple_box_signal = sd.box_signal_samples(N, M)
    shifted_box_signal = sd.box_signal_samples(N, M, u)

    cos_frequency = dft.dft_transform(cos_signal)
    sin_frequency = dft.dft_transform(sin_signal)
    simple_box_frequency = dft.dft_transform(simple_box_signal)
    shifted_box_frequency = dft.dft_transform(shifted_box_signal)

    idft_cos_signal = dft.inverse_dft_transform(cos_frequency)
    idft_sin_signal = dft.inverse_dft_transform(sin_frequency)
    idft_simple_box_signal = dft.inverse_dft_transform(simple_box_frequency)
    idft_shifted_box_signal = dft.inverse_dft_transform(shifted_box_frequency)

    x = list(range(N))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 4)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.5, hspace=0.75)
    fig.set_figheight(9)
    fig.set_figwidth(15)

    ####################################################################
    # Create the Cosine plots                                          #
    ####################################################################
    ax1[0].set_ylabel('Signal')
    ax1[0].set_xlabel('Sample')
    ax1[0].set_title('Cosine Signal')
    ax1[0].stem(x, cos_signal)

    ax1[1].set_ylabel('Frequency')
    ax1[1].set_xlabel('Sample')
    ax1[1].set_title('Real Cosine Frequency')
    ax1[1].stem(x, [real(r) for r in cos_frequency])

    ax1[2].set_ylabel('Frequency')
    ax1[2].set_xlabel('Sample')
    ax1[2].set_title('Imaginary Cosine Frequency')
    ax1[2].stem(x, [imag(i) for i in cos_frequency])

    ax1[3].set_ylabel('Signal')
    ax1[3].set_xlabel('Sample')
    ax1[3].set_title('IDFT Cosine Signal')
    ax1[3].stem(x, idft_cos_signal)

    ####################################################################
    # Create the Sine plots                                            #
    ####################################################################
    ax2[0].set_ylabel('Signal')
    ax2[0].set_xlabel('Sample')
    ax2[0].set_title('Sine Signal')
    ax2[0].stem(x, sin_signal)

    ax2[1].set_ylabel('Frequency')
    ax2[1].set_xlabel('Sample')
    ax2[1].set_title('Real Sine Frequency')
    ax2[1].stem(x, [real(r) for r in sin_frequency])

    ax2[2].set_ylabel('Frequency')
    ax2[2].set_xlabel('Sample')
    ax2[2].set_title('Imaginary Sine Frequency')
    ax2[2].stem(x, [imag(i) for i in sin_frequency])

    ax2[3].set_ylabel('Frequency')
    ax2[3].set_xlabel('Sample')
    ax2[3].set_title('IDFT Sine Signal')
    ax2[3].stem(x, idft_sin_signal)

    ####################################################################
    # Create the Simple Box plots                                      #
    ####################################################################
    ax3[0].set_ylabel('Signal')
    ax3[0].set_xlabel('Sample')
    ax3[0].set_title('Simple Box Signal')
    ax3[0].stem(x, simple_box_signal)

    ax3[1].set_ylabel('Frequency')
    ax3[1].set_xlabel('Sample')
    ax3[1].set_title('Real Simple Box Frequency')
    ax3[1].stem(x, [real(r) for r in simple_box_frequency])

    ax3[2].set_ylabel('Frequency')
    ax3[2].set_xlabel('Sample')
    ax3[2].set_title('Imaginary Simple Box Frequency')
    ax3[2].stem(x, [imag(i) for i in simple_box_frequency])

    ax3[3].set_ylabel('Signal')
    ax3[3].set_xlabel('Sample')
    ax3[3].set_title('IDFT Simple Box Signal')
    ax3[3].stem(x, idft_simple_box_signal)

    ####################################################################
    # Create the Shifted Box plots                                     #
    ####################################################################
    ax4[0].set_ylabel('Signal')
    ax4[0].set_xlabel('Sample')
    ax4[0].set_title('Shifted Box Signal')
    ax4[0].stem(x, shifted_box_signal)

    ax4[1].set_ylabel('Frequency')
    ax4[1].set_xlabel('Sample')
    ax4[1].set_title('Real Shifted Box Frequency')
    ax4[1].stem(x, [real(r) for r in shifted_box_frequency])

    ax4[2].set_ylabel('Frequency')
    ax4[2].set_xlabel('Sample')
    ax4[2].set_title('Imaginary Shifted Box Frequency')
    ax4[2].stem(x, [imag(i) for i in shifted_box_frequency])

    ax4[3].set_ylabel('Signal')
    ax4[3].set_xlabel('Sample')
    ax4[3].set_title('IDFT Shifted Box Signal')
    ax4[3].stem(x, idft_shifted_box_signal)

    plt.show()

def verify_dft_properties():
    print("Please select an integer k for the Kronecker Delta:")
    k = int(input())

    print("Please select a non-negative integer N")
    N = int(input())

    print("Please select a magnitude for the constance sequence")
    magnitude = int(input())

    kronecker_delta = sd.kronecker_delta(k, N)
    kd_freq = dft.dft_transform(kronecker_delta)

    constance_sequence = sd.constance_sequence(magnitude, N)
    cs_freq = dft.dft_transform(constance_sequence)

    x = list(range(N))

    fig, (ax1, ax2) = plt.subplots(2, 2)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.5, hspace=0.75)
    fig.set_figheight(9)
    fig.set_figwidth(15)

    ax1[0].set_ylabel('Signal')
    ax1[0].set_xlabel('Sample')
    ax1[0].set_title('Kronecker Delta Signal')
    ax1[0].stem(x, kronecker_delta)

    ax1[1].set_ylabel('Frequency')
    ax1[1].set_xlabel('Sample')
    ax1[1].set_title('Kronecker Delta Freq')
    ax1[1].stem(x, kd_freq)

    ax2[0].set_ylabel('Signal')
    ax2[0].set_xlabel('Sample')
    ax2[0].set_title('Kronecker Delta Signal')
    ax2[0].stem(x, constance_sequence)

    ax2[1].set_ylabel('Frequency')
    ax2[1].set_xlabel('Sample')
    ax2[1].set_title('Kronecker Delta Freq')
    ax2[1].stem(x, cs_freq)

    plt.show()
