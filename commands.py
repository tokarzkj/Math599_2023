import matplotlib.pyplot as plt
import numpy as np
from numpy import real, imag
import signaldefinitions as sd
import dft


def dft_graph():
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
    print("Select a subcommand: kronecker delta, box signal, modulated signal, or reverse signal")
    subcommand = input()

    print("Please select a non-negative integer N")
    N = int(input())
    x = list(range(N))

    if subcommand == "kronecker delta":
        kronecker_delta_properties(N, x)
    elif subcommand == "box signal":
        box_signal_properties(N, x)
    elif subcommand == "modulated signal":
        modulated_signal_properties(N, x)
    elif subcommand == "reverse signal":
        reversed_signal_properties(N, x)


def kronecker_delta_properties(N, x):
    print("Please select an integer k for the Kronecker Delta:")
    k = int(input())

    print("Please select a magnitude for the constance sequence")
    magnitude = int(input())

    kronecker_delta = sd.kronecker_delta(k, N)
    kd_freq = dft.dft_transform(kronecker_delta)

    constance_sequence = sd.constance_sequence(magnitude, N)
    cs_freq = dft.dft_transform(constance_sequence)

    fig, (ax1, ax2) = plt.subplots(2, 3)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.5, hspace=0.75)
    fig.set_figheight(9)
    fig.set_figwidth(15)

    ax1[0].set_ylabel('Signal')
    ax1[0].set_xlabel('Sample')
    ax1[0].set_title('Kronecker Delta Signal')
    ax1[0].stem(x, kronecker_delta)

    ax1[1].set_ylabel('Frequency')
    ax1[1].set_xlabel('Sample')
    ax1[1].set_title('Real(Kronecker Delta Freq)')
    ax1[1].stem(x, [real(r) for r in kd_freq])

    ax1[2].set_ylabel('Frequency')
    ax1[2].set_xlabel('Sample')
    ax1[2].set_title('Im(Kronecker Delta Freq)')
    ax1[2].stem(x, [imag(i) for i in kd_freq])

    ax2[0].set_ylabel('Signal')
    ax2[0].set_xlabel('Sample')
    ax2[0].set_title('Constance Sequence Signal')
    ax2[0].stem(x, constance_sequence)

    ax2[1].set_ylabel('Frequency')
    ax2[1].set_xlabel('Sample')
    ax2[1].set_title('Real(Constance Sequence Freq)')
    ax2[1].stem(x, [real(r) for r in cs_freq])

    ax2[2].set_ylabel('Frequency')
    ax2[2].set_xlabel('Sample')
    ax2[2].set_title('Im(Constance Sequence Freq)')
    ax2[2].stem(x, [imag(i) for i in cs_freq])

    plt.show()


def box_signal_properties(N, x):
    print("Please select an upper boundary M for the box signal (Must be less than N)")
    M = int(input())

    print("Please select a shift z for the box signal")
    z = int(input())

    box_signal = sd.box_signal_samples(N, M)
    box_freq = dft.dft_transform(box_signal)

    reversed_box_signal = sd.reverse_signal(box_signal)
    reversed_box_freq = dft.dft_transform(reversed_box_signal)

    shifted_box_signal = np.roll(box_signal, z)
    shifted_box_freq = dft.shifted_dft_transform(box_signal, z)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 3)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.5, hspace=0.75)
    fig.set_figheight(9)
    fig.set_figwidth(15)

    ax1[0].set_ylabel('Signal')
    ax1[0].set_xlabel('Sample')
    ax1[0].set_title('Box Signal')
    ax1[0].stem(x, box_signal)

    ax1[1].set_ylabel('Frequency')
    ax1[1].set_xlabel('Sample')
    ax1[1].set_title('Real(Box Freq)')
    ax1[1].stem(x, [real(r) for r in box_freq])

    ax1[2].set_ylabel('Frequency')
    ax1[2].set_xlabel('Sample')
    ax1[2].set_title('Im(Box Freq)')
    ax1[2].stem(x, [imag(i) for i in box_freq])

    ax2[0].set_ylabel('Signal')
    ax2[0].set_xlabel('Sample')
    ax2[0].set_title('Reversed Box Signal')
    ax2[0].stem(x, reversed_box_signal)

    ax2[1].set_ylabel('Frequency')
    ax2[1].set_xlabel('Sample')
    ax2[1].set_title('Real(Reversed Box Freq)')
    ax2[1].stem(x, [real(r) for r in reversed_box_freq])

    ax2[2].set_ylabel('Frequency')
    ax2[2].set_xlabel('Sample')
    ax2[2].set_title('Im(Reversed Box Freq)')
    ax2[2].stem(x, [imag(i) for i in reversed_box_freq])

    ax3[0].set_ylabel('Signal')
    ax3[0].set_xlabel('Sample')
    ax3[0].set_title('Shifted Box Signal')
    ax3[0].stem(x, shifted_box_signal)

    ax3[1].set_ylabel('Frequency')
    ax3[1].set_xlabel('Sample')
    ax3[1].set_title('Real(Shifted Box Freq)')
    ax3[1].stem(x, [real(r) for r in shifted_box_freq])

    ax3[2].set_ylabel('Frequency')
    ax3[2].set_xlabel('Sample')
    ax3[2].set_title('Im(Shifted Box Freq)')
    ax3[2].stem(x, [imag(i) for i in shifted_box_freq])

    plt.show()


def modulated_signal_properties(N, x):
    print("Please select an upper boundary M for the box signal (Must be less than N)")
    M = int(input())

    print("Please select a modulation l for the box signal")
    l = int(input())

    box_signal = sd.box_signal_samples(N, M)
    modulated_freq = dft.modulated_signal(box_signal, l)

    fig, (ax1) = plt.subplots(1, 3)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.5, hspace=0.75)
    fig.set_figheight(5)
    fig.set_figwidth(11)

    ax1[0].set_ylabel('Signal')
    ax1[0].set_xlabel('Sample')
    ax1[0].set_title('Shifted Box Signal')
    ax1[0].stem(x, box_signal)

    ax1[1].set_ylabel('Frequency')
    ax1[1].set_xlabel('Sample')
    ax1[1].set_title('Real(Modulated Box Freq)')
    ax1[1].stem(x, [real(r) for r in modulated_freq])

    ax1[2].set_ylabel('Frequency')
    ax1[2].set_xlabel('Sample')
    ax1[2].set_title('Im(Modulated Box Freq)')
    ax1[2].stem(x, [imag(i) for i in modulated_freq])

    plt.show()

def reversed_signal_properties(N, x):
    print("Please select an integer f:")
    f = int(input())

    sin_signal = sd.sin_samples(f, N)
    sin_frequency = dft.dft_transform(sin_signal)

    reverse_sin_signal = dft.reverse_signal(sin_signal)
    reverse_dft = dft.reversed_signal_dft(sin_signal)

    fig, (ax1, ax2) = plt.subplots(2, 3)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1, wspace=0.5, hspace=0.75)
    fig.set_figheight(9)
    fig.set_figwidth(11)

    ax1[0].set_ylabel('Signal')
    ax1[0].set_xlabel('Sample')
    ax1[0].set_title('Sin Signal')
    ax1[0].stem(x, sin_signal)

    ax1[1].set_ylabel('Frequency')
    ax1[1].set_xlabel('Sample')
    ax1[1].set_title('Real Sin Frequency')
    ax1[1].stem(x, [real(r) for r in sin_frequency])

    ax1[2].set_ylabel('Frequency')
    ax1[2].set_xlabel('Sample')
    ax1[2].set_title('Imaginary Sin Frequency')
    ax1[2].stem(x, [imag(i) for i in sin_frequency])

    ax2[0].set_ylabel('Signal')
    ax2[0].set_xlabel('Sample')
    ax2[0].set_title('Reverse Sin Signal')
    ax2[0].stem(x, reverse_sin_signal)

    ax2[1].set_ylabel('Frequency')
    ax2[1].set_xlabel('Sample')
    ax2[1].set_title('Real(Reverse Sin Frequency)')
    ax2[1].stem(x, [real(r) for r in reverse_dft])

    ax2[2].set_ylabel('Frequency')
    ax2[2].set_xlabel('Sample')
    ax2[2].set_title('Im(Reverse Sin Frequency)')
    ax2[2].stem(x, [imag(i) for i in reverse_dft])

    plt.show()
