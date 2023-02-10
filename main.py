import numpy as np
import matplotlib.pyplot as plt
from numpy import real, imag
import signaldefinitions as sd
import dft

if __name__ == '__main__':
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
    conjugate_cos_frequency = dft.conjugate_signal_transform(cos_signal)
    sin_frequency = dft.dft_transform(sin_signal)
    conjugate_sin_frequency = dft.conjugate_signal_transform(sin_signal)
    simple_box_frequency = dft.dft_transform(simple_box_signal)
    conjugate_simple_box = dft.dft_transform(simple_box_signal)
    shifted_box_frequency = dft.dft_transform(shifted_box_signal)
    conjugate_shifted_box = dft.dft_transform(shifted_box_signal)

    idft_cos_signal = dft.inverse_dft_transform(cos_frequency)
    idft_sin_signal = dft.inverse_dft_transform(sin_frequency)
    idft_simple_box_signal = dft.inverse_dft_transform(simple_box_frequency)
    idft_shifted_box_signal = dft.inverse_dft_transform(shifted_box_frequency)

    x = list(range(N))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 6)
    fig.subplots_adjust(wspace=0.5, hspace=0.75)
    fig.set_figheight(9)
    fig.set_figwidth(14)

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

    ax1[4].set_ylabel('Signal')
    ax1[4].set_xlabel('Sample')
    ax1[4].set_title('Real Conjugate Cosine Signal')
    ax1[4].stem(x, [real(r) for r in conjugate_cos_frequency])

    ax1[5].set_ylabel('Signal')
    ax1[5].set_xlabel('Sample')
    ax1[5].set_title('Imag Conjugate Cosine Signal')
    ax1[5].stem(x, [imag(i) for i in conjugate_cos_frequency])

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

    ax2[4].set_ylabel('Signal')
    ax2[4].set_xlabel('Sample')
    ax2[4].set_title('RealConjugate Sin')
    ax2[4].stem(x, [real(r) for r in conjugate_sin_frequency])

    ax2[5].set_ylabel('Signal')
    ax2[5].set_xlabel('Sample')
    ax2[5].set_title('Imag Conjugate Sin')
    ax2[5].stem(x, [imag(i) for i in conjugate_sin_frequency])

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

    ax3[4].set_ylabel('Signal')
    ax3[4].set_xlabel('Sample')
    ax3[4].set_title('Conjugate Simple Box')
    ax3[4].stem(x, [real(r) for r in conjugate_simple_box])

    ax3[5].set_ylabel('Signal')
    ax3[5].set_xlabel('Sample')
    ax3[5].set_title('Imag Conjugate Cosine Signal')
    ax3[5].stem(x, [imag(i) for i in conjugate_simple_box])

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

    ax4[4].set_ylabel('Signal')
    ax4[4].set_xlabel('Sample')
    ax4[4].set_title('Conjugate Shifted')
    ax4[4].stem(x, [real(r) for r in conjugate_shifted_box])

    ax4[5].set_ylabel('Signal')
    ax4[5].set_xlabel('Sample')
    ax4[5].set_title('Imag Conjugate Shifted')
    ax4[5].stem(x, [imag(i) for i in conjugate_shifted_box])

    plt.show()
