import numpy as np
import scipy.io

def awgn_channel(symbols, Eb_N0_dB, M, R_code):
    k = int(np.log2(M))
    Eb_N0_linear = 10 ** (Eb_N0_dB / 10)
    Es = np.mean(np.abs(symbols) ** 2)
    noise_variance = Es / (2 * k * R_code  * Eb_N0_linear)
    noise = np.sqrt(noise_variance) * (np.random.randn(*symbols.shape) + 1j * np.random.randn(*symbols.shape))
    return symbols + noise, noise_variance

