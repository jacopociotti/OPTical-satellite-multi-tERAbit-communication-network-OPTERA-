import numpy as np
import scipy.io

def qam_demod_opt(received_symbols, M):
    k = int(np.log2(M))
    k_axis = k // 2
    sqrtM = int(np.sqrt(M))
    norm_factor = np.sqrt((2/3)*(M-1)) # fattore di normalizzazione per avere potenza unitaria
    levels = np.arange(-(sqrtM-1), sqrtM, 2) / norm_factor # livelli di modulazione normalizzati

    demodulated_bits = np.zeros((len(received_symbols), k), dtype=int)

    vect = np.arange(sqrtM)
    gray_constellation = np.bitwise_xor(vect, np.floor(vect/2).astype(int))

    for i in range(len(received_symbols)):
        I = np.real(received_symbols[i])
        Q = np.imag(received_symbols[i])

        idx_I = np.argmin(np.abs(I - levels))
        idx_Q = np.argmin(np.abs(Q - levels))

        gray_val_I = gray_constellation[idx_I]
        gray_val_Q = gray_constellation[idx_Q]

        bits_I = np.array(list(np.binary_repr(gray_val_I, width=k_axis)), dtype=int)
        bits_Q = np.array(list(np.binary_repr(gray_val_Q, width=k_axis)), dtype=int)

        demodulated_bits[i, :k_axis] = bits_I
        demodulated_bits[i, k_axis:] = bits_Q

    return demodulated_bits.flatten()