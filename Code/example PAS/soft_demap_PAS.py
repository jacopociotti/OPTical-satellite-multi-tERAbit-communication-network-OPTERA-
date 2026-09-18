import numpy as np
from scipy.special import logsumexp
from demod_MQAM_optimized import qam_demod_opt

def qam_soft_demap(received_symbols, M, noise_var, composition):
    """
    Calcola LLR tenendo conto delle probabilità non uniformi
    delle ampiezze A ∈ {1,3,5,7}.
    Utilizza l'approssimazione Max-Log: (distanza minima bit=1) - (distanza minima bit=0) / N0.
    """
    k = int(np.log2(M))
    sqrtM = int(np.sqrt(M))
    norm_factor = np.sqrt((2/3)*(M-1))
    levels = np.arange(-(sqrtM-1), sqrtM, 2) / norm_factor
    
    # Crea i 64 simboli QAM ideali di riferimento
    I_grid, Q_grid = np.meshgrid(levels, levels)
    ideal_symbols = I_grid.flatten() + 1j * Q_grid.flatten() # es: -7-7j, -5-5j ecc...
    
    # Mappa quali bit (0 o 1) compongono ogni simbolo QAM ideale:  simbolo QAM di riferimento -> simbolo binario es: -7-7j -> 000000, -5-5j -> 001000
    ideal_bits = qam_demod_opt(ideal_symbols, M).reshape(M, k)


    composition = np.array(composition, dtype=float)
    p_amp = composition / np.sum(composition)  # Probabilità delle ampiezze A ∈ {1,3,5,7}

    p_dict = {
        1: p_amp[0],
        3: p_amp[1],
        5: p_amp[2],
        7: p_amp[3],
    }


    # Probabilità a priori dei simboli QAM:
    # P(X) = P(A_I) * P(A_Q) * P(S_I) * P(S_Q)
    # i segni sono uniformi, quindi il fattore 1/4 è comune
    # e può anche essere omesso negli LLR.
    symbol_priors = []

    for x in ideal_symbols:
        A_I = int(round(abs(np.real(x) * norm_factor)))
        A_Q = int(round(abs(np.imag(x) * norm_factor)))

        p_x = p_dict[A_I] * p_dict[A_Q]
        symbol_priors.append(p_x)

    symbol_priors = np.asarray(symbol_priors, dtype=float)
    log_priors = np.log(symbol_priors + 1e-300)


    received_symbols = np.asarray(received_symbols)
    llrs = np.zeros((len(received_symbols), k), dtype=float)
    
    # Matrice distanze: righe = simboli ricevuti, colonne = punti costellazione
    dist = np.abs(received_symbols[:, None] - ideal_symbols[None, :]) ** 2

    # Metrica logaritmica:
    # log P_X(x) - |y-x|^2/(2*sigma^2)
    metric = log_priors[None, :] - dist / (2 * noise_var)

    for bit_idx in range(k):
        mask0 = ideal_bits[:, bit_idx] == 0
        mask1 = ideal_bits[:, bit_idx] == 1

        logp0 = logsumexp(metric[:, mask0], axis=1)
        logp1 = logsumexp(metric[:, mask1], axis=1)

        llrs[:, bit_idx] = logp0 - logp1

    return llrs.flatten()