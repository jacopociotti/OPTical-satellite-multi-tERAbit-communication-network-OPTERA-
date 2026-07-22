import numpy as np


def amplitudes_to_pas64_amp_bits(amplitudes):
    """
    Converte A ∈ {1,3,5,7} nei 2 bit di ampiezza
    coerenti con la mappatura Gray del modulatore 64-QAM.

    A = 1  -> 10
    A = 3  -> 11
    A = 5  -> 01
    A = 7  -> 00
    """

    amplitudes = np.asarray(amplitudes, dtype=int).flatten()

    mapping = {
        1: [1, 0],
        3: [1, 1],
        5: [0, 1],
        7: [0, 0],
    }

    bits = []

    for A in amplitudes:
        if A not in mapping:
            raise ValueError(f"Ampiezza non valida: {A}")
        bits.extend(mapping[A])

    return np.array(bits, dtype=int)


def pas64_amp_bits_to_amplitudes(amp_bits):
    """
    Operazione inversa:
    2 bit di ampiezza -> A ∈ {1,3,5,7}
    """

    amp_bits = np.asarray(amp_bits, dtype=int).reshape(-1, 2)

    inv_mapping = {
        (1, 0): 1,
        (1, 1): 3,
        (0, 1): 5,
        (0, 0): 7,
    }

    amplitudes = []

    for b1, b2 in amp_bits:
        key = (int(b1), int(b2))

        if key not in inv_mapping:
            raise ValueError(f"Coppia di bit non valida: {key}")

        amplitudes.append(inv_mapping[key])

    return np.array(amplitudes, dtype=int)


def pas64_build_qam_bits(amp_bits, sign_bits):
    """
    Costruisce i bit Gray della 64-QAM a partire da:
    - amp_bits: 2 bit per asse 8-PAM
    - sign_bits: 1 bit per asse 8-PAM

    Ordine assi:
    asse 0 -> I del simbolo 0
    asse 1 -> Q del simbolo 0
    asse 2 -> I del simbolo 1
    asse 3 -> Q del simbolo 1
    ecc.
    es. 2 assi (I0, Q0) -> 1 simbolo QAM
        4 asssi -> 2 simboli QAM
    ecc.

    Output:
    qam_bits flattenati nel formato richiesto da qam_mod_opt:
    [I0_sign, I0_amp1, I0_amp2, Q0_sign, Q0_amp1, Q0_amp2, ...]
    """

    amp_bits = np.asarray(amp_bits, dtype=int).reshape(-1, 2)
    sign_bits = np.asarray(sign_bits, dtype=int).reshape(-1)

    n_axes = len(sign_bits)

    if amp_bits.shape[0] != n_axes:
        raise ValueError("amp_bits deve avere 2 bit per ogni asse 8-PAM.")

    if n_axes % 2 != 0:
        raise ValueError("Il numero di assi deve essere pari: I e Q per ogni simbolo.")

    axis_bits = np.column_stack([sign_bits, amp_bits])

    n_symbols = n_axes // 2
    qam_bits = np.zeros((n_symbols, 6), dtype=int)

    qam_bits[:, 0:3] = axis_bits[0::2]  # I , prende un elemento ogni due
    qam_bits[:, 3:6] = axis_bits[1::2]  # Q

    return qam_bits.flatten()


def pas64_extract_amp_sign_bits(qam_bits):
    """
    Operazione inversa di pas64_build_qam_bits.

    Input:
    qam_bits nel formato:
    [I_sign, I_amp1, I_amp2, Q_sign, Q_amp1, Q_amp2, ...]

    Output:
    - amp_bits flattenati
    - sign_bits
    """

    qam_bits = np.asarray(qam_bits).reshape(-1, 6)

    n_symbols = qam_bits.shape[0]
    axis_bits = np.zeros((2 * n_symbols, 3), dtype=qam_bits.dtype)

    axis_bits[0::2] = qam_bits[:, 0:3]  # I
    axis_bits[1::2] = qam_bits[:, 3:6]  # Q

    sign_bits = axis_bits[:, 0]
    amp_bits = axis_bits[:, 1:3].reshape(-1)

    return amp_bits, sign_bits


def pas64_codeword_to_qam_bits(codeword, K):
    """
    Riordina una codeword LDPC sistematica in bit 64-QAM PAS.

    Assunzione:
    codeword = [bit sistematici | bit di parità]

    I primi 2N/3 bit sono ampiezze.
    I rimanenti bit sistematici, se presenti, vanno nei segni.
    I bit di parità vanno nei segni.
    """

    codeword = np.asarray(codeword, dtype=int).flatten()
    N = len(codeword)

    if N % 6 != 0:
        raise ValueError("Per 64-QAM, la lunghezza N della codeword deve essere multipla di 6.")

    n_symbols = N // 6
    n_amp_bits = 4 * n_symbols
    n_sign_bits = 2 * n_symbols

    if K < n_amp_bits:
        raise ValueError(
            "Il rate LDPC è troppo basso per PAS 64-QAM senza puncturing: serve K >= 2N/3."
        )

    amp_bits = codeword[:n_amp_bits]

    extra_sign_info_bits = codeword[n_amp_bits:K]
    parity_bits = codeword[K:]

    sign_bits = np.concatenate([extra_sign_info_bits, parity_bits])

    if len(sign_bits) != n_sign_bits:
        raise ValueError("Numero di bit di segno non coerente.")

    return pas64_build_qam_bits(amp_bits, sign_bits)


def pas64_qam_bits_to_codeword(qam_bits, N, K):
    """
    Operazione inversa:
    dai bit demappati dalla 64-QAM ricostruisce l'ordine della codeword LDPC.
    """

    amp_bits, sign_bits = pas64_extract_amp_sign_bits(qam_bits)

    n_symbols = N // 6
    n_amp_bits = 4 * n_symbols
    n_extra_sign_info_bits = K - n_amp_bits

    codeword = np.zeros(N, dtype=int)

    codeword[:n_amp_bits] = amp_bits
    codeword[n_amp_bits:K] = sign_bits[:n_extra_sign_info_bits]
    codeword[K:] = sign_bits[n_extra_sign_info_bits:]

    return codeword


def pas64_qam_llrs_to_codeword_order(qam_llrs, N, K):
    """
    Riordina gli LLR dalla disposizione 64-QAM:
        [sI, aI1, aI2, sQ, aQ1, aQ2]

    alla disposizione LDPC:
        [amp_bits, extra_info_bits, parity_bits]

    Mantiene dtype float.
    """

    qam_llrs = np.asarray(qam_llrs, dtype=float).reshape(-1, 6)

    if N % 6 != 0:
        raise ValueError("N deve essere multiplo di 6 per 64-QAM.")

    n_symbols = N // 6
    n_axes = 2 * n_symbols
    n_amp_bits = 4 * n_symbols
    n_extra_info_bits = K - n_amp_bits

    if qam_llrs.shape[0] != n_symbols:
        raise ValueError("Numero di simboli QAM non coerente con N.")

    axis_llrs = np.zeros((n_axes, 3), dtype=float)

    # Asse I
    axis_llrs[0::2, :] = qam_llrs[:, 0:3]

    # Asse Q
    axis_llrs[1::2, :] = qam_llrs[:, 3:6]

    sign_llrs = axis_llrs[:, 0]
    amp_llrs = axis_llrs[:, 1:3].reshape(-1)

    llr_codeword = np.zeros(N, dtype=float)

    llr_codeword[:n_amp_bits] = amp_llrs
    llr_codeword[n_amp_bits:K] = sign_llrs[:n_extra_info_bits]
    llr_codeword[K:] = sign_llrs[n_extra_info_bits:]

    return llr_codeword
