from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def _gray_pam_levels_and_bits(M: int):
    """
    Restituisce livelli PAM normalizzati e relative label Gray.

    Per M=64:
        sqrt(M)=8 livelli per asse
        3 bit per asse

    L'ordinamento e la normalizzazione sono identici a qam_mod_opt().
    """
    M = int(M)
    k = int(np.log2(M))
    sqrtM = int(np.sqrt(M))
    k_axis = k // 2

    if sqrtM * sqrtM != M or 2 * k_axis != k:
        raise ValueError("M deve essere una QAM quadrata con un numero pari di bit/simbolo.")

    norm_factor = np.sqrt((2.0 / 3.0) * (M - 1))
    levels = np.arange(-(sqrtM - 1), sqrtM, 2, dtype=np.float64) / norm_factor

    # qam_mod_opt() usa come label binaria il valore Gray associato
    # all'indice crescente del livello PAM.
    idx = np.arange(sqrtM, dtype=np.int64)
    gray = idx ^ (idx >> 1)
    shifts = np.arange(k_axis - 1, -1, -1, dtype=np.int64)
    level_bits = ((gray[:, None] >> shifts) & 1).astype(np.uint8)

    return levels, level_bits


def _level_log_priors_64qam_pas(levels, composition):
    """
    Prior per i livelli signed PAM.

    composition = [n1, n3, n5, n7] definisce P_A(A).
    Assumiamo segni uniformi, quindi P_Xaxis(+A)=P_Xaxis(-A)=P_A(A)/2.

    Il fattore 1/2 è comune a tutti i livelli e potrebbe essere omesso;
    lo manteniamo per chiarezza.
    """
    composition = np.asarray(composition, dtype=np.float64).reshape(-1)

    if composition.size != 4:
        raise ValueError("Per PAS 64-QAM la composizione deve avere 4 elementi [n1,n3,n5,n7].")
    if np.any(composition < 0.0) or np.sum(composition) <= 0.0:
        raise ValueError("Composizione non valida.")

    p_amp = composition / np.sum(composition)
    amplitudes = np.array([1, 3, 5, 7], dtype=np.int64)
    p_dict = {int(a): float(p) for a, p in zip(amplitudes, p_amp)}

    # levels sono normalizzati; recuperiamo l'ampiezza non normalizzata
    # dalla loro posizione nell'8-PAM: [-7,-5,-3,-1,+1,+3,+5,+7].
    sqrtM = len(levels)
    signed_integer_levels = np.arange(-(sqrtM - 1), sqrtM, 2, dtype=np.int64)

    priors = np.empty(sqrtM, dtype=np.float64)
    for i, a_signed in enumerate(signed_integer_levels):
        priors[i] = 0.5 * p_dict[abs(int(a_signed))]

    if np.any(priors <= 0.0):
        raise ValueError(
            "Tutte le ampiezze devono avere probabilita' positiva nel demapper corrente."
        )

    return np.log(priors)


def qam_soft_demap_pas_awgn(
    received_symbols,
    M: int,
    noise_var: float,
    composition=(240, 120, 80, 40),
    batch_size: int = 200_000,
    dtype=np.float32,
):
    """
    Soft demapper Log-MAP esatto per square M-QAM PAS su AWGN complesso.

    Convenzioni:
        y = x + n
        n = n_I + j n_Q
        Var(n_I) = Var(n_Q) = noise_var

    Per ciascun asse PAM usa la metrica:
        log P_Xaxis(x) - (y_axis - x)^2 / (2*noise_var)

    con prior di ampiezza ricavato dalla composizione CCDM e segni uniformi.

    Output LLR:
        L = log P(bit=0 | y) - log P(bit=1 | y)

    quindi:
        L > 0 -> hard bit 0
        L < 0 -> hard bit 1

    L'ordine degli LLR coincide con qam_mod_opt():
        [I_b0, I_b1, I_b2, Q_b0, Q_b1, Q_b2, ...]

    Il calcolo e' separato in due 8-PAM: non costruisce mai una matrice
    Nsimboli x 64 e quindi e' adatto a sequenze Staircase molto lunghe.
    """
    y = np.asarray(received_symbols).reshape(-1)
    M = int(M)
    noise_var = float(noise_var)
    batch_size = int(batch_size)

    if y.size == 0:
        return np.empty(0, dtype=dtype)
    if noise_var <= 0.0 or not np.isfinite(noise_var):
        raise ValueError("noise_var deve essere finita e > 0.")
    if batch_size <= 0:
        raise ValueError("batch_size deve essere > 0.")

    k = int(np.log2(M))
    k_axis = k // 2

    levels, level_bits = _gray_pam_levels_and_bits(M)

    if M != 64:
        raise ValueError(
            "Questa versione PAS usa composition=[n1,n3,n5,n7] ed e' progettata per M=64."
        )

    log_priors = _level_log_priors_64qam_pas(levels, composition)

    masks0 = [level_bits[:, b] == 0 for b in range(k_axis)]
    masks1 = [level_bits[:, b] == 1 for b in range(k_axis)]

    llrs = np.empty((y.size, k), dtype=dtype)
    inv_2sigma2 = 1.0 / (2.0 * noise_var)

    for start in range(0, y.size, batch_size):
        end = min(start + batch_size, y.size)
        yb = y[start:end]

        I = np.asarray(yb.real, dtype=np.float64)
        Q = np.asarray(yb.imag, dtype=np.float64)

        metric_I = (
            log_priors[None, :]
            - (I[:, None] - levels[None, :]) ** 2 * inv_2sigma2
        )
        metric_Q = (
            log_priors[None, :]
            - (Q[:, None] - levels[None, :]) ** 2 * inv_2sigma2
        )

        for b in range(k_axis):
            llrs[start:end, b] = (
                logsumexp(metric_I[:, masks0[b]], axis=1)
                - logsumexp(metric_I[:, masks1[b]], axis=1)
            )
            llrs[start:end, k_axis + b] = (
                logsumexp(metric_Q[:, masks0[b]], axis=1)
                - logsumexp(metric_Q[:, masks1[b]], axis=1)
            )

    return llrs.reshape(-1)


if __name__ == "__main__":
    # Sanity check minimale sui parametri del setup corrente.
    levels, bits = _gray_pam_levels_and_bits(64)
    print("8-PAM levels:", levels)
    print("Gray labels:\n", bits)
    print(
        "P_A =",
        np.asarray((240, 120, 80, 40), dtype=float) / 480.0,
    )
