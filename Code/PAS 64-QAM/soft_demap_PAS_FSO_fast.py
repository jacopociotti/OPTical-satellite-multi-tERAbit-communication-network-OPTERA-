# PAS-AWARE SOFT DEMAPPER PER FSO
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

def _gray_pam_levels_and_bits(
    M: int,
):
    """
    Stessi livelli e stesso Gray labeling di qam_mod_opt().
    """

    M = int(M)

    k = int(np.log2(M))
    sqrtM = int(np.sqrt(M))
    k_axis = k // 2

    if (
        sqrtM * sqrtM != M
        or 2 * k_axis != k
    ):
        raise ValueError(
            "M deve essere una QAM quadrata."
        )

    norm_factor = np.sqrt(
        (2.0 / 3.0) * (M - 1)
    )

    levels = (
        np.arange(
            -(sqrtM - 1),
            sqrtM,
            2,
            dtype=np.float64,
        )
        / norm_factor
    )

    idx = np.arange(
        sqrtM,
        dtype=np.int64,
    )

    gray = idx ^ (idx >> 1)

    shifts = np.arange(
        k_axis - 1,
        -1,
        -1,
        dtype=np.int64,
    )

    level_bits = (
        (
            gray[:, None]
            >> shifts
        )
        & 1
    ).astype(np.uint8)

    return (
        levels,
        level_bits,
    )


def _level_log_priors_64qam_pas(
    levels,
    composition,
):
    """
    Prior dei livelli signed 8-PAM:

        P(+A) = P(-A) = P_A(A)/2.
    """

    composition = np.asarray(
        composition,
        dtype=np.float64,
    ).reshape(-1)

    if composition.size != 4:
        raise ValueError(
            "COMPOSITION deve contenere "
            "[n1,n3,n5,n7]."
        )

    if (
        np.any(composition <= 0.0)
        or np.sum(composition) <= 0.0
    ):
        raise ValueError(
            "Composizione non valida."
        )

    p_amp = (
        composition
        / np.sum(composition)
    )

    amplitudes = np.array(
        [1, 3, 5, 7],
        dtype=np.int64,
    )

    p_dict = {
        int(a): float(p)
        for a, p in zip(
            amplitudes,
            p_amp,
        )
    }

    sqrtM = len(levels)

    signed_integer_levels = np.arange(
        -(sqrtM - 1),
        sqrtM,
        2,
        dtype=np.int64,
    )

    priors = np.empty(
        sqrtM,
        dtype=np.float64,
    )

    for idx, a_signed in enumerate(
        signed_integer_levels
    ):
        priors[idx] = (
            0.5
            * p_dict[
                abs(int(a_signed))
            ]
        )

    return np.log(priors)


def qam_soft_demap_pas_fso(
    received_symbols,
    h,
    M: int,
    noise_var: float,
    composition=(240, 120, 80, 40),
    batch_size: int = 100_000,
    dtype=np.float32,
):
    """
    Log-MAP esatto PAS-aware per:

        y = h*x + n

    con perfect CSI.

    Per ogni simbolo ruotiamo la fase nota del canale:

        z = y * conj(h)/|h|
          = |h|*x + n'

    con n' avente la stessa varianza per componente reale.

    Poi, separatamente sui due assi 8-PAM:

        metric(a)
          = log P_Xaxis(a)
            - (z_axis - |h|*a)^2
              / (2*sigma^2)

    Output:
        L = log P(bit=0|y,h)
            - log P(bit=1|y,h)

    quindi:
        L > 0 -> bit 0
        L < 0 -> bit 1.
    """

    y = np.asarray(
        received_symbols
    ).reshape(-1)

    h = np.asarray(
        h
    ).reshape(-1)

    if y.size != h.size:
        raise ValueError(
            "received_symbols e h devono "
            "avere la stessa lunghezza."
        )

    if y.size == 0:
        return np.empty(
            0,
            dtype=dtype,
        )

    noise_var = float(
        noise_var
    )

    if (
        not np.isfinite(noise_var)
        or noise_var <= 0.0
    ):
        raise ValueError(
            "noise_var deve essere "
            "finita e > 0."
        )

    M = int(M)

    if M != 64:
        raise ValueError(
            "Questa funzione PAS e' "
            "configurata per 64-QAM."
        )

    k = int(np.log2(M))
    k_axis = k // 2

    levels, level_bits = (
        _gray_pam_levels_and_bits(M)
    )

    log_priors = (
        _level_log_priors_64qam_pas(
            levels,
            composition,
        )
    )

    masks0 = [
        level_bits[:, b] == 0
        for b in range(k_axis)
    ]

    masks1 = [
        level_bits[:, b] == 1
        for b in range(k_axis)
    ]

    llrs = np.empty(
        (y.size, k),
        dtype=dtype,
    )

    inv_2sigma2 = (
        1.0
        / (2.0 * noise_var)
    )

    tiny = np.finfo(
        np.float64
    ).tiny

    for start in range(
        0,
        y.size,
        int(batch_size),
    ):

        end = min(
            start + int(batch_size),
            y.size,
        )

        yb = y[start:end]
        hb = h[start:end]

        gain = np.abs(
            hb
        ).astype(
            np.float64,
            copy=False,
        )

        # Per gain -> 0 scegliamo fase unitaria 1.
        # In quel caso tutti i candidate points
        # collassano comunque nell'origine.
        phase_corrector = np.ones(
            hb.size,
            dtype=np.complex128,
        )

        nonzero = gain > tiny

        phase_corrector[nonzero] = (
            np.conj(
                hb[nonzero]
            )
            / gain[nonzero]
        )

        z = (
            yb
            * phase_corrector
        )

        I = np.asarray(
            z.real,
            dtype=np.float64,
        )

        Q = np.asarray(
            z.imag,
            dtype=np.float64,
        )

        candidates = (
            gain[:, None]
            * levels[None, :]
        )

        metric_I = (
            log_priors[None, :]
            - (
                I[:, None]
                - candidates
            ) ** 2
            * inv_2sigma2
        )

        metric_Q = (
            log_priors[None, :]
            - (
                Q[:, None]
                - candidates
            ) ** 2
            * inv_2sigma2
        )

        for b in range(k_axis):

            llrs[start:end, b] = (
                logsumexp(
                    metric_I[
                        :,
                        masks0[b],
                    ],
                    axis=1,
                )
                - logsumexp(
                    metric_I[
                        :,
                        masks1[b],
                    ],
                    axis=1,
                )
            )

            llrs[
                start:end,
                k_axis + b,
            ] = (
                logsumexp(
                    metric_Q[
                        :,
                        masks0[b],
                    ],
                    axis=1,
                )
                - logsumexp(
                    metric_Q[
                        :,
                        masks1[b],
                    ],
                    axis=1,
                )
            )

    return llrs.reshape(-1)