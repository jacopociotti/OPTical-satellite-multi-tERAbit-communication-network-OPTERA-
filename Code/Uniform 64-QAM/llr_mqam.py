import numpy as np


def qam_soft_demap(received_symbols, M, noise_var, batch_size=200_000):

    y = np.asarray(received_symbols)

    k = int(np.log2(M))
    sqrtM = int(np.sqrt(M))
    k_axis = k // 2

    if sqrtM * sqrtM != M or 2 * k_axis != k:
        raise ValueError("M deve essere una QAM quadrata.")

    # Costellazione normalizzata: Es = 1
    norm_factor = np.float32(np.sqrt((2.0 / 3.0) * (M - 1)))

    levels = (np.arange(-(sqrtM - 1), sqrtM, 2, dtype=np.float32) / norm_factor)

    # Gray labeling degli sqrt(M) livelli PAM
    idx = np.arange(sqrtM, dtype=np.int32)
    gray = idx ^ (idx >> 1)

    shifts = np.arange(k_axis - 1, -1, -1, dtype=np.int32)

    level_bits = ((gray[:, None] >> shifts) & 1).astype(np.uint8)

    # float32: metà memoria rispetto alla versione originale
    llrs = np.empty((len(y), k), dtype=np.float32)

    scale = np.float32(1.0 / (2.0 * noise_var))

    for start in range(0, len(y), batch_size):

        end = min(start + batch_size, len(y))

        I = np.asarray(y.real[start:end], dtype=np.float32)
        Q = np.asarray(y.imag[start:end], dtype=np.float32)

        for b in range(k_axis):

            levels_0 = levels[level_bits[:, b] == 0]
            levels_1 = levels[level_bits[:, b] == 1]

            # ---------------- I ----------------

            dist_I_0 = np.min((I[:, None] - levels_0[None, :]) ** 2, axis=1)

            dist_I_1 = np.min((I[:, None] - levels_1[None, :]) ** 2, axis=1)

            llrs[start:end, b] = (dist_I_1 - dist_I_0) * scale

            # ---------------- Q ----------------

            dist_Q_0 = np.min((Q[:, None] - levels_0[None, :]) ** 2, axis=1)

            dist_Q_1 = np.min((Q[:, None] - levels_1[None, :]) ** 2, axis=1)

            llrs[start:end, k_axis + b] = (dist_Q_1 - dist_Q_0) * scale

    return llrs.reshape(-1)



def qam_soft_demap_fso(received_symbols, h, M, noise_var, batch_size=200_000):

    y = np.asarray(received_symbols)
    h = np.asarray(h)

    k = int(np.log2(M))
    sqrtM = int(np.sqrt(M))
    k_axis = k // 2

    if sqrtM * sqrtM != M or 2 * k_axis != k:
        raise ValueError("M deve essere una QAM quadrata.")

    # Costellazione normalizzata: Es = 1
    norm_factor = np.float32(np.sqrt((2.0 / 3.0) * (M - 1)))

    levels = (np.arange(-(sqrtM - 1), sqrtM, 2, dtype=np.float32) / norm_factor)
 
    # Gray labeling dei livelli PAM
    idx = np.arange(sqrtM, dtype=np.int32)

    gray = idx ^ (idx >> 1)

    shifts = np.arange(k_axis - 1, -1, -1, dtype=np.int32)

    level_bits = ((gray[:, None] >> shifts) & 1).astype(np.uint8)

    llrs = np.empty((len(y), k), dtype=np.float32)

    for start in range(0, len(y), batch_size):

        end = min(start + batch_size, len(y))

        # Batch
        y_batch = y[start:end]
        h_batch = h[start:end]

        # Matched filtering:
        # r = h* y
        r = np.conj(h_batch) * y_batch

        # g = |h|^2
        g = np.abs(h_batch) ** 2

        # Protezione numerica nei deep fade
        g = np.maximum(g, 1e-12)

        I = np.asarray(r.real, dtype=np.float32)

        Q = np.asarray(r.imag, dtype=np.float32)

        g = np.asarray(g, dtype=np.float32)

        # Dopo h*:
        #
        # r = g*x + n'
        #
        # quindi la costellazione PAM è scalata di g.

        for b in range(k_axis):

            levels_0 = levels[level_bits[:, b] == 0]

            levels_1 = levels[level_bits[:, b] == 1]

            # ==========================
            # ASSE I
            # ==========================

            dist_I_0 = np.min((I[:, None] - g[:, None] * levels_0[None, :]) ** 2, axis=1)

            dist_I_1 = np.min((I[:, None] - g[:, None] * levels_1[None, :]) ** 2, axis=1)

            llrs[start:end, b] = (dist_I_1 - dist_I_0) / ( 2.0 * noise_var * g)

            # ==========================
            # ASSE Q
            # ==========================

            dist_Q_0 = np.min((Q[:, None] - g[:, None] * levels_0[None, :]) ** 2, axis=1)

            dist_Q_1 = np.min((Q[:, None] - g[:, None] * levels_1[None, :]) ** 2, axis=1)

            llrs[start:end, k_axis + b] = (dist_Q_1 - dist_Q_0) / (2.0 * noise_var * g)

    return llrs.reshape(-1)