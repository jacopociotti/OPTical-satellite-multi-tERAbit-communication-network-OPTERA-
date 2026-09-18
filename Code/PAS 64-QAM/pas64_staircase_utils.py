from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# Gray labeling usato dal modulatore/demodulatore 8-PAM interno alla 64-QAM:
#
# livello: -7  -5  -3  -1  +1  +3  +5  +7
# Gray:     000 001 011 010 110 111 101 100
#
# Separando il primo bit come "segno", i due bit di ampiezza sono:
# A=1 -> 10
# A=3 -> 11
# A=5 -> 01
# A=7 -> 00
_QAM_AMP_BITS = {
    1: (1, 0),
    3: (1, 1),
    5: (0, 1),
    7: (0, 0),
}

_INV_QAM_AMP_BITS = {
    bits: amplitude
    for amplitude, bits in _QAM_AMP_BITS.items()
}


@dataclass(frozen=True)
class PAS64StaircaseLayout:
    m: int
    r: int
    ccdm_length: int

    @property
    def block_bits(self) -> int:
        return self.m * self.m

    @property
    def info_bits_per_block(self) -> int:
        return self.m * (self.m - self.r)

    @property
    def parity_bits_per_block(self) -> int:
        return self.m * self.r

    @property
    def n_symbols_per_block(self) -> int:
        if self.block_bits % 6 != 0:
            raise ValueError(
                "m*m deve essere multiplo di 6 per la 64-QAM."
            )
        return self.block_bits // 6

    @property
    def n_axes_per_block(self) -> int:
        return 2 * self.n_symbols_per_block

    @property
    def n_amp_code_bits_per_block(self) -> int:
        # 2 bit di ampiezza per asse 8-PAM.
        return 2 * self.n_axes_per_block

    @property
    def n_extra_sign_info_bits_per_block(self) -> int:
        return (
            self.info_bits_per_block
            - self.n_amp_code_bits_per_block
        )

    @property
    def n_sign_bits_per_block(self) -> int:
        # Un bit di segno per asse.
        return self.n_axes_per_block

    @property
    def n_ccdm_per_block(self) -> int:
        if self.n_axes_per_block % self.ccdm_length != 0:
            raise ValueError(
                "La lunghezza CCDM deve dividere esattamente "
                "il numero di ampiezze per blocco Staircase."
            )
        return self.n_axes_per_block // self.ccdm_length

    def validate(self) -> None:
        if self.m <= 0:
            raise ValueError("m deve essere > 0.")
        if self.r <= 0 or self.r >= self.m:
            raise ValueError("Serve 0 < r < m.")
        if self.ccdm_length <= 0:
            raise ValueError("ccdm_length deve essere > 0.")

        # Forza le proprietà che includono controlli.
        _ = self.n_symbols_per_block
        _ = self.n_ccdm_per_block

        if self.n_extra_sign_info_bits_per_block < 0:
            raise ValueError(
                "Rate Staircase troppo basso per PAS 64-QAM: "
                "non ci sono abbastanza bit sistematici per "
                "contenere tutti i bit di ampiezza."
            )

        expected_sign_bits = (
            self.n_extra_sign_info_bits_per_block
            + self.parity_bits_per_block
        )

        if expected_sign_bits != self.n_sign_bits_per_block:
            raise ValueError(
                "La struttura non chiude sui bit di segno: "
                f"extra_info_sign + parity = {expected_sign_bits}, "
                f"ma servono {self.n_sign_bits_per_block} segni."
            )


def _normalize_amp_xor_mask(amp_xor_mask) -> np.ndarray:
    mask = np.asarray(amp_xor_mask, dtype=np.uint8).reshape(-1)

    if mask.size != 2:
        raise ValueError("amp_xor_mask deve contenere 2 bit.")
    if np.any(mask > 1):
        raise ValueError("amp_xor_mask deve contenere solo 0/1.")

    return mask


def amplitudes_to_pas64_code_amp_bits(
    amplitudes,
    amp_xor_mask=(1, 0),
) -> np.ndarray:
    """
    Ampiezze CCDM A in {1,3,5,7} -> bit SISTEMATICI del codice.

    La 64-QAM usa i Gray amplitude-label bits:
        A=1 -> 10
        A=3 -> 11
        A=5 -> 01
        A=7 -> 00

    Introduciamo un XOR opzionale:
        qam_amp_bits = code_amp_bits XOR amp_xor_mask

    Con il default amp_xor_mask=(1,0):
        A=1 -> code bits 00
        A=3 -> code bits 01
        A=5 -> code bits 11
        A=7 -> code bits 10

    Questo è utile nello Staircase perché i blocchi di terminazione
    hanno bit sistematici tutti zero: con il mask di default gli zeri
    vengono quindi trasmessi con ampiezza minima A=1, non A=7.
    """
    amplitudes = np.asarray(amplitudes).reshape(-1)
    mask = _normalize_amp_xor_mask(amp_xor_mask)

    qam_amp_bits = np.empty((amplitudes.size, 2), dtype=np.uint8)

    for i, amplitude_value in enumerate(amplitudes):
        amplitude = int(amplitude_value)

        if amplitude not in _QAM_AMP_BITS:
            raise ValueError(
                f"Ampiezza non valida: {amplitude}. "
                "Sono ammesse 1,3,5,7."
            )

        qam_amp_bits[i] = _QAM_AMP_BITS[amplitude]

    code_amp_bits = np.bitwise_xor(
        qam_amp_bits,
        mask[None, :],
    )

    return code_amp_bits.reshape(-1)


def pas64_code_amp_bits_to_amplitudes(
    code_amp_bits,
    amp_xor_mask=(1, 0),
) -> np.ndarray:
    """
    Operazione inversa di amplitudes_to_pas64_code_amp_bits().
    """
    bits = np.asarray(code_amp_bits, dtype=np.uint8).reshape(-1, 2)
    mask = _normalize_amp_xor_mask(amp_xor_mask)

    if np.any(bits > 1):
        raise ValueError("I bit devono essere 0/1.")

    qam_amp_bits = np.bitwise_xor(
        bits,
        mask[None, :],
    )

    amplitudes = np.empty(qam_amp_bits.shape[0], dtype=np.int8)

    for i, pair in enumerate(qam_amp_bits):
        key = (int(pair[0]), int(pair[1]))

        if key not in _INV_QAM_AMP_BITS:
            raise ValueError(f"Coppia di bit non valida: {key}.")

        amplitudes[i] = _INV_QAM_AMP_BITS[key]

    return amplitudes


def pas64_build_qam_bits(
    qam_amp_bits,
    sign_bits,
) -> np.ndarray:
    """
    Costruisce i bit Gray della 64-QAM nel formato:
        [I_sign, I_amp1, I_amp2, Q_sign, Q_amp1, Q_amp2, ...]
    """
    amp = np.asarray(qam_amp_bits, dtype=np.uint8).reshape(-1, 2)
    sign = np.asarray(sign_bits, dtype=np.uint8).reshape(-1)

    if np.any(amp > 1) or np.any(sign > 1):
        raise ValueError("I bit devono essere 0/1.")

    n_axes = sign.size

    if amp.shape[0] != n_axes:
        raise ValueError(
            "Servono 2 amplitude bits per ogni asse 8-PAM."
        )
    if n_axes % 2 != 0:
        raise ValueError("Il numero di assi deve essere pari.")

    axis_bits = np.empty((n_axes, 3), dtype=np.uint8)
    axis_bits[:, 0] = sign
    axis_bits[:, 1:3] = amp

    qam_bits = np.empty((n_axes // 2, 6), dtype=np.uint8)
    qam_bits[:, 0:3] = axis_bits[0::2]
    qam_bits[:, 3:6] = axis_bits[1::2]

    return qam_bits.reshape(-1)


def pas64_extract_qam_amp_sign_bits(
    qam_bits,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inversa di pas64_build_qam_bits().
    """
    qam = np.asarray(qam_bits, dtype=np.uint8).reshape(-1, 6)

    if np.any(qam > 1):
        raise ValueError("I bit devono essere 0/1.")

    n_symbols = qam.shape[0]
    axis_bits = np.empty((2 * n_symbols, 3), dtype=np.uint8)

    axis_bits[0::2] = qam[:, 0:3]
    axis_bits[1::2] = qam[:, 3:6]

    sign_bits = axis_bits[:, 0].copy()
    qam_amp_bits = axis_bits[:, 1:3].reshape(-1).copy()

    return qam_amp_bits, sign_bits


def pas64_extract_qam_amp_sign_llrs(
    qam_llrs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stessa operazione di estrazione, ma sugli LLR.
    """
    qam = np.asarray(qam_llrs).reshape(-1, 6)

    n_symbols = qam.shape[0]
    axis_llrs = np.empty((2 * n_symbols, 3), dtype=qam.dtype)

    axis_llrs[0::2] = qam[:, 0:3]
    axis_llrs[1::2] = qam[:, 3:6]

    sign_llrs = axis_llrs[:, 0].copy()
    amp_llrs = axis_llrs[:, 1:3].reshape(-1).copy()

    return amp_llrs, sign_llrs


def staircase_info_from_pas(
    amplitudes,
    sign_info_bits,
    m: int,
    r: int,
    amp_xor_mask=(1, 0),
) -> np.ndarray:
    """
    Costruisce i bit informativi di UN blocco Staircase:
        [code_amp_bits | sign_info_bits]

    L'ordinamento è quello che l'encoder Staircase poi reshapa in B_i,S di shape (m, m-r).
    """
    layout = PAS64StaircaseLayout(
        m=m,
        r=r,
        ccdm_length=len(np.asarray(amplitudes).reshape(-1)),
    )

    # Qui il ccdm_length fittizio serve solo a costruire l'oggetto;
    # non usiamo n_ccdm_per_block.
    if layout.block_bits % 6 != 0:
        raise ValueError("m*m deve essere multiplo di 6.")

    n_axes = 2 * (layout.block_bits // 6)
    n_amp_code_bits = 2 * n_axes
    n_info = m * (m - r)
    n_extra_sign = n_info - n_amp_code_bits

    amplitudes = np.asarray(amplitudes).reshape(-1)
    sign_info_bits = np.asarray(
        sign_info_bits,
        dtype=np.uint8,
    ).reshape(-1)

    if amplitudes.size != n_axes:
        raise ValueError(
            f"Servono {n_axes} ampiezze per blocco, "
            f"ricevute {amplitudes.size}."
        )

    if sign_info_bits.size != n_extra_sign:
        raise ValueError(
            f"Servono {n_extra_sign} sign-info bits, "
            f"ricevuti {sign_info_bits.size}."
        )

    code_amp_bits = amplitudes_to_pas64_code_amp_bits(
        amplitudes,
        amp_xor_mask=amp_xor_mask,
    )

    info_bits = np.concatenate(
        [code_amp_bits, sign_info_bits]
    ).astype(np.uint8, copy=False)

    if info_bits.size != n_info:
        raise RuntimeError("Dimensione info_bits PAS incoerente.")

    return info_bits


def staircase_info_to_pas(
    info_bits,
    m: int,
    r: int,
    amp_xor_mask=(1, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bit informativi decodificati di UN blocco Staircase
    -> ampiezze CCDM + sign-info bits.
    """
    info = np.asarray(info_bits, dtype=np.uint8).reshape(-1)

    n_info = m * (m - r)
    block_bits = m * m

    if block_bits % 6 != 0:
        raise ValueError("m*m deve essere multiplo di 6.")

    n_axes = 2 * (block_bits // 6)
    n_amp_code_bits = 2 * n_axes

    if info.size != n_info:
        raise ValueError(
            f"Servono {n_info} bit informativi, "
            f"ricevuti {info.size}."
        )

    code_amp_bits = info[:n_amp_code_bits]
    sign_info_bits = info[n_amp_code_bits:].copy()

    amplitudes = pas64_code_amp_bits_to_amplitudes(
        code_amp_bits,
        amp_xor_mask=amp_xor_mask,
    )

    return amplitudes, sign_info_bits


def staircase_blocks_to_qam_bits(
    encoded_blocks,
    m: int,
    r: int,
    amp_xor_mask=(1, 0),
) -> np.ndarray:
    """
    Mappa una sequenza di blocchi Staircase B_i nella disposizione
    richiesta dal modulatore 64-QAM PAS.

    Ogni B_i è:
        [B_i,S | B_i,R]

    Dai systematic bits:
        primi 2/3 dei bit del blocco -> amplitude code bits
        rimanenti                    -> sign-info bits

    Dai parity bits:
        -> restanti sign bits
    """
    blocks = np.asarray(encoded_blocks, dtype=np.uint8)

    if blocks.ndim == 2:
        blocks = blocks.reshape(1, m, m)

    if blocks.ndim != 3 or blocks.shape[1:] != (m, m):
        raise ValueError(
            f"encoded_blocks deve avere shape (B,{m},{m})."
        )

    block_bits = m * m
    if block_bits % 6 != 0:
        raise ValueError("m*m deve essere multiplo di 6.")

    n_symbols = block_bits // 6
    n_axes = 2 * n_symbols
    n_amp_code_bits = 2 * n_axes
    n_info = m * (m - r)
    n_extra_sign_info = n_info - n_amp_code_bits
    n_parity = m * r

    if n_extra_sign_info < 0:
        raise ValueError("Rate troppo basso per PAS 64-QAM.")
    if n_extra_sign_info + n_parity != n_axes:
        raise ValueError("Numero di sign bits non coerente.")

    mask = _normalize_amp_xor_mask(amp_xor_mask)
    qam_bits_out = np.empty(
        blocks.shape[0] * block_bits,
        dtype=np.uint8,
    )

    write = 0

    for block in blocks:
        info_flat = block[:, : m - r].reshape(-1)
        parity_flat = block[:, m - r :].reshape(-1)

        code_amp_bits = info_flat[:n_amp_code_bits]
        sign_info_bits = info_flat[n_amp_code_bits:]

        qam_amp_bits = np.bitwise_xor(
            code_amp_bits.reshape(-1, 2),
            mask[None, :],
        ).reshape(-1)

        sign_bits = np.concatenate(
            [sign_info_bits, parity_flat]
        )

        qam_bits_block = pas64_build_qam_bits(
            qam_amp_bits,
            sign_bits,
        )

        qam_bits_out[write : write + block_bits] = qam_bits_block
        write += block_bits

    return qam_bits_out


def qam_bits_to_staircase_blocks(
    qam_bits,
    num_blocks: int,
    m: int,
    r: int,
    amp_xor_mask=(1, 0),
) -> np.ndarray:
    """
    Operazione inversa di staircase_blocks_to_qam_bits().
    """
    bits = np.asarray(qam_bits, dtype=np.uint8).reshape(-1)

    block_bits = m * m
    expected = int(num_blocks) * block_bits

    if bits.size != expected:
        raise ValueError(
            f"Servono {expected} QAM bits, ricevuti {bits.size}."
        )

    n_symbols = block_bits // 6
    n_axes = 2 * n_symbols
    n_amp_code_bits = 2 * n_axes
    n_info = m * (m - r)
    n_extra_sign_info = n_info - n_amp_code_bits
    n_parity = m * r

    mask = _normalize_amp_xor_mask(amp_xor_mask)
    blocks = np.empty(
        (num_blocks, m, m),
        dtype=np.uint8,
    )

    for block_idx in range(num_blocks):
        start = block_idx * block_bits
        end = start + block_bits

        qam_amp_bits, sign_bits = (
            pas64_extract_qam_amp_sign_bits(bits[start:end])
        )

        code_amp_bits = np.bitwise_xor(
            qam_amp_bits.reshape(-1, 2),
            mask[None, :],
        ).reshape(-1)

        sign_info_bits = sign_bits[:n_extra_sign_info]
        parity_bits = sign_bits[n_extra_sign_info:]

        if parity_bits.size != n_parity:
            raise RuntimeError("Numero di parity bits incoerente.")

        info_flat = np.concatenate(
            [code_amp_bits, sign_info_bits]
        )

        block = blocks[block_idx]
        block[:, : m - r] = info_flat.reshape(m, m - r)
        block[:, m - r :] = parity_bits.reshape(m, r)

    return blocks


def qam_llrs_to_staircase_llrs(
    qam_llrs,
    num_blocks: int,
    m: int,
    r: int,
    amp_xor_mask=(1, 0),
    dtype=np.float32,
) -> np.ndarray:
    """
    Riordina gli LLR QAM PAS nella disposizione originale dei blocchi B_i,
    pronta per decoder.decode_sequence().

    Convenzione:
        LLR > 0 -> bit 0
        LLR < 0 -> bit 1

    Se qam_bit = code_bit XOR 1, allora:
        L_code = -L_qam
    """
    llrs = np.asarray(qam_llrs).reshape(-1)

    block_bits = m * m
    expected = int(num_blocks) * block_bits

    if llrs.size != expected:
        raise ValueError(
            f"Servono {expected} LLR, ricevuti {llrs.size}."
        )

    n_symbols = block_bits // 6
    n_axes = 2 * n_symbols
    n_amp_code_bits = 2 * n_axes
    n_info = m * (m - r)
    n_extra_sign_info = n_info - n_amp_code_bits
    n_parity = m * r

    mask = _normalize_amp_xor_mask(amp_xor_mask)
    mask_flat = np.tile(mask, n_axes)

    out_blocks = np.empty(
        (num_blocks, m, m),
        dtype=dtype,
    )

    for block_idx in range(num_blocks):
        start = block_idx * block_bits
        end = start + block_bits

        qam_amp_llrs, sign_llrs = (
            pas64_extract_qam_amp_sign_llrs(llrs[start:end])
        )

        code_amp_llrs = np.asarray(
            qam_amp_llrs,
            dtype=dtype,
        ).copy()

        # LLR flip per i bit attraversati da XOR=1.
        code_amp_llrs[mask_flat == 1] *= -1.0

        sign_llrs = np.asarray(sign_llrs, dtype=dtype)

        sign_info_llrs = sign_llrs[:n_extra_sign_info]
        parity_llrs = sign_llrs[n_extra_sign_info:]

        if parity_llrs.size != n_parity:
            raise RuntimeError("Numero di parity LLR incoerente.")

        info_llrs = np.concatenate(
            [code_amp_llrs, sign_info_llrs]
        )

        block = out_blocks[block_idx]
        block[:, : m - r] = info_llrs.reshape(m, m - r)
        block[:, m - r :] = parity_llrs.reshape(m, r)

    return out_blocks.reshape(-1)


def pas64_qam_amp_bits_from_code_amp_bits(
    code_amp_bits,
    amp_xor_mask=(1, 0),
) -> np.ndarray:
    """
    Helper: code amplitude bits -> QAM Gray amplitude-label bits.
    """
    bits = np.asarray(code_amp_bits, dtype=np.uint8).reshape(-1, 2)
    mask = _normalize_amp_xor_mask(amp_xor_mask)

    return np.bitwise_xor(bits, mask[None, :]).reshape(-1)


def shaped_8pam_second_moment(composition) -> float:
    """
    E[A^2] per A in {1,3,5,7} con probabilità date dalla composizione.
    """
    composition = np.asarray(composition, dtype=np.float64).reshape(-1)

    if composition.size != 4:
        raise ValueError("composition deve avere 4 elementi.")
    if np.any(composition < 0) or np.sum(composition) <= 0:
        raise ValueError("Composizione non valida.")

    p = composition / np.sum(composition)
    a = np.array([1.0, 3.0, 5.0, 7.0])

    return float(np.sum(p * a * a))


def shaped_64qam_second_moment(composition) -> float:
    """
    E[|X|^2] non normalizzato = 2 E[A^2].
    """
    return 2.0 * shaped_8pam_second_moment(composition)


if __name__ == "__main__":
    m = 720
    r = 96
    composition = (240, 120, 80, 40)

    layout = PAS64StaircaseLayout(
        m=m,
        r=r,
        ccdm_length=sum(composition),
    )
    layout.validate()

    print("block_bits =", layout.block_bits)
    print("n_symbols_per_block =", layout.n_symbols_per_block)
    print("n_axes_per_block =", layout.n_axes_per_block)
    print(
        "n_amp_code_bits_per_block =",
        layout.n_amp_code_bits_per_block,
    )
    print(
        "n_extra_sign_info_bits_per_block =",
        layout.n_extra_sign_info_bits_per_block,
    )
    print(
        "parity_bits_per_block =",
        layout.parity_bits_per_block,
    )
    print("n_ccdm_per_block =", layout.n_ccdm_per_block)
    print(
        "E[A^2] =",
        shaped_8pam_second_moment(composition),
    )
    print(
        "E[|X|^2] shaped non-normalizzato =",
        shaped_64qam_second_moment(composition),
    )
