from __future__ import annotations

from functools import lru_cache
import math
import numpy as np


def multinomial_count(counts) -> int:
    """
    Numero di sequenze distinte con composizione fissata.

    Usa un prodotto di coefficienti binomiali:
        n! / prod_i n_i!
    evitando di calcolare più fattoriali enormi.
    """
    counts = tuple(int(c) for c in counts)

    if len(counts) == 0:
        raise ValueError("counts non può essere vuoto.")
    if any(c < 0 for c in counts):
        raise ValueError("Tutti i conteggi devono essere >= 0.")

    out = 1
    placed = 0

    for c in counts:
        out *= math.comb(placed + c, c)
        placed += c

    return out


def bits_to_int(bits) -> int:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)

    if np.any(bits > 1):
        raise ValueError("I bit devono essere 0/1.")

    value = 0
    for b in bits:
        value = (value << 1) | int(b)

    return value


def int_to_bits(value: int, k: int) -> np.ndarray:
    value = int(value)
    k = int(k)

    if k < 0:
        raise ValueError("k deve essere >= 0.")
    if value < 0 or value >= (1 << k):
        raise ValueError("value fuori dal range rappresentabile con k bit.")

    return np.fromiter(
        ((value >> i) & 1 for i in range(k - 1, -1, -1)),
        dtype=np.uint8,
        count=k,
    )


class CCDM4ary:
    """
    CCDM quaternario tramite enumerative ranking/unranking.

    Rispetto alla prima versione, NON ricalcola un coefficiente
    multinomiale tramite factorial ad ogni candidato.

    Se nello stato corrente:
        M = n! / prod_i n_i!

    il numero di sequenze che iniziano con il simbolo i è:
        M_i = M * n_i // n

    Questo permette di aggiornare ricorsivamente il numero di sequenze.
    """

    def __init__(
        self,
        composition,
        alphabet=(1, 3, 5, 7),
    ):
        self.composition = tuple(int(c) for c in composition)
        self.alphabet = tuple(int(a) for a in alphabet)

        if len(self.composition) != 4:
            raise ValueError("Per il CCDM 4-ario servono 4 conteggi.")
        if len(self.alphabet) != 4:
            raise ValueError("L'alfabeto deve avere 4 simboli.")
        if len(set(self.alphabet)) != 4:
            raise ValueError("I simboli dell'alfabeto devono essere distinti.")
        if any(c < 0 for c in self.composition):
            raise ValueError("La composizione deve contenere valori >= 0.")

        self.n = sum(self.composition)
        if self.n <= 0:
            raise ValueError("La lunghezza CCDM deve essere > 0.")

        self.M_seq = multinomial_count(self.composition)
        self.k = self.M_seq.bit_length() - 1

        if self.k <= 0:
            raise ValueError("La composizione scelta non trasporta bit.")

        self._symbol_to_index = {
            symbol: idx for idx, symbol in enumerate(self.alphabet)
        }

    def encode(self, input_bits) -> np.ndarray:
        """
        k bit uniformi -> sequenza di n ampiezze a composizione fissata.
        """
        input_bits = np.asarray(input_bits, dtype=np.uint8).reshape(-1)

        if input_bits.size != self.k:
            raise ValueError(
                f"input_bits deve avere lunghezza {self.k}, "
                f"ricevuti {input_bits.size}."
            )

        rank = bits_to_int(input_bits)

        # Usiamo solo i primi 2^k ranking, come nella versione precedente.
        if rank >= (1 << self.k):
            raise ValueError("Rank fuori range.")

        counts = list(self.composition)
        M_current = self.M_seq
        out = np.empty(self.n, dtype=np.int8)

        for pos in range(self.n):
            remaining = self.n - pos
            selected = False

            for symbol_index, symbol in enumerate(self.alphabet):
                count = counts[symbol_index]

                if count == 0:
                    continue

                # Numero di sequenze nel ramo che inizia con questo simbolo.
                branch_count = (M_current * count) // remaining

                if rank < branch_count:
                    out[pos] = symbol
                    counts[symbol_index] -= 1
                    M_current = branch_count
                    selected = True
                    break

                rank -= branch_count

            if not selected:
                raise RuntimeError(
                    "Errore interno nell'unranking CCDM: "
                    "nessun ramo selezionato."
                )

        return out

    def decode(self, amplitude_sequence) -> np.ndarray:
        """
        Sequenza di n ampiezze valida -> k bit uniformi.
        """
        seq = np.asarray(amplitude_sequence).reshape(-1)

        if seq.size != self.n:
            raise ValueError(
                f"La sequenza deve avere lunghezza {self.n}, "
                f"ricevuti {seq.size}."
            )

        counts = list(self.composition)
        M_current = self.M_seq
        rank = 0

        for pos, symbol_value in enumerate(seq):
            symbol = int(symbol_value)

            if symbol not in self._symbol_to_index:
                raise ValueError(f"Simbolo CCDM non valido: {symbol}.")

            true_index = self._symbol_to_index[symbol]
            remaining = self.n - pos

            if counts[true_index] <= 0:
                raise ValueError(
                    "La sequenza non rispetta la composizione fissata."
                )

            # Tutti i rami lessicograficamente precedenti.
            for candidate_index in range(true_index):
                count = counts[candidate_index]

                if count == 0:
                    continue

                rank += (M_current * count) // remaining

            # Entra nel ramo del simbolo realmente osservato.
            true_branch_count = (
                M_current * counts[true_index]
            ) // remaining

            counts[true_index] -= 1
            M_current = true_branch_count

        if any(counts):
            raise ValueError(
                "La sequenza non rispetta la composizione fissata."
            )

        # Il CCDM usa soltanto i ranking [0, 2^k - 1].
        if rank >= (1 << self.k):
            raise ValueError(
                "Sequenza a composizione valida, ma rank fuori "
                "dal sottoinsieme CCDM utilizzato."
            )

        return int_to_bits(rank, self.k)

    def encode_batch(self, input_bits) -> np.ndarray:
        """
        input_bits: shape (B, k)
        output:     shape (B, n)
        """
        x = np.asarray(input_bits, dtype=np.uint8)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.ndim != 2 or x.shape[1] != self.k:
            raise ValueError(
                f"encode_batch richiede shape (B, {self.k})."
            )

        out = np.empty((x.shape[0], self.n), dtype=np.int8)

        for i in range(x.shape[0]):
            out[i] = self.encode(x[i])

        return out

    def decode_batch(self, amplitude_sequences) -> np.ndarray:
        """
        amplitude_sequences: shape (B, n)
        output:              shape (B, k)
        """
        x = np.asarray(amplitude_sequences)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.ndim != 2 or x.shape[1] != self.n:
            raise ValueError(
                f"decode_batch richiede shape (B, {self.n})."
            )

        out = np.empty((x.shape[0], self.k), dtype=np.uint8)

        for i in range(x.shape[0]):
            out[i] = self.decode(x[i])

        return out


@lru_cache(maxsize=32)
def _cached_ccdm(composition_tuple, alphabet_tuple):
    return CCDM4ary(composition_tuple, alphabet_tuple)


def _get_cached(composition, alphabet=(1, 3, 5, 7)):
    return _cached_ccdm(
        tuple(int(c) for c in composition),
        tuple(int(a) for a in alphabet),
    )


# ---------------------------------------------------------------------
# API compatibile con il vecchio ccdm_4ary.py
# ---------------------------------------------------------------------

def ccdm_4ary_k_from_composition(composition):
    return _get_cached(composition).k


def ccdm_4ary_encode(
    input_bits,
    composition,
    alphabet=(1, 3, 5, 7),
):
    return _get_cached(composition, alphabet).encode(input_bits)


def ccdm_4ary_decode(
    amplitude_sequence,
    composition,
    alphabet=(1, 3, 5, 7),
):
    return _get_cached(composition, alphabet).decode(
        amplitude_sequence
    )


if __name__ == "__main__":
    rng = np.random.default_rng(12345)

    composition = (240, 120, 80, 40)
    ccdm = CCDM4ary(composition)

    print("CCDM composition:", composition)
    print("n =", ccdm.n)
    print("k =", ccdm.k)
    print("M_seq bit_length =", ccdm.M_seq.bit_length())

    u = rng.integers(0, 2, ccdm.k, dtype=np.uint8)
    amplitudes = ccdm.encode(u)
    u_hat = ccdm.decode(amplitudes)

    print("Round-trip corretto:", np.array_equal(u, u_hat))
    print(
        "Composizione osservata:",
        [int(np.count_nonzero(amplitudes == a)) for a in ccdm.alphabet],
    )
