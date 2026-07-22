import numpy as np
import math


def multinomial_count(counts):
    """
    Numero di sequenze distinte con composizione fissata.
    counts = [n1, n3, n5, n7]
    """
    n = sum(counts)
    out = math.factorial(n)

    for c in counts:
        out //= math.factorial(c)

    return out


def bits_to_int(bits):
    bits = np.asarray(bits, dtype=int).flatten()
    value = 0

    for b in bits:
        value = (value << 1) | int(b)

    return value


def int_to_bits(value, k):
    return np.array(
        [(value >> i) & 1 for i in range(k - 1, -1, -1)],
        dtype=int
    )


def ccdm_4ary_k_from_composition(composition):
    """
    Numero di bit uniformi che il CCDM può mappare.

    composition = [n1, n3, n5, n7]
    """
    M_seq = multinomial_count(composition)

    if M_seq <= 0:
        raise ValueError("Numero di sequenze non valido.")

    k = M_seq.bit_length() - 1  # equivalente a floor(log2(M_seq))

    return k


def ccdm_4ary_encode(input_bits, composition, alphabet=(1, 3, 5, 7)):
    """
    CCDM quaternario tramite unranking enumerativo.

    input_bits:
        bit uniformi di lunghezza k

    composition:
        [n1, n3, n5, n7]

    output:
        sequenza di ampiezze A ∈ {1,3,5,7}
        con composizione fissata.
    """

    composition = list(composition)
    n = sum(composition)

    k = ccdm_4ary_k_from_composition(composition)

    input_bits = np.asarray(input_bits, dtype=int).flatten()

    if len(input_bits) != k:
        raise ValueError(f"input_bits deve avere lunghezza k={k}")

    rank = bits_to_int(input_bits)

    if rank >= 2**k:
        raise ValueError("Rank fuori range.")

    counts = composition.copy()
    sequence = []

    for _ in range(n):

        for symbol_index, symbol in enumerate(alphabet):

            if counts[symbol_index] == 0:   # Nessun simbolo di questo tipo rimasto
                continue

            trial_counts = counts.copy()    # serve per simulare cosa succede se scelgo questo simbolo
            trial_counts[symbol_index] -= 1

            num_sequences = multinomial_count(trial_counts)

            if rank < num_sequences:
                sequence.append(symbol)
                counts[symbol_index] -= 1
                break
            else:
                rank -= num_sequences

    return np.array(sequence, dtype=int)


def ccdm_4ary_decode(amplitude_sequence, composition, alphabet=(1, 3, 5, 7)):
    """
    Inverse CCDM quaternario tramite ranking enumerativo.

    amplitude_sequence:
        sequenza A ∈ {1,3,5,7}

    composition:
        [n1, n3, n5, n7]

    output:
        bit uniformi ricostruiti
    """

    amplitude_sequence = np.asarray(amplitude_sequence, dtype=int).flatten()
    composition = list(composition)

    k = ccdm_4ary_k_from_composition(composition)

    counts = composition.copy()
    rank = 0

    symbol_to_index = {a: i for i, a in enumerate(alphabet)} # Mappa simbolo -> indice nell'alfabeto es. {1:0, 3:1, 5:2, 7:3}
                                                             # symbol_to_index[symbol] = index -> symbol_to_index[1] = 0, symbol_to_index[3] = 1, ecc.
    for symbol in amplitude_sequence:

        if symbol not in symbol_to_index:
            raise ValueError(f"Simbolo non valido: {symbol}")

        true_index = symbol_to_index[symbol]

        for candidate_index in range(true_index):

            if counts[candidate_index] == 0:
                continue

            trial_counts = counts.copy()
            trial_counts[candidate_index] -= 1

            rank += multinomial_count(trial_counts)

        counts[true_index] -= 1

        if counts[true_index] < 0:
            raise ValueError("La sequenza non rispetta la composizione fissata.")

    if rank >= 2**k:
        raise ValueError(
            "Questa sequenza ha rank fuori dal sottoinsieme usato dal CCDM."
        )

    return int_to_bits(rank, k)