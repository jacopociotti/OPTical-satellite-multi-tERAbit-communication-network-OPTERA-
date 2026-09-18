import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _parity_u64(value):
    """Restituisce la parità dei 64 bit di value."""
    value ^= value >> 32
    value ^= value >> 16
    value ^= value >> 8
    value ^= value >> 4
    value ^= value >> 2
    value ^= value >> 1
    return np.uint8(value & np.uint64(1))


@njit(cache=True, parallel=True)
def _compute_parity_rows_packed(info_words, parity_words, parity_out):
    """
    Calcola parity_out = info * P mod 2 usando dati impacchettati in uint64.

    info_words  : (m, ceil(K/64))
    parity_words: (r, ceil(K/64)); ogni riga contiene una colonna di P
    parity_out  : (m, r)
    """
    num_rows = info_words.shape[0]
    num_parity = parity_words.shape[0]
    num_words = info_words.shape[1]

    for row in prange(num_rows):
        for parity_idx in range(num_parity):
            accumulator = np.uint64(0)
            for word_idx in range(num_words):
                accumulator ^= (
                    info_words[row, word_idx]
                    & parity_words[parity_idx, word_idx]
                )
            parity_out[row, parity_idx] = _parity_u64(accumulator)


class LDPC_Staircase_Encoder:
    """
    Encoder LDPC Staircase ottimizzato.

    La struttura matematica è identica alla versione originale:
        A_i = [B_{i-1}^T | B_{i,S}]
        B_{i,R} = A_i P mod 2
        B_i = [B_{i,S} | B_{i,R}]

    L'ottimizzazione consiste nel calcolare solamente i r bit di parità e
    nell'impacchettare i bit in parole uint64.
    """

    def __init__(self, m, r, G_matrix, component_LDPC_encoder=None):
        self.m = int(m)
        self.r = int(r)
        self.N = 2 * self.m
        self.K_info_per_row = self.N - self.r
        self.K_info_bits_per_block = self.m * (self.m - self.r)

        G = np.asarray(G_matrix)
        expected_shape = (self.K_info_per_row, self.N)
        if G.shape != expected_shape:
            raise ValueError(
                f"G_matrix deve avere shape {expected_shape}, ricevuta {G.shape}."
            )

        # Verifica necessaria perché vengono estratti direttamente gli ultimi r
        # bit della codeword, come già faceva la versione originale.
        systematic_part = G[:, : self.K_info_per_row]
        identity = np.eye(self.K_info_per_row, dtype=systematic_part.dtype)
        if not np.array_equal(systematic_part, identity):
            raise ValueError(
                "G_matrix non è sistematica nella forma [I | P]. "
                "L'encoder ottimizzato richiede la stessa ipotesi della "
                "versione originale che estraeva codeword[-r:]."
            )

        parity_matrix = np.ascontiguousarray(
            G[:, self.K_info_per_row :], dtype=np.uint8
        )

        # Ogni riga di parity_words rappresenta una colonna di P. I 1344 bit
        # diventano 21 parole uint64 per il codice (1440, 1344).
        self._parity_words = self._pack_columns_to_uint64(parity_matrix)
        self._num_words = self._parity_words.shape[1]

        # Stato Staircase B_0 e buffer riutilizzati.
        self.B_prev = np.zeros((self.m, self.m), dtype=np.uint8)
        self._left_bytes = np.empty((self.m, (self.m + 7) // 8), dtype=np.uint8)
        self._info_bytes = np.empty(
            (self.m, ((self.m - self.r) + 7) // 8), dtype=np.uint8
        )
        total_bytes = (self.K_info_per_row + 7) // 8
        if total_bytes % 8 != 0:
            total_bytes += 8 - (total_bytes % 8)
        self._row_bytes = np.zeros((self.m, total_bytes), dtype=np.uint8)
        self._row_words = self._row_bytes.view(np.uint64).reshape(
            self.m, total_bytes // 8
        )

        # Mantiene la compatibilità con la vecchia firma del costruttore.
        self.encode_row = component_LDPC_encoder

        self._warmup()

    @staticmethod
    def _pack_columns_to_uint64(parity_matrix):
        """Impacchetta le colonne di P in righe di parole uint64."""
        # P.T ha shape (r, K). bitorder='little' mette il bit q nella
        # posizione q % 8, coerente con il successivo view uint64.
        packed_bytes = np.packbits(
            parity_matrix.T,
            axis=1,
            bitorder="little",
        )

        num_bytes = packed_bytes.shape[1]
        padded_bytes = ((num_bytes + 7) // 8) * 8
        if padded_bytes != num_bytes:
            padded = np.zeros(
                (packed_bytes.shape[0], padded_bytes), dtype=np.uint8
            )
            padded[:, :num_bytes] = packed_bytes
            packed_bytes = padded
        else:
            packed_bytes = np.ascontiguousarray(packed_bytes)

        return packed_bytes.view(np.uint64).reshape(
            packed_bytes.shape[0], padded_bytes // 8
        )

    def _warmup(self):
        """Compila il kernel Numba prima della misura dei tempi."""
        dummy_info = np.zeros((1, self._num_words), dtype=np.uint64)
        dummy_out = np.zeros((1, self.r), dtype=np.uint8)
        _compute_parity_rows_packed(
            dummy_info,
            self._parity_words,
            dummy_out,
        )

    def reset(self):
        """Ripristina correttamente il blocco iniziale B_0 a zero."""
        self.B_prev.fill(0)

    def _pack_component_rows(self, B_i_Sx):
        """
        Costruisce la rappresentazione impacchettata di
        A_i = [B_prev.T | B_i_Sx] senza cambiare l'ordine dei bit.
        """
        left = np.packbits(self.B_prev.T, axis=1, bitorder="little")
        right = np.packbits(B_i_Sx, axis=1, bitorder="little")

        left_width = left.shape[1]
        right_width = right.shape[1]

        # I due segmenti hanno lunghezze multiple di 8 nel caso in esame:
        # 720 bit = 90 byte; 624 bit = 78 byte.
        self._row_bytes.fill(0)
        self._row_bytes[:, :left_width] = left
        self._row_bytes[:, left_width : left_width + right_width] = right

        return self._row_words[:, : self._num_words]

    def encode_block_into(self, info_bits, output_block):
        """Codifica un blocco direttamente nell'array output_block."""
        info = np.asarray(info_bits, dtype=np.uint8)
        expected = self.K_info_bits_per_block
        if info.size != expected:
            raise ValueError(
                f"encode_block richiede {expected} bit, ricevuti {info.size}."
            )

        if output_block.shape != (self.m, self.m):
            raise ValueError(
                f"output_block deve avere shape {(self.m, self.m)}."
            )
        if output_block.dtype != np.uint8:
            raise TypeError("output_block deve avere dtype=np.uint8.")

        B_i_Sx = info.reshape(self.m, self.m - self.r)
        output_block[:, : self.m - self.r] = B_i_Sx

        component_rows = self._pack_component_rows(B_i_Sx)
        parity_out = output_block[:, self.m - self.r :]
        _compute_parity_rows_packed(
            component_rows,
            self._parity_words,
            parity_out,
        )

        # Copia nel buffer di stato, evitando una nuova allocazione.
        np.copyto(self.B_prev, output_block)
        return output_block

    def encode_block(self, info_bits):
        """Codifica un singolo blocco B_i."""
        output = np.empty((self.m, self.m), dtype=np.uint8)
        return self.encode_block_into(info_bits, output)

    def encode_sequence(self, bit_stream, num_termination_blocks=2):
        """
        Codifica il flusso e restituisce un array uint8 di shape
        (num_blocchi_totali, m, m).
        """

        self.B_prev.fill(0)  # Reset dello stato iniziale B_0


        assert np.count_nonzero(self.B_prev) == 0, \
            "Errore: B0 non è stato azzerato"

        bits = np.asarray(bit_stream, dtype=np.uint8).reshape(-1)
        if bits.size == 0:
            raise ValueError("bit_stream non può essere vuoto.")
        if np.any(bits > 1):
            raise ValueError("bit_stream deve contenere solamente 0 e 1.")

        num_blocks = int(
            np.ceil(bits.size / self.K_info_bits_per_block)
        )
        total_info_bits = num_blocks * self.K_info_bits_per_block

        if bits.size != total_info_bits:
            padded_bits = np.zeros(total_info_bits, dtype=np.uint8)
            padded_bits[: bits.size] = bits
            bits = padded_bits
        elif not bits.flags.c_contiguous:
            bits = np.ascontiguousarray(bits)

        total_blocks = num_blocks + int(num_termination_blocks)
        encoded_blocks = np.empty(
            (total_blocks, self.m, self.m), dtype=np.uint8
        )

        # Ogni nuova sequenza deve partire dallo stato B_0 = 0.
        self.reset()

        for block_idx in range(num_blocks):
            start = block_idx * self.K_info_bits_per_block
            end = start + self.K_info_bits_per_block
            self.encode_block_into(
                bits[start:end],
                encoded_blocks[block_idx],
            )

        zeros_info = np.zeros(
            self.K_info_bits_per_block, dtype=np.uint8
        )
        for termination_idx in range(num_termination_blocks):
            output_idx = num_blocks + termination_idx
            self.encode_block_into(
                zeros_info,
                encoded_blocks[output_idx],
            )

        return encoded_blocks
