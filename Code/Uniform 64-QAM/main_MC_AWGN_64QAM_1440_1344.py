"""
Monte Carlo 64-QAM su canale AWGN con LDPC Staircase (1440, 1344).

Per ogni Eb/N0:
    - vengono generate sequenze indipendenti;
    - l'encoder riparte da B0 = 0;
    - viene generata una nuova realizzazione AWGN;
    - si misura la BER pre-FEC;
    - si decodifica la sequenza Staircase;
    - si accumulano BER, BLER e Sequence FER post-FEC;
    - il punto termina dopo un minimo di sequenze e un numero obiettivo di
      sequenze fallite, oppure al raggiungimento di MAX_SEQUENCES.

NOTA STATISTICA:
    Nel waterfall osservato il decoder tende a produrre o zero errori oppure
    moltissimi errori nella stessa sequenza. Per questo il criterio di arresto
    principale è basato sul numero di SEQUENZE FALLITE e non sul numero di bit
    errati.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from Opt_LDPC_Staircase_Encoder import LDPC_Staircase_Encoder
from Opt_LDPC_Staircase_Decoder_syndrome_window_adaptive_2 import (
    LDPC_Staircase_Decoder,
)
from soft_demap_PAS_AWGN_fast import qam_soft_demap_pas_awgn
from ldpc_encoder import ldpc_encode
from spa_minsum_numba import NumbaMinSumDecoder
from mod_MQAM_optimized import qam_mod_opt
from soft_demap import qam_soft_demap


# =====================================================================
# LOGGING
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Evita il log blocco-per-blocco durante il Monte Carlo.
logging.getLogger(
    "Opt_LDPC_Staircase_Decoder_syndrome_window_adaptive_2"
).setLevel(logging.WARNING)
logging.getLogger(
    "LDPC_Staircase_Decoder_syndrome_window_adaptive_2"
).setLevel(logging.WARNING)

# =====================================================================
# PARAMETRI LDPC STAIRCASE
# =====================================================================

# Codice componente LDPC (1440, 1344)
m = 720
r = 96

# Decoder Staircase
w = 7
alpha_c = 0.75
alpha_n = 0.375
v_max = 8
v_I_max = 24
spa_max_iter = 20

# Manteniamo la stessa lunghezza della simulazione già fatta.
num_info_blocks = 122
num_termination_blocks = 2

info_bits_per_block = m * (m - r)
N_bits = num_info_blocks * info_bits_per_block


# =====================================================================
# PARAMETRI 64-QAM
# =====================================================================

M = 64
k_mod = int(np.log2(M))  # 6 bit/simbolo

# Numero di simboli QAM processati alla volta.
# Ridurre a 50_000 se la RAM è limitata; aumentare se si vuole più velocità.
QAM_CHUNK_SYMBOLS = 100_000


# =====================================================================
# PARAMETRI MONTE CARLO
# =====================================================================

# Prima scansione della regione del cliff individuata con la singola sequenza.
Eb_N0_dB_range = np.array([
    10.63,
    10.635,
    10.64,
    10.645,
])

# Configurazione volutamente "esplorativa" per non rendere troppo pesanti
# i punti sotto soglia, nei quali una sequenza può essere molto costosa.
# Dopo aver localizzato meglio il waterfall si possono aumentare questi valori.
MIN_SEQUENCES = 15
TARGET_FAILED_SEQUENCES = 10
MAX_SEQUENCES = 40

# Salva un checkpoint dopo ogni sequenza.
CHECKPOINT_EVERY = 1

# Seed separati per bit sorgente e rumore.
SOURCE_SEED = 12345
CHANNEL_SEED = 67890

RESULTS_DIR = Path("risultati_AWGN_64QAM_MonteCarlo")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# FUNZIONI DI SUPPORTO
# =====================================================================

def zero_event_upper_bound_95(num_trials: int) -> float:
    """
    Limite superiore unilaterale al 95% quando si osservano zero eventi:

        p_upper = 1 - 0.05^(1/N)

    Per N grande è circa 3/N.
    """
    if num_trials <= 0:
        return np.nan
    return float(-np.expm1(np.log(0.05) / num_trials))


def release_decoder_state(decoder) -> None:
    """Libera le grandi memorie create da decode_sequence()."""
    for name in ("L_I", "L_E_left", "L_E_right"):
        if hasattr(decoder, name):
            setattr(decoder, name, None)


def awgn_64qam_llr_from_bits(
    coded_bits: np.ndarray,
    eb_n0_db: float,
    rate: float,
    rng: np.random.Generator,
    chunk_symbols: int = QAM_CHUNK_SYMBOLS,
) -> tuple[np.ndarray, np.float32, int]:
    """
    64-QAM -> AWGN -> Max-Log LLR, elaborando la sequenza a chunk.

    La normalizzazione e il mapping sono quelli di qam_mod_opt().

    Con Es = 1 e k = log2(M):

        sigma^2 = 1 / (2 * R * k * Eb/N0)

    dove sigma^2 è la varianza di CIASCUNA componente reale del rumore
    complesso n = n_I + j n_Q.

    Restituisce:
        rx_llrs          : LLR float32 nell'ordine originale dei coded_bits
        noise_variance   : sigma^2 per componente reale
        pre_fec_errors   : errori hard prima del decoder Staircase
    """
    coded_bits = np.asarray(coded_bits, dtype=np.uint8).reshape(-1)

    if coded_bits.size == 0:
        raise ValueError("coded_bits non può essere vuoto.")
    if coded_bits.size % k_mod != 0:
        raise ValueError(
            f"Per {M}-QAM il numero di bit deve essere multiplo di {k_mod}."
        )
    if rate <= 0.0:
        raise ValueError("Il code rate deve essere positivo.")
    if chunk_symbols <= 0:
        raise ValueError("chunk_symbols deve essere positivo.")

    eb_n0_linear = 10.0 ** (float(eb_n0_db) / 10.0)

    noise_variance = np.float32(
        1.0 / (2.0 * float(rate) * k_mod * eb_n0_linear)
    )
    noise_std = np.float32(np.sqrt(noise_variance))

    num_symbols = coded_bits.size // k_mod
    rx_llrs = np.empty(coded_bits.size, dtype=np.float32)

    pre_fec_errors = 0

    for sym_start in range(0, num_symbols, chunk_symbols):
        sym_end = min(sym_start + chunk_symbols, num_symbols)

        bit_start = sym_start * k_mod
        bit_end = sym_end * k_mod

        bits_chunk = coded_bits[bit_start:bit_end]

        # 64-QAM Gray, Es = 1.
        tx_symbols = qam_mod_opt(bits_chunk, M)

        n_symbols_chunk = sym_end - sym_start

        noise_real = rng.standard_normal(n_symbols_chunk).astype(
            np.float32,
            copy=False,
        )
        noise_imag = rng.standard_normal(n_symbols_chunk).astype(
            np.float32,
            copy=False,
        )

        noise_real *= noise_std
        noise_imag *= noise_std

        noise = noise_real + 1j * noise_imag
        received_symbols = tx_symbols + noise

        llr_chunk = qam_soft_demap_pas_awgn(
            received_symbols,
            M,
            float(noise_variance),
            composition=(1, 1, 1, 1),
            batch_size=QAM_CHUNK_SYMBOLS,
            dtype=np.float32,
        )
        llr_chunk = np.asarray(llr_chunk, dtype=np.float32)

        rx_llrs[bit_start:bit_end] = llr_chunk

        hard_chunk = llr_chunk < 0.0
        pre_fec_errors += int(
            np.count_nonzero(hard_chunk != bits_chunk)
        )

        del tx_symbols
        del noise_real
        del noise_imag
        del noise
        del received_symbols
        del llr_chunk

    return rx_llrs, noise_variance, pre_fec_errors


def save_checkpoint(
    eb_n0_db: float,
    sequence_count: int,
    counters: dict[str, float | int],
) -> None:
    """Salva i contatori cumulativi del punto corrente."""
    tag = f"{eb_n0_db:.3f}".replace(".", "p")
    filename = RESULTS_DIR / f"checkpoint_{tag}dB.npz"

    np.savez(
        filename,
        Eb_N0_dB=np.float64(eb_n0_db),
        sequence_count=np.int64(sequence_count),
        **counters,
    )


# =====================================================================
# CARICAMENTO MATRICI
# =====================================================================

logger.info("Caricamento matrici LDPC...")

H_matrix = np.load(
    "prova\\H_raw_1440_1344.npy",
    allow_pickle=True,
)
H_matrix = sp.csr_matrix(H_matrix)

G_matrix = np.load(
    "prova\\G_sys_1440_1344.npy",
    allow_pickle=True,
)

perm = np.load(
    "prova\\perm_1440_1344.npy",
    allow_pickle=True,
)


# =====================================================================
# ENCODER + DECODER
# =====================================================================

encoder = LDPC_Staircase_Encoder(
    m,
    r,
    G_matrix,
    ldpc_encode,
)

component_decoder = NumbaMinSumDecoder(
    H_matrix,
    alpha=0.75,
    dtype=np.float32,
)

# Compilazione iniziale Numba fuori dai timer Monte Carlo.
component_decoder.warmup_staircase_kernels(
    m,
    perm=perm,
)

decoder = LDPC_Staircase_Decoder(
    m,
    r,
    w,
    alpha_c,
    alpha_n,
    v_max,
    v_I_max,
    H_matrix,
    component_decoder,
    spa_max_iter=spa_max_iter,
    perm=perm,
    track_extrinsic_stats=False,
)


# =====================================================================
# RNG
# =====================================================================

source_rng = np.random.default_rng(SOURCE_SEED)
channel_rng = np.random.default_rng(CHANNEL_SEED)


# =====================================================================
# CONTENITORI RISULTATI
# =====================================================================

pre_fec_ber_list = []
ber_list = []
bler_list = []
sequence_fer_list = []
sequence_fer_upper95_list = []

num_sequences_list = []
total_info_bits_list = []
total_coded_bits_list = []

post_fec_errors_list = []
pre_fec_errors_list = []
wrong_blocks_list = []
processed_blocks_list = []
wrong_sequences_list = []

average_calls_list = []
average_sweeps_list = []
average_window_size_list = []
time_per_info_bit_list = []
decode_time_list = []


# =====================================================================
# MONTE CARLO
# =====================================================================

simulation_start = time.perf_counter()

for point_index, Eb_N0_dB in enumerate(Eb_N0_dB_range, start=1):

    logger.info("=" * 80)
    logger.info(
        "PUNTO %d/%d - Eb/N0 = %.3f dB",
        point_index,
        len(Eb_N0_dB_range),
        Eb_N0_dB,
    )
    logger.info(
        "Stop: MIN_SEQUENCES=%d, TARGET_FAILED_SEQUENCES=%d, "
        "MAX_SEQUENCES=%d",
        MIN_SEQUENCES,
        TARGET_FAILED_SEQUENCES,
        MAX_SEQUENCES,
    )
    logger.info("=" * 80)

    sequence_count = 0

    total_info_bits = 0
    total_coded_bits = 0

    total_post_fec_errors = 0
    total_pre_fec_errors = 0

    total_wrong_blocks = 0
    total_processed_blocks = 0
    total_wrong_sequences = 0

    total_component_calls = 0
    total_sweeps_weighted = 0.0
    total_window_weighted = 0.0
    total_decode_time = 0.0

    point_start = time.perf_counter()
    stop_reason = "raggiunto MAX_SEQUENCES"

    while sequence_count < MAX_SEQUENCES:

        enough_sequences = sequence_count >= MIN_SEQUENCES
        enough_failures = (
            total_wrong_sequences >= TARGET_FAILED_SEQUENCES
        )

        if enough_sequences and enough_failures:
            stop_reason = (
                "raggiunto il target di sequenze fallite"
            )
            break

        trial_number = sequence_count + 1
        trial_start = time.perf_counter()

        logger.info(
            "Eb/N0 %.3f dB - sequenza Monte Carlo %d",
            Eb_N0_dB,
            trial_number,
        )

        # -------------------------------------------------------------
        # 1. SORGENTE
        # -------------------------------------------------------------

        bits = source_rng.integers(
            0,
            2,
            N_bits,
            dtype=np.uint8,
        )

        # -------------------------------------------------------------
        # 2. ENCODING STAIRCASE
        # -------------------------------------------------------------

        # encode_sequence() dell'encoder ottimizzato resetta già B0.
        # Lo azzeriamo anche esplicitamente per rendere il Monte Carlo evidente.
        encoder.reset()

        if np.count_nonzero(encoder.B_prev) != 0:
            raise RuntimeError("B0 non è stato azzerato correttamente.")

        encode_start = time.perf_counter()

        encoded_blocks = encoder.encode_sequence(
            bits,
            num_termination_blocks=num_termination_blocks,
        )

        encode_time = time.perf_counter() - encode_start

        bits_stream = encoded_blocks.reshape(-1)

        # Rate effettivo della sequenza, inclusi i 2 blocchi di terminazione.
        R_code = N_bits / bits_stream.size

        if bits_stream.size % k_mod != 0:
            raise RuntimeError(
                "La sequenza codificata non è divisibile in simboli 64-QAM."
            )

        # -------------------------------------------------------------
        # 3. 64-QAM + AWGN + SOFT DEMAPPER
        # -------------------------------------------------------------

        channel_start = time.perf_counter()

        rx_llrs, noise_variance, pre_fec_errors = (
            awgn_64qam_llr_from_bits(
                coded_bits=bits_stream,
                eb_n0_db=Eb_N0_dB,
                rate=R_code,
                rng=channel_rng,
                chunk_symbols=QAM_CHUNK_SYMBOLS,
            )
        )

        channel_time = time.perf_counter() - channel_start

        # Il decoder non deve modificare gli LLR di canale.
        rx_llrs.setflags(write=False)

        coded_bits_this_sequence = bits_stream.size

        # I blocchi codificati non servono più.
        del bits_stream
        del encoded_blocks
        gc.collect()

        # -------------------------------------------------------------
        # 4. DECODER STAIRCASE
        # -------------------------------------------------------------

        decoded_bits = decoder.decode_sequence(
            rx_llrs,
            original_bits=bits,
            num_termination_blocks=num_termination_blocks,
        )
        decoded_bits = decoded_bits[:N_bits]

        post_fec_errors = int(
            np.count_nonzero(bits != decoded_bits)
        )

        metrics = dict(decoder.last_metrics)

        processed_blocks = int(metrics["processed_blocks"])
        wrong_blocks = int(metrics["wrong_blocks"])
        component_calls = int(metrics["component_decoder_calls"])
        average_sweeps = float(metrics["average_sweeps"])
        average_window = float(metrics["average_window_size"])
        decode_time = float(metrics["total_decode_time"])

        sequence_failed = int(post_fec_errors > 0)

        # -------------------------------------------------------------
        # 5. AGGIORNA CONTATORI CUMULATIVI
        # -------------------------------------------------------------

        sequence_count += 1

        total_info_bits += N_bits
        total_coded_bits += coded_bits_this_sequence

        total_pre_fec_errors += pre_fec_errors
        total_post_fec_errors += post_fec_errors

        total_wrong_sequences += sequence_failed
        total_wrong_blocks += wrong_blocks
        total_processed_blocks += processed_blocks

        total_component_calls += component_calls
        total_sweeps_weighted += average_sweeps * processed_blocks
        total_window_weighted += average_window * processed_blocks
        total_decode_time += decode_time

        # -------------------------------------------------------------
        # 6. STIME PARZIALI
        # -------------------------------------------------------------

        pre_fec_ber = total_pre_fec_errors / total_coded_bits
        ber = total_post_fec_errors / total_info_bits
        bler = total_wrong_blocks / total_processed_blocks
        sequence_fer = total_wrong_sequences / sequence_count

        avg_calls = total_component_calls / total_processed_blocks
        avg_sweeps = total_sweeps_weighted / total_processed_blocks
        avg_window = total_window_weighted / total_processed_blocks

        trial_time = time.perf_counter() - trial_start

        logger.info(
            "Seq %d: pre-FEC errors=%d, post-FEC errors=%d, "
            "wrong blocks=%d/%d, failed=%s",
            sequence_count,
            pre_fec_errors,
            post_fec_errors,
            wrong_blocks,
            processed_blocks,
            bool(sequence_failed),
        )

        logger.info(
            "Cumulativo: pre-BER=%.6e, BER=%.6e, BLER=%.6e, "
            "Sequence FER=%.6e (%d/%d)",
            pre_fec_ber,
            ber,
            bler,
            sequence_fer,
            total_wrong_sequences,
            sequence_count,
        )

        logger.info(
            "Complessità cumulativa: avg calls=%.2f, avg sweeps=%.2f, "
            "avg window=%.2f",
            avg_calls,
            avg_sweeps,
            avg_window,
        )

        logger.info(
            "Tempi seq: totale=%.2f min, encode=%.2fs, "
            "QAM+AWGN+LLR=%.2fs, decode=%.2fs, sigma2=%.6e, R=%.6f",
            trial_time / 60.0,
            encode_time,
            channel_time,
            decode_time,
            float(noise_variance),
            R_code,
        )

        if total_wrong_sequences == 0:
            logger.info(
                "Zero sequenze fallite: limite superiore 95%% sulla "
                "Sequence FER = %.6e",
                zero_event_upper_bound_95(sequence_count),
            )

        # -------------------------------------------------------------
        # 7. CHECKPOINT
        # -------------------------------------------------------------

        if sequence_count % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                eb_n0_db=Eb_N0_dB,
                sequence_count=sequence_count,
                counters={
                    "total_info_bits": np.int64(total_info_bits),
                    "total_coded_bits": np.int64(total_coded_bits),
                    "pre_fec_errors": np.int64(total_pre_fec_errors),
                    "post_fec_errors": np.int64(total_post_fec_errors),
                    "wrong_sequences": np.int64(total_wrong_sequences),
                    "wrong_blocks": np.int64(total_wrong_blocks),
                    "processed_blocks": np.int64(total_processed_blocks),
                    "component_calls": np.int64(total_component_calls),
                    "decode_time_s": np.float64(total_decode_time),
                },
            )

        # -------------------------------------------------------------
        # 8. LIBERA MEMORIA
        # -------------------------------------------------------------

        del decoded_bits
        del rx_llrs
        del bits
        del metrics

        release_decoder_state(decoder)
        gc.collect()

    # =================================================================
    # RISULTATI FINALI DEL PUNTO
    # =================================================================

    point_time = time.perf_counter() - point_start

    pre_fec_ber = total_pre_fec_errors / total_coded_bits
    ber = total_post_fec_errors / total_info_bits
    bler = total_wrong_blocks / total_processed_blocks
    sequence_fer = total_wrong_sequences / sequence_count

    avg_calls = total_component_calls / total_processed_blocks
    avg_sweeps = total_sweeps_weighted / total_processed_blocks
    avg_window = total_window_weighted / total_processed_blocks
    time_per_info_bit = total_decode_time / total_info_bits

    sequence_fer_upper95 = (
        zero_event_upper_bound_95(sequence_count)
        if total_wrong_sequences == 0
        else np.nan
    )

    pre_fec_ber_list.append(pre_fec_ber)
    ber_list.append(ber)
    bler_list.append(bler)
    sequence_fer_list.append(sequence_fer)
    sequence_fer_upper95_list.append(sequence_fer_upper95)

    num_sequences_list.append(sequence_count)
    total_info_bits_list.append(total_info_bits)
    total_coded_bits_list.append(total_coded_bits)

    pre_fec_errors_list.append(total_pre_fec_errors)
    post_fec_errors_list.append(total_post_fec_errors)
    wrong_blocks_list.append(total_wrong_blocks)
    processed_blocks_list.append(total_processed_blocks)
    wrong_sequences_list.append(total_wrong_sequences)

    average_calls_list.append(avg_calls)
    average_sweeps_list.append(avg_sweeps)
    average_window_size_list.append(avg_window)
    time_per_info_bit_list.append(time_per_info_bit)
    decode_time_list.append(total_decode_time)

    logger.info("=" * 80)
    logger.info(
        "FINE Eb/N0 %.3f dB - %s",
        Eb_N0_dB,
        stop_reason,
    )
    logger.info(
        "Sequenze=%d, fallite=%d, tempo punto=%.2f h",
        sequence_count,
        total_wrong_sequences,
        point_time / 3600.0,
    )
    logger.info(
        "pre-FEC BER=%.6e | post-FEC BER=%.6e | "
        "BLER=%.6e | Sequence FER=%.6e",
        pre_fec_ber,
        ber,
        bler,
        sequence_fer,
    )
    logger.info(
        "avg calls=%.2f | avg sweeps=%.2f | avg window=%.2f | "
        "time/info-bit=%.6e s",
        avg_calls,
        avg_sweeps,
        avg_window,
        time_per_info_bit,
    )

    if total_wrong_sequences == 0:
        logger.info(
            "Sequence FER non misurata: limite superiore 95%% = %.6e",
            sequence_fer_upper95,
        )

    logger.info("=" * 80)

    # Salvataggio progressivo della curva completata fino a questo punto.
    n_done = len(ber_list)

    np.savez(
        RESULTS_DIR / "AWGN_64QAM_MonteCarlo_progressivo2.npz",
        Eb_N0_dB=np.asarray(Eb_N0_dB_range[:n_done], dtype=float),
        pre_fec_ber=np.asarray(pre_fec_ber_list, dtype=float),
        ber=np.asarray(ber_list, dtype=float),
        bler=np.asarray(bler_list, dtype=float),
        sequence_fer=np.asarray(sequence_fer_list, dtype=float),
        sequence_fer_upper95=np.asarray(
            sequence_fer_upper95_list,
            dtype=float,
        ),
        num_sequences=np.asarray(num_sequences_list, dtype=np.int64),
        total_info_bits=np.asarray(total_info_bits_list, dtype=np.int64),
        total_coded_bits=np.asarray(total_coded_bits_list, dtype=np.int64),
        pre_fec_errors=np.asarray(pre_fec_errors_list, dtype=np.int64),
        post_fec_errors=np.asarray(post_fec_errors_list, dtype=np.int64),
        wrong_sequences=np.asarray(wrong_sequences_list, dtype=np.int64),
        wrong_blocks=np.asarray(wrong_blocks_list, dtype=np.int64),
        processed_blocks=np.asarray(processed_blocks_list, dtype=np.int64),
        average_calls=np.asarray(average_calls_list, dtype=float),
        average_sweeps=np.asarray(average_sweeps_list, dtype=float),
        average_window_size=np.asarray(
            average_window_size_list,
            dtype=float,
        ),
        decode_time_s=np.asarray(decode_time_list, dtype=float),
        time_per_info_bit_s=np.asarray(
            time_per_info_bit_list,
            dtype=float,
        ),
    )


# =====================================================================
# ARRAY FINALI
# =====================================================================

Eb_N0_dB_array = np.asarray(Eb_N0_dB_range, dtype=float)

pre_fec_ber_array = np.asarray(pre_fec_ber_list, dtype=float)
ber_array = np.asarray(ber_list, dtype=float)
bler_array = np.asarray(bler_list, dtype=float)
sequence_fer_array = np.asarray(sequence_fer_list, dtype=float)
sequence_fer_upper95_array = np.asarray(
    sequence_fer_upper95_list,
    dtype=float,
)

num_sequences_array = np.asarray(num_sequences_list, dtype=np.int64)
total_info_bits_array = np.asarray(total_info_bits_list, dtype=np.int64)
total_coded_bits_array = np.asarray(total_coded_bits_list, dtype=np.int64)

pre_fec_errors_array = np.asarray(pre_fec_errors_list, dtype=np.int64)
post_fec_errors_array = np.asarray(post_fec_errors_list, dtype=np.int64)
wrong_sequences_array = np.asarray(wrong_sequences_list, dtype=np.int64)
wrong_blocks_array = np.asarray(wrong_blocks_list, dtype=np.int64)
processed_blocks_array = np.asarray(processed_blocks_list, dtype=np.int64)

average_calls_array = np.asarray(average_calls_list, dtype=float)
average_sweeps_array = np.asarray(average_sweeps_list, dtype=float)
average_window_size_array = np.asarray(
    average_window_size_list,
    dtype=float,
)
decode_time_array = np.asarray(decode_time_list, dtype=float)
time_per_info_bit_array = np.asarray(time_per_info_bit_list, dtype=float)


# =====================================================================
# SALVATAGGIO FINALE
# =====================================================================

output_file = RESULTS_DIR / "AWGN_64QAM_MonteCarlo_122_blocchi2.npz"

np.savez(
    output_file,
    Eb_N0_dB=Eb_N0_dB_array,
    M=np.int64(M),
    m=np.int64(m),
    r=np.int64(r),
    w=np.int64(w),
    alpha_c=np.float64(alpha_c),
    alpha_n=np.float64(alpha_n),
    v_max=np.int64(v_max),
    v_I_max=np.int64(v_I_max),
    spa_max_iter=np.int64(spa_max_iter),
    num_info_blocks=np.int64(num_info_blocks),
    num_termination_blocks=np.int64(num_termination_blocks),
    N_bits_per_sequence=np.int64(N_bits),
    min_sequences=np.int64(MIN_SEQUENCES),
    target_failed_sequences=np.int64(TARGET_FAILED_SEQUENCES),
    max_sequences=np.int64(MAX_SEQUENCES),
    source_seed=np.int64(SOURCE_SEED),
    channel_seed=np.int64(CHANNEL_SEED),
    pre_fec_ber=pre_fec_ber_array,
    ber=ber_array,
    bler=bler_array,
    sequence_fer=sequence_fer_array,
    sequence_fer_upper95=sequence_fer_upper95_array,
    num_sequences=num_sequences_array,
    total_info_bits=total_info_bits_array,
    total_coded_bits=total_coded_bits_array,
    pre_fec_errors=pre_fec_errors_array,
    post_fec_errors=post_fec_errors_array,
    wrong_sequences=wrong_sequences_array,
    wrong_blocks=wrong_blocks_array,
    processed_blocks=processed_blocks_array,
    average_calls=average_calls_array,
    average_sweeps=average_sweeps_array,
    average_window_size=average_window_size_array,
    decode_time_s=decode_time_array,
    time_per_info_bit_s=time_per_info_bit_array,
)

logger.info("Risultati salvati in: %s", output_file)


# =====================================================================
# PLOT 1: BER PRE-FEC E POST-FEC
# =====================================================================

plt.figure(figsize=(8, 6))

# PRE-FEC BER
plt.semilogy(
    Eb_N0_dB_array,
    pre_fec_ber_array,
    marker="o",
    linestyle="--",
    label="Pre-FEC 64-QAM",
)

# POST-FEC BER: solo punti con almeno un errore osservato
positive_mask = ber_array > 0.0

if np.any(positive_mask):

    plt.semilogy(
        Eb_N0_dB_array[positive_mask],
        ber_array[positive_mask],
        marker="s",
        linestyle="-",
        label="Post-FEC Staircase",
    )


# PUNTI CON ZERO ERRORI

# BER = 0 non può essere mostrata su scala logaritmica.
#
# Li posizioniamo graficamente a 1/N, ma:
#   - NON sono valori BER misurati;
#   - NON vengono collegati alla curva post-FEC;
#   - usiamo un marker diverso.

zero_mask = ber_array == 0.0

if np.any(zero_mask):

    zero_display = (
        1.0 /
        total_info_bits_array[zero_mask]
    )

    plt.semilogy(
        Eb_N0_dB_array[zero_mask],
        zero_display,
        marker="v",
        linestyle="None",
        markersize=8,
        label="0 errori osservati (marker a 1/N)",
    )

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate")
plt.title("Monte Carlo 64-QAM AWGN - LDPC Staircase")

plt.grid(
    True,
    which="both",
    linestyle=":",
)

plt.legend()
plt.tight_layout()

# plt.savefig(
#     RESULTS_DIR / "BER_64QAM_AWGN_MonteCarlo.png",
#     dpi=200,
# )

plt.show()


# =====================================================================
# PLOT 2: SEQUENCE FER
# =====================================================================

plt.figure(figsize=(8, 6))

# FER realmente misurata
positive_fer_mask = sequence_fer_array > 0.0

if np.any(positive_fer_mask):

    plt.semilogy(
        Eb_N0_dB_array[positive_fer_mask],
        sequence_fer_array[positive_fer_mask],
        marker="o",
        linestyle="-",
        label="Sequence FER misurata",
    )

# ZERO SEQUENZE FALLITE
zero_fer_mask = sequence_fer_array == 0.0

if np.any(zero_fer_mask):

    plt.semilogy(
        Eb_N0_dB_array[zero_fer_mask],
        sequence_fer_upper95_array[zero_fer_mask],
        marker="v",
        linestyle="None",
        markersize=8,
        label="95% upper bound (0 sequenze fallite)",
    )


plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Sequence FER")
plt.title("Probabilità di fallimento della sequenza Staircase")

plt.grid(
    True,
    which="both",
    linestyle=":",
)

plt.legend()
plt.tight_layout()

# plt.savefig(
#     RESULTS_DIR / "Sequence_FER_64QAM_AWGN_MonteCarlo.png",
#     dpi=200,
# )

plt.show()


# =====================================================================
# PLOT 3: COMPLESSITÀ MEDIA
# =====================================================================

plt.figure(figsize=(8, 6))
plt.plot(
    Eb_N0_dB_array,
    average_calls_array,
    marker="o",
    linestyle="-",
)
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Chiamate medie al decoder componente per blocco")
plt.title("Complessità media del decoder Staircase")
plt.grid(True, linestyle=":")
plt.tight_layout()
# plt.savefig(
#     RESULTS_DIR / "CALLS_64QAM_AWGN_MonteCarlo.png",
#     dpi=200,
# )
plt.show()


total_simulation_time = time.perf_counter() - simulation_start

logger.info("=" * 80)
logger.info(
    "SIMULAZIONE COMPLETATA in %.2f ore",
    total_simulation_time / 3600.0,
)
logger.info("File risultati: %s", output_file)
logger.info("=" * 80)
