"""
Monte Carlo PAS 64-QAM su canale AWGN con LDPC Staircase (1440, 1344).

La struttura Monte Carlo, l'early stopping, i checkpoint, il salvataggio
progressivo/finale e i plot seguono lo stesso metodo del main
main_MC_FSO_64QAM_1440_1344.py.

Sono disponibili due modalità:

1) SIMULATION_MODE = "exploratory"
   Il punto termina quando:
       - sono state simulate almeno MIN_SEQUENCES;
       - sono state osservate almeno TARGET_FAILED_SEQUENCES sequenze fallite;
   oppure quando si raggiunge MAX_SEQUENCES.

2) SIMULATION_MODE = "fixed"
   Per ogni Eb/N0 vengono simulate esattamente FIXED_SEQUENCES sequenze.

Una sequenza è considerata fallita se, dopo il decoder Staircase,
rimane almeno un bit sistematico/informativo errato.

Catena:
    source bits
    -> CCDM
    -> systematic LDPC Staircase
    -> PAS mapping
    -> 64-QAM
    -> AWGN
    -> PAS-aware soft demapper
    -> inverse PAS LLR mapping
    -> LDPC Staircase decoder
    -> inverse PAS / inverse CCDM (metriche sorgente)
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from ccdm_4ary_fast import CCDM4ary
from pas64_staircase_utils import (
    PAS64StaircaseLayout,
    staircase_info_from_pas,
    staircase_info_to_pas,
    staircase_blocks_to_qam_bits,
    qam_llrs_to_staircase_llrs,
)
from soft_demap_PAS_AWGN_fast import qam_soft_demap_pas_awgn

from Opt_LDPC_Staircase_Encoder import LDPC_Staircase_Encoder
from Opt_LDPC_Staircase_Decoder_syndrome_window_adaptive_2 import (
    LDPC_Staircase_Decoder,
)
from ldpc_encoder import ldpc_encode
from spa_minsum_numba import NumbaMinSumDecoder
from mod_MQAM_optimized import qam_mod_opt


# =====================================================================
# LOGGING
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Come nel main FSO: niente log INFO blocco-per-blocco del decoder.
logging.getLogger(
    "Opt_LDPC_Staircase_Decoder_syndrome_window_adaptive_2"
).setLevel(logging.WARNING)
logging.getLogger(
    "LDPC_Staircase_Decoder_syndrome_window_adaptive_2"
).setLevel(logging.WARNING)


# =====================================================================
# PARAMETRI LDPC STAIRCASE
# =====================================================================

m = 720
r = 96

w = 7
alpha_c = 0.75
alpha_n = 0.375
v_max = 8
v_I_max = 24
spa_max_iter = 20

NUM_INFO_BLOCKS = 122
NUM_TERMINATION_BLOCKS = 2


# =====================================================================
# PARAMETRI PAS / 64-QAM
# =====================================================================

M = 64
K_MOD = int(np.log2(M))

COMPOSITION = (240, 120, 80, 40)
AMP_XOR_MASK = (1, 0)

QAM_DEMAP_BATCH_SYMBOLS = 100_000


# =====================================================================
# PUNTI Eb/N0
# =====================================================================

EB_N0_DB_RANGE = np.array([
    8.50,
])


# =====================================================================
# MODALITÀ MONTE CARLO
# =====================================================================

# "exploratory" -> early stopping su sequenze fallite
# "fixed"       -> stesso numero di sequenze per ogni Eb/N0
SIMULATION_MODE = "exploratory"


# =====================================================================
# MODALITÀ ESPLORATIVA
# =====================================================================

# Stesso schema del main FSO allegato:
MIN_SEQUENCES = 15
TARGET_FAILED_SEQUENCES = 10
MAX_SEQUENCES = 40


# =====================================================================
# MODALITÀ FIXED
# =====================================================================

FIXED_SEQUENCES = 100


# =====================================================================
# SEED E RISULTATI
# =====================================================================

SOURCE_SEED = 12345
NOISE_SEED = 67890

CHECKPOINT_EVERY = 1

MATRIX_DIR = Path("prova")

# Come nel main FSO: una sola directory dedicata che contiene
# checkpoint, progressivo, finale e plot.
RESULTS_DIR = Path("risultati_PAS_AWGN_64QAM_MonteCarlo")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# FUNZIONI DI SUPPORTO MONTE CARLO
# =====================================================================

def zero_event_upper_bound_95(num_trials: int) -> float:
    """
    Limite superiore unilaterale al 95% quando si osservano zero eventi:

        p_upper = 1 - 0.05^(1/N)

    Per N grande è circa 3/N.
    """
    if num_trials <= 0:
        return np.nan

    return float(
        -np.expm1(
            np.log(0.05) / float(num_trials)
        )
    )


def release_decoder_state(decoder) -> None:
    """Libera le grandi memorie create da decode_sequence()."""
    for name in ("L_I", "L_E_left", "L_E_right"):
        if hasattr(decoder, name):
            setattr(decoder, name, None)


def should_stop_exploratory(
    sequence_count: int,
    failed_sequences: int,
) -> bool:
    """
    Stesso early stopping del main FSO.

    Fermiamo il punto solamente se:
        sequence_count >= MIN_SEQUENCES
    e
        failed_sequences >= TARGET_FAILED_SEQUENCES
    """
    return (sequence_count >= MIN_SEQUENCES and failed_sequences >= TARGET_FAILED_SEQUENCES)


def save_checkpoint(
    eb_n0_db: float,
    sequence_count: int,
    counters: dict[str, float | int],
) -> None:
    """Salva i contatori cumulativi del punto corrente."""
    tag = f"{eb_n0_db:.3f}".replace(".", "p")

    np.savez(
        RESULTS_DIR / f"checkpoint_{tag}dB_2.npz",
        Eb_N0_dB=np.float64(eb_n0_db),
        sequence_count=np.int64(sequence_count),
        **counters,
    )


# =====================================================================
# FUNZIONI PAS / AWGN
# =====================================================================

def get_rates(layout: PAS64StaircaseLayout, ccdm: CCDM4ary) -> dict:
    source_bits_per_info_block = (
        layout.n_ccdm_per_block * ccdm.k
        + layout.n_extra_sign_info_bits_per_block
    )

    source_bits_per_sequence = (NUM_INFO_BLOCKS * source_bits_per_info_block)

    systematic_bits_per_sequence = (NUM_INFO_BLOCKS * layout.info_bits_per_block)

    tx_blocks = NUM_INFO_BLOCKS + NUM_TERMINATION_BLOCKS
    coded_bits_per_sequence = tx_blocks * layout.block_bits

    qam_symbols_per_sequence = coded_bits_per_sequence // K_MOD

    R_pas_info_block = (source_bits_per_info_block / layout.block_bits)

    R_pas_sequence = (source_bits_per_sequence / coded_bits_per_sequence)

    net_bits_per_qam_symbol = K_MOD * R_pas_sequence

    return {
        "source_bits_per_info_block": source_bits_per_info_block,
        "source_bits_per_sequence": source_bits_per_sequence,
        "systematic_bits_per_sequence": systematic_bits_per_sequence,
        "coded_bits_per_sequence": coded_bits_per_sequence,
        "qam_symbols_per_sequence": qam_symbols_per_sequence,
        "R_pas_info_block": R_pas_info_block,
        "R_pas_sequence": R_pas_sequence,
        "net_bits_per_qam_symbol": net_bits_per_qam_symbol,
    }


def exact_pas_sequence_es(
    composition,
    num_info_blocks: int,
    num_termination_blocks: int,
) -> float:
    """
    Energia media ESATTA della sequenza trasmessa con qam_mod_opt().

    qam_mod_opt normalizza la 64-QAM uniforme con norm^2 = 42.

    Per un blocco PAS informativo:
        Es_info = 2 * E[A^2] / 42.

    Per i termination block, AMP_XOR_MASK=(1,0) fa corrispondere
    i bit sistematici 00 ad A=1:
        Es_term = 2 / 42.

    Tutti i blocchi contengono lo stesso numero di simboli QAM,
    quindi la media di sequenza è la media pesata per blocchi.
    """
    comp = np.asarray(composition, dtype=np.float64)
    p_amp = comp / np.sum(comp)

    amp2 = np.array([1.0, 9.0, 25.0, 49.0]) # A^2 per 64-QAM
    E_A2 = float(np.dot(p_amp, amp2))

    Es_info = 2.0 * E_A2 / 42.0
    Es_term = 2.0 / 42.0

    return (num_info_blocks * Es_info + num_termination_blocks * Es_term) / (num_info_blocks + num_termination_blocks)


def build_pas_information_stream(
    rng: np.random.Generator,
    layout: PAS64StaircaseLayout,
    ccdm: CCDM4ary,
):
    """
    Genera UNA nuova sorgente Monte Carlo:

        source DM bits -> CCDM -> amplitude labels
        source sign-info bits -> bit sistematici Staircase

    Restituisce anche i bit sorgente necessari per le metriche PAS.
    """
    info_stream = np.empty(
        NUM_INFO_BLOCKS * layout.info_bits_per_block,
        dtype=np.uint8,
    )

    source_dm = np.empty((NUM_INFO_BLOCKS, layout.n_ccdm_per_block, ccdm.k), dtype=np.uint8) # matrice tridimensionale 122 x 360 x 817

    source_sign = np.empty((NUM_INFO_BLOCKS, layout.n_extra_sign_info_bits_per_block), dtype=np.uint8) # matrice 122 x 103680

    for block_idx in range(NUM_INFO_BLOCKS):
        source_dm[block_idx] = rng.integers(0, 2, size=(layout.n_ccdm_per_block, ccdm.k), dtype=np.uint8) # per ogni blocco informativo riempie la corrispondente matrice 360 x 817 con valori casuali

        amplitudes = ccdm.encode_batch(source_dm[block_idx]).reshape(-1)

        source_sign[block_idx] = rng.integers(0, 2, size=layout.n_extra_sign_info_bits_per_block, dtype=np.uint8) # genera un vettore di 103680 bit casuali

        info_block = staircase_info_from_pas(amplitudes, source_sign[block_idx], m=m, r=r, amp_xor_mask=AMP_XOR_MASK)

        start = block_idx * layout.info_bits_per_block
        end = start + layout.info_bits_per_block
        info_stream[start:end] = info_block

    return info_stream, source_dm, source_sign


# =====================================================================
# 64-QAM + AWGN + PAS SOFT DEMAPPER A BLOCCHI
# =====================================================================

def awgn_pas_staircase_llrs(
    encoded_blocks: np.ndarray,
    eb_n0_db: float,
    R_pas_sequence: float,
    Es_sequence: float,
    noise_rng: np.random.Generator,
):
    """
    Processa UN blocco Staircase alla volta:

        B_i
        -> PAS QAM bits
        -> qam_mod_opt
        -> AWGN
        -> PAS-aware Log-MAP
        -> inverse PAS LLR mapping
        -> LLR nell'ordine originale di B_i

    In memoria rimane soltanto l'array finale di LLR Staircase.
    """
    blocks = np.asarray(encoded_blocks, dtype=np.uint8)

    num_tx_blocks = blocks.shape[0]
    bits_per_block = m * m

    staircase_llrs = np.empty(num_tx_blocks * bits_per_block,dtype=np.float32)

    eb_n0_linear = 10.0 ** (float(eb_n0_db) / 10.0)

    noise_var = (float(Es_sequence) / (2.0 * float(R_pas_sequence) * K_MOD * eb_n0_linear))
    noise_std = np.float32(np.sqrt(noise_var))

    total_pre_fec_errors = 0
    energy_sum = 0.0
    total_symbols = 0

    for block_idx in range(num_tx_blocks):
        block = blocks[block_idx:block_idx + 1]

        qam_bits = staircase_blocks_to_qam_bits(block,m=m,r=r,amp_xor_mask=AMP_XOR_MASK)

        tx = qam_mod_opt(qam_bits, M)

        energy_sum += float(np.sum(np.abs(tx) ** 2, dtype=np.float64))
        total_symbols += tx.size

        nr = noise_rng.standard_normal(tx.size).astype(np.float32,copy=False)
        ni = noise_rng.standard_normal(tx.size).astype(np.float32,copy=False)
        nr *= noise_std
        ni *= noise_std

        rx = tx + (nr + 1j * ni)

        qam_llrs = qam_soft_demap_pas_awgn(rx, M,noise_var,composition=COMPOSITION,batch_size=QAM_DEMAP_BATCH_SYMBOLS,dtype=np.float32)

        hard_qam = qam_llrs < 0.0
        total_pre_fec_errors += int(np.count_nonzero(hard_qam != qam_bits))

        block_llrs = qam_llrs_to_staircase_llrs(qam_llrs,num_blocks=1,m=m,r=r,amp_xor_mask=AMP_XOR_MASK,dtype=np.float32)

        start = block_idx * bits_per_block
        end = start + bits_per_block
        staircase_llrs[start:end] = block_llrs

    measured_Es = energy_sum / total_symbols

    return (staircase_llrs,float(noise_var),int(total_pre_fec_errors),float(measured_Es))


# =====================================================================
# METRICHE PAS POST-FEC
# =====================================================================

def pas_post_fec_metrics(
    decoded_info: np.ndarray,
    info_stream: np.ndarray,
    source_dm: np.ndarray,
    source_sign: np.ndarray,
    layout: PAS64StaircaseLayout,
    ccdm: CCDM4ary,
) -> dict:
    """
    Metriche sorgente PAS.

    Ottimizzazione importante:
    - per i blocchi Staircase perfettamente decodificati NON eseguiamo
      l'inverse CCDM, perché sappiamo già che tutti i source bits sono corretti;
    - l'inverse CCDM viene eseguito solo nei blocchi che contengono almeno
      un errore post-FEC.

    Questo riduce molto il costo del Monte Carlo sopra il cliff.
    """
    decoded = np.asarray(decoded_info, dtype=np.uint8).reshape(-1)
    reference = np.asarray(info_stream, dtype=np.uint8).reshape(-1)

    n_amp = layout.n_amp_code_bits_per_block
    n_sign = layout.n_extra_sign_info_bits_per_block

    amp_code_errors = 0
    sign_info_errors = 0

    dm_failures = 0
    dm_bit_errors_valid = 0

    total_ccdm = NUM_INFO_BLOCKS * layout.n_ccdm_per_block

    # Tutti i CCDM dei blocchi corretti sono automaticamente validi.
    dm_valid = 0

    erroneous_info_blocks = 0

    for block_idx in range(NUM_INFO_BLOCKS):
        start = block_idx * layout.info_bits_per_block
        end = start + layout.info_bits_per_block

        ref_block = reference[start:end]
        dec_block = decoded[start:end]

        diff = dec_block != ref_block

        if not np.any(diff):
            dm_valid += layout.n_ccdm_per_block
            continue

        erroneous_info_blocks += 1

        amp_err = int(np.count_nonzero(diff[:n_amp]))
        sign_err = int(np.count_nonzero(diff[n_amp:n_amp + n_sign]))

        amp_code_errors += amp_err
        sign_info_errors += sign_err

        # Se in questo blocco non ci sono errori di ampiezza, tutti i suoi
        # CCDM sono corretti e non serve decodificarli.
        if amp_err == 0:
            dm_valid += layout.n_ccdm_per_block
            continue

        amp_hat, _ = staircase_info_to_pas(dec_block,m=m,r=r,amp_xor_mask=AMP_XOR_MASK)

        amp_hat = amp_hat.reshape(layout.n_ccdm_per_block,ccdm.n)

        u_ref = source_dm[block_idx]

        for j in range(layout.n_ccdm_per_block):
            try:
                u_hat = ccdm.decode(amp_hat[j])
            except ValueError:
                dm_failures += 1
                continue

            dm_valid += 1
            dm_bit_errors_valid += int(np.count_nonzero(u_hat != u_ref[j]))

    # Nei blocchi completamente corretti gli errori erano zero.
    # Nei blocchi errati abbiamo contato separatamente amp/sign.
    amp_bits_total = NUM_INFO_BLOCKS * n_amp
    sign_bits_total = NUM_INFO_BLOCKS * n_sign

    dm_bits_valid = dm_valid * ccdm.k

    source_sequence_error = int(sign_info_errors > 0 or dm_failures > 0 or dm_bit_errors_valid > 0)

    return {
        "amp_code_errors": amp_code_errors,
        "amp_code_bits": amp_bits_total,
        "sign_info_errors": sign_info_errors,
        "sign_info_bits": sign_bits_total,
        "dm_failures": dm_failures,
        "dm_total": total_ccdm,
        "dm_valid": dm_valid,
        "dm_bit_errors_valid": dm_bit_errors_valid,
        "dm_bits_valid": dm_bits_valid,
        "source_sequence_error": source_sequence_error,
        "erroneous_info_blocks": erroneous_info_blocks,
    }


# =====================================================================
# INIZIALIZZAZIONE PAS
# =====================================================================

ccdm = CCDM4ary(COMPOSITION)

layout = PAS64StaircaseLayout(m=m,r=r,ccdm_length=ccdm.n)
layout.validate()

rates = get_rates(layout, ccdm)

Es_sequence = exact_pas_sequence_es(COMPOSITION,NUM_INFO_BLOCKS,NUM_TERMINATION_BLOCKS)

logger.info("=" * 90)
logger.info("PAS + LDPC STAIRCASE + 64-QAM + AWGN -- MONTE CARLO")
logger.info("=" * 90)
logger.info("Info / termination blocks = %d / %d", NUM_INFO_BLOCKS, NUM_TERMINATION_BLOCKS)
logger.info("CCDM n / k                 = %d / %d", ccdm.n, ccdm.k)
logger.info("CCDM per info block        = %d", layout.n_ccdm_per_block)
logger.info("Source bits / info block   = %d", rates["source_bits_per_info_block"])
logger.info("Source bits / sequence     = %d", rates["source_bits_per_sequence"])
logger.info("Systematic bits / sequence = %d", rates["systematic_bits_per_sequence"])
logger.info("Coded bits / sequence      = %d", rates["coded_bits_per_sequence"])
logger.info("R_PAS info block           = %.9f", rates["R_pas_info_block"])
logger.info("R_PAS sequence incl. term  = %.9f", rates["R_pas_sequence"])
logger.info("Net bits / QAM symbol      = %.9f", rates["net_bits_per_qam_symbol"])
logger.info("Es sequence exact          = %.9f", Es_sequence)
logger.info("=" * 90)


# =====================================================================
# CARICAMENTO MATRICI
# =====================================================================

logger.info("Caricamento matrici LDPC...")

H_matrix = np.load(MATRIX_DIR / "H_raw_1440_1344.npy", allow_pickle=True)
H_matrix = sp.csr_matrix(H_matrix)

G_matrix = np.load(MATRIX_DIR / "G_sys_1440_1344.npy", allow_pickle=True)

perm = np.load(MATRIX_DIR / "perm_1440_1344.npy",allow_pickle=True)


# =====================================================================
# INIZIALIZZAZIONE ENCODER E DECODER
# =====================================================================

encoder = LDPC_Staircase_Encoder(m,r,G_matrix,ldpc_encode)

component_decoder = NumbaMinSumDecoder(H_matrix,alpha=0.75,dtype=np.float32)

component_decoder.warmup_staircase_kernels(m,perm=perm)

decoder = LDPC_Staircase_Decoder(m,r,w,alpha_c,alpha_n,v_max,v_I_max,H_matrix,component_decoder,spa_max_iter=spa_max_iter,perm=perm,track_extrinsic_stats=False)


# =====================================================================
# GENERATORI CASUALI
# =====================================================================

source_rng = np.random.default_rng(SOURCE_SEED)
noise_rng = np.random.default_rng(NOISE_SEED)


# =====================================================================
# CONTENITORI RISULTATI
# =====================================================================

pre_fec_ber_list = []
ber_list = []
bler_list = []

sequence_fer_list = []
sequence_fer_upper95_list = []

failure_probability_list = []

num_sequences_list = []
total_info_bits_list = []
total_coded_bits_list = []

pre_fec_errors_list = []
post_fec_errors_list = []

wrong_sequences_list = []
wrong_blocks_list = []
processed_blocks_list = []
failed_blocks_list = []

average_calls_list = []
average_sweeps_list = []
average_window_size_list = []

decode_time_list = []
time_per_info_bit_list = []

# Metriche specifiche PAS
amp_code_ber_list = []
sign_info_ber_list = []
ccdm_failure_rate_list = []
dm_ber_valid_list = []
source_sequence_fer_list = []
source_wrong_sequences_list = []

amp_code_errors_list = []
sign_info_errors_list = []
ccdm_failures_list = []


# =====================================================================
# CICLO SUI PUNTI Eb/N0
# =====================================================================

simulation_start = time.perf_counter()

for point_index, Eb_N0_dB in enumerate(EB_N0_DB_RANGE,start=1):

    logger.info("=" * 90)
    logger.info("PUNTO %d/%d - Eb/N0 = %.3f dB", point_index,len(EB_N0_DB_RANGE), Eb_N0_dB)
    logger.info("Modalità Monte Carlo: %s",SIMULATION_MODE)

    if SIMULATION_MODE == "exploratory":
        logger.info(
            "Stop esplorativo: min_seq=%d, "
            "target_failed_seq=%d, max_seq=%d",
            MIN_SEQUENCES,
            TARGET_FAILED_SEQUENCES,
            MAX_SEQUENCES,
        )
        max_trials_this_point = MAX_SEQUENCES

    elif SIMULATION_MODE == "fixed":
        logger.info("Numero fisso di sequenze: %d",FIXED_SEQUENCES)
        max_trials_this_point = FIXED_SEQUENCES

    else:
        raise ValueError('SIMULATION_MODE deve essere "exploratory" oppure "fixed".')

    logger.info("=" * 90)

    # -------------------------------------------------------------
    # Contatori cumulativi del punto
    # -------------------------------------------------------------

    sequence_count = 0

    total_info_bits = 0
    total_coded_bits = 0

    total_pre_fec_errors = 0
    total_post_fec_errors = 0

    total_wrong_sequences = 0

    total_wrong_blocks = 0
    total_processed_blocks = 0
    total_failed_blocks = 0

    total_component_calls = 0
    total_sweeps_weighted = 0.0
    total_window_weighted = 0.0

    total_decode_time = 0.0

    # PAS
    total_amp_errors = 0
    total_amp_bits = 0

    total_sign_errors = 0
    total_sign_bits = 0

    total_ccdm_failures = 0
    total_ccdm_blocks = 0

    total_dm_bit_errors_valid = 0
    total_dm_bits_valid = 0

    total_source_wrong_sequences = 0

    point_start = time.perf_counter()

    # =============================================================
    # MONTE CARLO DEL PUNTO
    # =============================================================

    while sequence_count < max_trials_this_point:

        # Come nel main FSO: controllo PRIMA della nuova sequenza.
        if (
            SIMULATION_MODE == "exploratory"
            and should_stop_exploratory(
                sequence_count,
                total_wrong_sequences,
            )
        ):
            stop_reason = (
                "target di sequenze fallite raggiunto "
                "dopo il minimo di sequenze"
            )
            break

        trial_number = sequence_count + 1
        trial_start = time.perf_counter()

        logger.info("Eb/N0 %.3f dB - sequenza Monte Carlo %d", Eb_N0_dB, trial_number)

        # ---------------------------------------------------------
        # 1. SORGENTE + CCDM
        # ---------------------------------------------------------

        source_start = time.perf_counter()

        (info_stream, source_dm, source_sign) = build_pas_information_stream(source_rng, layout, ccdm)

        source_time = time.perf_counter() - source_start

        # ---------------------------------------------------------
        # 2. ENCODER STAIRCASE
        # ---------------------------------------------------------

        encoder.reset()

        if np.count_nonzero(encoder.B_prev) != 0:
            raise RuntimeError(
                "B0 non è stato azzerato correttamente."
            )

        encode_start = time.perf_counter()

        encoded_blocks = encoder.encode_sequence(info_stream, num_termination_blocks=NUM_TERMINATION_BLOCKS)

        encode_time = time.perf_counter() - encode_start

        encoded_blocks = np.asarray(encoded_blocks, dtype=np.uint8)

        # ---------------------------------------------------------
        # 3. PAS + 64-QAM + AWGN + SOFT DEMAPPER
        # ---------------------------------------------------------

        channel_demap_start = time.perf_counter()

        (rx_llrs, noise_var, pre_fec_errors, measured_Es) = awgn_pas_staircase_llrs(encoded_blocks, Eb_N0_dB, rates["R_pas_sequence"], Es_sequence, noise_rng)

        channel_demap_time = (time.perf_counter() - channel_demap_start)

        if not np.isclose(
            measured_Es,
            Es_sequence,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"Es misurata {measured_Es} "
                f"!= Es attesa {Es_sequence}."
            )

        # ---------------------------------------------------------
        # 4. DECODER LDPC STAIRCASE
        # ---------------------------------------------------------

        decode_start = time.perf_counter()

        decoded_bits = decoder.decode_sequence(rx_llrs,original_bits=info_stream,num_termination_blocks=NUM_TERMINATION_BLOCKS)

        measured_decode_time = (time.perf_counter() - decode_start)

        decoded_bits = np.asarray(decoded_bits,dtype=np.uint8).reshape(-1)

        decoded_bits = decoded_bits[:info_stream.size]

        post_fec_errors = int(np.count_nonzero(info_stream != decoded_bits))

        sequence_failed = int(post_fec_errors > 0)

        metrics = dict(decoder.last_metrics)

        processed_blocks = int(metrics["processed_blocks"])
        wrong_blocks = int(metrics["wrong_blocks"])
        failed_blocks = int(metrics.get("failed_blocks", 0))

        component_calls = int(metrics["component_decoder_calls"])

        average_sweeps = float(metrics["average_sweeps"])
        average_window = float(metrics["average_window_size"])

        decode_time = float(metrics.get("total_decode_time",measured_decode_time))

        # ---------------------------------------------------------
        # 5. METRICHE PAS / SOURCE
        # ---------------------------------------------------------

        pas_metrics = pas_post_fec_metrics(decoded_bits,info_stream,source_dm,source_sign,layout, ccdm)

        source_sequence_failed = int(pas_metrics["source_sequence_error"])

        if sequence_failed != source_sequence_failed:
            logger.warning(
                "Sequence FER FEC=%d e source FER=%d differiscono.",
                sequence_failed,
                source_sequence_failed,
            )

        # ---------------------------------------------------------
        # 6. AGGIORNAMENTO CONTATORI
        # ---------------------------------------------------------

        sequence_count += 1

        total_info_bits += info_stream.size
        total_coded_bits += rates["coded_bits_per_sequence"]

        total_pre_fec_errors += pre_fec_errors
        total_post_fec_errors += post_fec_errors

        total_wrong_sequences += sequence_failed

        total_processed_blocks += processed_blocks
        total_wrong_blocks += wrong_blocks
        total_failed_blocks += failed_blocks

        total_component_calls += component_calls

        total_sweeps_weighted += (average_sweeps * processed_blocks)
        total_window_weighted += (average_window * processed_blocks)

        total_decode_time += decode_time

        total_amp_errors += pas_metrics["amp_code_errors"]
        total_amp_bits += pas_metrics["amp_code_bits"]

        total_sign_errors += pas_metrics["sign_info_errors"]
        total_sign_bits += pas_metrics["sign_info_bits"]

        total_ccdm_failures += pas_metrics["dm_failures"]
        total_ccdm_blocks += pas_metrics["dm_total"]

        total_dm_bit_errors_valid += pas_metrics["dm_bit_errors_valid"]
        total_dm_bits_valid += pas_metrics["dm_bits_valid"]

        total_source_wrong_sequences += (source_sequence_failed)

        # ---------------------------------------------------------
        # 7. STIME CUMULATIVE
        # ---------------------------------------------------------

        pre_fec_ber_partial = (total_pre_fec_errors / total_coded_bits)

        ber_partial = (total_post_fec_errors / total_info_bits)

        bler_partial = (total_wrong_blocks / total_processed_blocks if total_processed_blocks > 0 else 0.0)

        sequence_fer_partial = (total_wrong_sequences / sequence_count)

        amp_ber_partial = (total_amp_errors / total_amp_bits if total_amp_bits > 0 else 0.0)

        sign_ber_partial = (total_sign_errors / total_sign_bits if total_sign_bits > 0 else 0.0)

        ccdm_failure_partial = (total_ccdm_failures / total_ccdm_blocks if total_ccdm_blocks > 0 else 0.0)

        trial_time = (time.perf_counter() - trial_start)
        logger.info(
            "Seq %d conclusa in %.2f min | "
            "source+CCDM %.2fs | encoder %.2fs | "
            "AWGN+demap %.2fs | decode %.2fs",
            sequence_count,
            trial_time / 60.0,
            source_time,
            encode_time,
            channel_demap_time,
            decode_time,
        )

        logger.info(
            "Seq %d: pre-FEC errors=%d | "
            "post-FEC errors=%d | "
            "wrong blocks=%d/%d | failed=%s",
            sequence_count,
            pre_fec_errors,
            post_fec_errors,
            wrong_blocks,
            processed_blocks,
            bool(sequence_failed),
        )

        logger.info(
            "Canale seq %d: sigma2=%.6e | Es=%.9f",
            sequence_count,
            float(noise_var),
            measured_Es,
        )

        logger.info(
            "CUMULATIVO: pre-BER=%.6e | BER=%.6e | "
            "BLER=%.6e | Sequence FER=%.6e (%d/%d)",
            pre_fec_ber_partial,
            ber_partial,
            bler_partial,
            sequence_fer_partial,
            total_wrong_sequences,
            sequence_count,
        )

        logger.info(
            "PAS cumulativo: ampBER=%.6e | signBER=%.6e | "
            "CCDM failure=%.6e",
            amp_ber_partial,
            sign_ber_partial,
            ccdm_failure_partial,
        )

        # ---------------------------------------------------------
        # 8. CHECKPOINT
        # ---------------------------------------------------------

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
                    "failed_blocks": np.int64(total_failed_blocks),
                    "component_calls": np.int64(total_component_calls),
                    "decode_time_s": np.float64(total_decode_time),
                    "amp_code_errors": np.int64(total_amp_errors),
                    "sign_info_errors": np.int64(total_sign_errors),
                    "ccdm_failures": np.int64(total_ccdm_failures),
                    "source_wrong_sequences": np.int64(total_source_wrong_sequences),
                },
            )

        # ---------------------------------------------------------
        # 9. LIBERAZIONE MEMORIA
        # ---------------------------------------------------------

        del decoded_bits
        del metrics

        release_decoder_state(decoder)

        del rx_llrs
        del encoded_blocks
        del info_stream
        del source_dm
        del source_sign

        gc.collect()

    else:
        if SIMULATION_MODE == "fixed":
            stop_reason = (f"completate {FIXED_SEQUENCES} sequenze fisse")
        else:
            stop_reason = (f"raggiunto MAX_SEQUENCES={MAX_SEQUENCES}")

    # =============================================================
    # RISULTATI FINALI DEL PUNTO
    # =============================================================

    point_time = (time.perf_counter() - point_start)

    pre_fec_ber = (total_pre_fec_errors / total_coded_bits)

    ber = (total_post_fec_errors / total_info_bits)

    bler = (total_wrong_blocks / total_processed_blocks if total_processed_blocks > 0 else 0.0)

    sequence_fer = (total_wrong_sequences / sequence_count)

    failure_probability = (total_failed_blocks / total_processed_blocks if total_processed_blocks > 0 else 0.0)

    average_calls = (total_component_calls / total_processed_blocks if total_processed_blocks > 0 else 0.0)

    average_sweeps_final = (total_sweeps_weighted / total_processed_blocks if total_processed_blocks > 0 else 0.0)

    average_window_final = (total_window_weighted / total_processed_blocks if total_processed_blocks > 0 else 0.0)

    time_per_info_bit = (total_decode_time / total_info_bits)

    amp_code_ber = (total_amp_errors / total_amp_bits if total_amp_bits > 0 else 0.0)

    sign_info_ber = (total_sign_errors / total_sign_bits if total_sign_bits > 0 else 0.0)

    ccdm_failure_rate = (total_ccdm_failures / total_ccdm_blocks if total_ccdm_blocks > 0 else 0.0)

    dm_ber_valid = (total_dm_bit_errors_valid / total_dm_bits_valid if total_dm_bits_valid > 0 else np.nan)

    source_sequence_fer = (total_source_wrong_sequences / sequence_count)

    if total_wrong_sequences == 0:
        sequence_fer_upper95 = (zero_event_upper_bound_95(sequence_count))
    else:
        sequence_fer_upper95 = np.nan

    # -------------------------------------------------------------
    # Liste finali
    # -------------------------------------------------------------

    pre_fec_ber_list.append(pre_fec_ber)
    ber_list.append(ber)
    bler_list.append(bler)

    sequence_fer_list.append(sequence_fer)
    sequence_fer_upper95_list.append(sequence_fer_upper95)

    failure_probability_list.append(failure_probability)

    num_sequences_list.append(sequence_count)
    total_info_bits_list.append(total_info_bits)
    total_coded_bits_list.append(total_coded_bits)

    pre_fec_errors_list.append(total_pre_fec_errors)
    post_fec_errors_list.append(total_post_fec_errors)

    wrong_sequences_list.append(total_wrong_sequences)
    wrong_blocks_list.append(total_wrong_blocks)
    processed_blocks_list.append(total_processed_blocks)
    failed_blocks_list.append(total_failed_blocks)

    average_calls_list.append(average_calls)
    average_sweeps_list.append(average_sweeps_final)
    average_window_size_list.append(average_window_final)

    decode_time_list.append(total_decode_time)
    time_per_info_bit_list.append(time_per_info_bit)

    amp_code_ber_list.append(amp_code_ber)
    sign_info_ber_list.append(sign_info_ber)
    ccdm_failure_rate_list.append(ccdm_failure_rate)
    dm_ber_valid_list.append(dm_ber_valid)
    source_sequence_fer_list.append(source_sequence_fer)
    source_wrong_sequences_list.append(total_source_wrong_sequences)

    amp_code_errors_list.append(total_amp_errors)
    sign_info_errors_list.append(total_sign_errors)
    ccdm_failures_list.append(total_ccdm_failures)

    logger.info("=" * 90)
    logger.info(
        "FINE Eb/N0 %.2f dB: %s",
        Eb_N0_dB,
        stop_reason,
    )
    logger.info(
        "Sequenze=%d | tempo punto=%.2f h",
        sequence_count,
        point_time / 3600.0,
    )
    logger.info(
        "Pre-FEC BER = %.6e (%d errori)",
        pre_fec_ber,
        total_pre_fec_errors,
    )
    logger.info(
        "Post-FEC BER = %.6e (%d errori)",
        ber,
        total_post_fec_errors,
    )
    logger.info(
        "BLER = %.6e (%d/%d blocchi)",
        bler,
        total_wrong_blocks,
        total_processed_blocks,
    )
    logger.info(
        "Sequence FER = %.6e (%d/%d sequenze)",
        sequence_fer,
        total_wrong_sequences,
        sequence_count,
    )

    if total_wrong_sequences == 0:
        logger.info(
            "Zero sequenze fallite: upper bound 95%% "
            "Sequence FER = %.6e",
            sequence_fer_upper95,
        )

    logger.info(
        "Failure probability decoder = %.6e",
        failure_probability,
    )
    logger.info(
        "Average calls/block = %.2f | "
        "avg sweeps = %.3f | avg window = %.3f",
        average_calls,
        average_sweeps_final,
        average_window_final,
    )
    logger.info(
        "PAS: ampBER=%.6e | signBER=%.6e | "
        "CCDM failure=%.6e | source SeqFER=%.6e",
        amp_code_ber,
        sign_info_ber,
        ccdm_failure_rate,
        source_sequence_fer,
    )
    logger.info("=" * 90)

    # =============================================================
    # SALVATAGGIO PROGRESSIVO DELLA CURVA
    # =============================================================

    np.savez(
        RESULTS_DIR / "PAS_AWGN_64QAM_MonteCarlo_progressivo_test.npz",

        Eb_N0_dB=np.asarray(EB_N0_DB_RANGE[:len(ber_list)],dtype=float),

        pre_fec_ber=np.asarray(pre_fec_ber_list,dtype=float),
        ber=np.asarray(ber_list,dtype=float),
        bler=np.asarray(bler_list,dtype=float),

        sequence_fer=np.asarray(sequence_fer_list,dtype=float),
        sequence_fer_upper95=np.asarray(sequence_fer_upper95_list,dtype=float),

        failure_probability=np.asarray(failure_probability_list,dtype=float),

        num_sequences=np.asarray(num_sequences_list,dtype=np.int64),
        total_info_bits=np.asarray(total_info_bits_list,dtype=np.int64),
        total_coded_bits=np.asarray(total_coded_bits_list,dtype=np.int64),

        pre_fec_errors=np.asarray(pre_fec_errors_list,dtype=np.int64),
        post_fec_errors=np.asarray(post_fec_errors_list,dtype=np.int64),

        wrong_sequences=np.asarray(wrong_sequences_list,dtype=np.int64),
        wrong_blocks=np.asarray(wrong_blocks_list,dtype=np.int64),
        processed_blocks=np.asarray(processed_blocks_list,dtype=np.int64),
        failed_blocks=np.asarray(failed_blocks_list,dtype=np.int64),

        average_calls=np.asarray(average_calls_list,dtype=float),
        average_sweeps=np.asarray(average_sweeps_list,dtype=float),
        average_window_size=np.asarray(average_window_size_list,dtype=float),

        decode_time_s=np.asarray(decode_time_list,dtype=float),
        time_per_info_bit_s=np.asarray(time_per_info_bit_list,dtype=float),

        amp_code_ber=np.asarray(amp_code_ber_list,dtype=float),
        sign_info_ber=np.asarray(sign_info_ber_list,dtype=float),
        ccdm_failure_rate=np.asarray(ccdm_failure_rate_list,dtype=float),
        dm_ber_valid=np.asarray(dm_ber_valid_list,dtype=float),
        source_sequence_fer=np.asarray(source_sequence_fer_list, dtype=float),
        source_wrong_sequences=np.asarray(source_wrong_sequences_list,dtype=np.int64),

        amp_code_errors=np.asarray(amp_code_errors_list,dtype=np.int64),
        sign_info_errors=np.asarray(sign_info_errors_list,dtype=np.int64),
        ccdm_failures=np.asarray(ccdm_failures_list,dtype=np.int64),
    )


# =====================================================================
# CONVERSIONE FINALE IN ARRAY
# =====================================================================

Eb_N0_dB_array = np.asarray(EB_N0_DB_RANGE,dtype=float)

pre_fec_ber_array = np.asarray(pre_fec_ber_list,dtype=float)
ber_array = np.asarray(ber_list,dtype=float)
bler_array = np.asarray(bler_list,dtype=float)

sequence_fer_array = np.asarray(sequence_fer_list, dtype=float)
sequence_fer_upper95_array = np.asarray(sequence_fer_upper95_list,dtype=float)

failure_probability_array = np.asarray(failure_probability_list,dtype=float)

num_sequences_array = np.asarray(num_sequences_list, dtype=np.int64)
total_info_bits_array = np.asarray(total_info_bits_list,dtype=np.int64)
total_coded_bits_array = np.asarray(total_coded_bits_list,dtype=np.int64)

pre_fec_errors_array = np.asarray(pre_fec_errors_list,dtype=np.int64)
post_fec_errors_array = np.asarray(post_fec_errors_list,dtype=np.int64)

wrong_sequences_array = np.asarray(wrong_sequences_list,dtype=np.int64)
wrong_blocks_array = np.asarray(wrong_blocks_list,dtype=np.int64)
processed_blocks_array = np.asarray(processed_blocks_list,dtype=np.int64)
failed_blocks_array = np.asarray(failed_blocks_list,dtype=np.int64)
average_calls_array = np.asarray(average_calls_list,dtype=float)
average_sweeps_array = np.asarray(average_sweeps_list,dtype=float)
average_window_size_array = np.asarray(average_window_size_list,dtype=float)

decode_time_array = np.asarray(decode_time_list, dtype=float)
time_per_info_bit_array = np.asarray(time_per_info_bit_list,dtype=float)

amp_code_ber_array = np.asarray(amp_code_ber_list,dtype=float,)
sign_info_ber_array = np.asarray(sign_info_ber_list,dtype=float)
ccdm_failure_rate_array = np.asarray(ccdm_failure_rate_list,dtype=float)
dm_ber_valid_array = np.asarray(dm_ber_valid_list,dtype=float)
source_sequence_fer_array = np.asarray(source_sequence_fer_list,dtype=float)
source_wrong_sequences_array = np.asarray(source_wrong_sequences_list,dtype=np.int64)

amp_code_errors_array = np.asarray(amp_code_errors_list,dtype=np.int64)
sign_info_errors_array = np.asarray(sign_info_errors_list, dtype=np.int64)
ccdm_failures_array = np.asarray(ccdm_failures_list,dtype=np.int64)


# =====================================================================
# SALVATAGGIO FINALE
# =====================================================================

output_file = (
    RESULTS_DIR
    / f"PAS_AWGN_64QAM_MonteCarlo_122_blocchi_{SIMULATION_MODE}_test.npz"
)

np.savez(
    output_file,

    Eb_N0_dB=Eb_N0_dB_array,

    M=np.int64(M),
    bits_per_symbol=np.int64(K_MOD),

    m=np.int64(m),
    r=np.int64(r),
    w=np.int64(w),

    alpha_c=np.float64(alpha_c),
    alpha_n=np.float64(alpha_n),

    v_max=np.int64(v_max),
    v_I_max=np.int64(v_I_max),
    spa_max_iter=np.int64(spa_max_iter),

    num_info_blocks=np.int64(NUM_INFO_BLOCKS),
    num_termination_blocks=np.int64(NUM_TERMINATION_BLOCKS),

    composition=np.asarray(COMPOSITION,dtype=np.int64),
    amp_xor_mask=np.asarray(AMP_XOR_MASK,dtype=np.int64),
    ccdm_n=np.int64(ccdm.n),
    ccdm_k=np.int64(ccdm.k),

    R_pas_info_block=np.float64(rates["R_pas_info_block"]),
    R_pas_sequence=np.float64(rates["R_pas_sequence"]),
    net_bits_per_qam_symbol=np.float64(rates["net_bits_per_qam_symbol"]),
    Es_sequence=np.float64(Es_sequence),

    simulation_mode=np.array(SIMULATION_MODE),

    min_sequences=np.int64(MIN_SEQUENCES),
    target_failed_sequences=np.int64(TARGET_FAILED_SEQUENCES),
    max_sequences=np.int64(MAX_SEQUENCES),
    fixed_sequences=np.int64(FIXED_SEQUENCES),

    source_seed=np.int64(SOURCE_SEED),
    noise_seed=np.int64(NOISE_SEED),

    pre_fec_ber=pre_fec_ber_array,
    ber=ber_array,
    bler=bler_array,

    sequence_fer=sequence_fer_array,
    sequence_fer_upper95=(sequence_fer_upper95_array),

    failure_probability=(failure_probability_array),

    num_sequences=num_sequences_array,

    total_info_bits=total_info_bits_array,
    total_coded_bits=total_coded_bits_array,

    pre_fec_errors=pre_fec_errors_array,
    post_fec_errors=post_fec_errors_array,

    wrong_sequences=wrong_sequences_array,
    wrong_blocks=wrong_blocks_array,
    processed_blocks=processed_blocks_array,
    failed_blocks=failed_blocks_array,

    average_calls=average_calls_array,
    average_sweeps=average_sweeps_array,
    average_window_size=(average_window_size_array),

    decode_time_s=decode_time_array,
    time_per_info_bit_s=(time_per_info_bit_array),

    amp_code_ber=amp_code_ber_array,
    sign_info_ber=sign_info_ber_array,
    ccdm_failure_rate=(ccdm_failure_rate_array),
    dm_ber_valid=dm_ber_valid_array,

    source_sequence_fer=(source_sequence_fer_array),
    source_wrong_sequences=(source_wrong_sequences_array),

    amp_code_errors=(amp_code_errors_array),
    sign_info_errors=(sign_info_errors_array),
    ccdm_failures=(ccdm_failures_array),
)

logger.info("Risultati salvati in: %s", output_file)


# # =====================================================================
# # PLOT 1: BER PRE-FEC E POST-FEC
# # =====================================================================

# plt.figure(figsize=(8, 6))

# plt.semilogy(
#     Eb_N0_dB_array,
#     pre_fec_ber_array,
#     marker="o",
#     linestyle="--",
#     label="Pre-FEC PAS 64-QAM",
# )

# # Come nel main FSO: colleghiamo solo le BER realmente misurate > 0.
# positive_mask = ber_array > 0.0

# if np.any(positive_mask):
#     plt.semilogy(
#         Eb_N0_dB_array[positive_mask],
#         ber_array[positive_mask],
#         marker="s",
#         linestyle="-",
#         label="Post-FEC Staircase",
#     )

# # Zero errori: marker separato a 1/N, non un valore BER misurato.
# zero_mask = ber_array == 0.0

# if np.any(zero_mask):
#     zero_display = (
#         1.0
#         / total_info_bits_array[zero_mask]
#     )

#     plt.semilogy(
#         Eb_N0_dB_array[zero_mask],
#         zero_display,
#         marker="v",
#         linestyle="None",
#         label="0 errori osservati (marker a 1/N)",
#     )

# plt.xlabel("Eb/N0 (dB)")
# plt.ylabel("Bit Error Rate")
# plt.title(
#     "Monte Carlo PAS 64-QAM AWGN - LDPC Staircase"
# )
# plt.grid(
#     True,
#     which="both",
#     linestyle=":",
# )
# plt.legend()
# plt.tight_layout()

# plt.savefig(
#     RESULTS_DIR
#     / "BER_PAS_64QAM_AWGN_MonteCarlo_3_sx.png",
#     dpi=200,
# )

# plt.show()


# # =====================================================================
# # PLOT 2: SEQUENCE FER
# # =====================================================================

# plt.figure(figsize=(8, 6))

# positive_fer_mask = (
#     sequence_fer_array > 0.0
# )

# if np.any(positive_fer_mask):
#     plt.semilogy(
#         Eb_N0_dB_array[positive_fer_mask],
#         sequence_fer_array[
#             positive_fer_mask
#         ],
#         marker="o",
#         linestyle="-",
#         label="Sequence FER misurata",
#     )

# zero_fer_mask = (
#     sequence_fer_array == 0.0
# )

# if np.any(zero_fer_mask):
#     upper = (
#         sequence_fer_upper95_array[
#             zero_fer_mask
#         ]
#     )

#     plt.semilogy(
#         Eb_N0_dB_array[zero_fer_mask],
#         upper,
#         marker="v",
#         linestyle="None",
#         label="95% upper bound (0 sequenze fallite)",
#     )

# plt.xlabel("Eb/N0 (dB)")
# plt.ylabel("Sequence FER")
# plt.title(
#     "Probabilità di fallimento della sequenza PAS-Staircase"
# )
# plt.grid(
#     True,
#     which="both",
#     linestyle=":",
# )
# plt.legend()
# plt.tight_layout()

# plt.savefig(
#     RESULTS_DIR
#     / "Sequence_FER_PAS_64QAM_AWGN_MonteCarlo_3_sx.png",
#     dpi=200,
# )

# plt.show()


# # =====================================================================
# # PLOT 3: BLER
# # =====================================================================

# plt.figure(figsize=(8, 6))

# positive_bler_mask = (
#     bler_array > 0.0
# )

# if np.any(positive_bler_mask):
#     plt.semilogy(
#         Eb_N0_dB_array[
#             positive_bler_mask
#         ],
#         bler_array[
#             positive_bler_mask
#         ],
#         marker="o",
#         linestyle="-",
#     )

# plt.xlabel("Eb/N0 (dB)")
# plt.ylabel("Block Error Rate")
# plt.title(
#     "BLER PAS 64-QAM AWGN - LDPC Staircase"
# )
# plt.grid(
#     True,
#     which="both",
#     linestyle=":",
# )
# plt.tight_layout()

# plt.savefig(
#     RESULTS_DIR
#     / "BLER_PAS_64QAM_AWGN_MonteCarlo_3_sx.png",
#     dpi=200,
# )

# plt.show()


# # =====================================================================
# # PLOT 4: COMPLESSITÀ MEDIA
# # =====================================================================

# plt.figure(figsize=(8, 6))

# plt.plot(
#     Eb_N0_dB_array,
#     average_calls_array,
#     marker="o",
#     linestyle="-",
# )

# plt.xlabel("Eb/N0 (dB)")
# plt.ylabel(
#     "Chiamate medie al decoder componente per blocco"
# )
# plt.title(
#     "Complessità media decoder Staircase - PAS"
# )
# plt.grid(
#     True,
#     linestyle=":",
# )
# plt.tight_layout()

# plt.savefig(
#     RESULTS_DIR
#     / "CALLS_PAS_64QAM_AWGN_MonteCarlo_3_sx.png",
#     dpi=200,
# )

# plt.show()


# =====================================================================
# FINE
# =====================================================================

total_simulation_time = (
    time.perf_counter() - simulation_start
)

logger.info("=" * 90)
logger.info(
    "SIMULAZIONE COMPLETATA in %.2f ore",
    total_simulation_time / 3600.0,
)
logger.info(
    "File risultati: %s",
    output_file,
)
logger.info(
    "Directory risultati: %s",
    RESULTS_DIR,
)
logger.info("=" * 90)
