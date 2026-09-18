"""
Monte Carlo 64-QAM su canale FSO con LDPC Staircase (1440, 1344).

Canale usato:
    h = sqrt(I_tur * I_pt * I_att) * exp(j*phase)
    y = h*x + n

dove:
    - I_tur: turbolenza Gamma-Gamma;
    - I_pt : pointing loss;
    - I_att: attenuazione atmosferica deterministica;
    - phase: fase a blocchi;
    - n: AWGN complesso.

Il demapper qam_soft_demap_fso() usa perfect CSI (h noto al ricevitore).

Sono disponibili due modalità:

1) SIMULATION_MODE = "exploratory"
   Serve per localizzare rapidamente il waterfall.
   Il punto termina quando:
       - sono state simulate almeno MIN_SEQUENCES;
       - sono state osservate almeno TARGET_FAILED_SEQUENCES sequenze fallite;
   oppure quando si raggiunge MAX_SEQUENCES.

2) SIMULATION_MODE = "fixed"
   Per ogni Eb/N0 vengono simulate esattamente FIXED_SEQUENCES sequenze.
   Non c'è early stopping dipendente dagli errori.

Per ogni punto vengono accumulate:
    - BER pre-FEC;
    - BER post-FEC;
    - BLER;
    - Sequence FER;
    - failure probability interna del decoder;
    - complessità media (calls, sweeps, window);
    - tempo di decodifica;
    - statistiche semplici del canale FSO.

IMPORTANTE:
    Una sequenza è considerata "fallita" se, dopo il decoder Staircase,
    rimane almeno un bit informativo errato.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from FSO_channel import fso_coherent_channel
from llr_mqam import qam_soft_demap_fso
from mod_MQAM_optimized import qam_mod_opt

from Opt_LDPC_Staircase_Encoder import LDPC_Staircase_Encoder
from Opt_LDPC_Staircase_Decoder_syndrome_window_adaptive_2 import (LDPC_Staircase_Decoder)
from ldpc_encoder import ldpc_encode
from spa_minsum_numba import NumbaMinSumDecoder



# LOGGING

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Evita il log blocco-per-blocco durante il Monte Carlo.
logging.getLogger("Opt_LDPC_Staircase_Decoder_syndrome_window_adaptive_2").setLevel(logging.WARNING)
logging.getLogger("LDPC_Staircase_Decoder_syndrome_window_adaptive_2").setLevel(logging.WARNING)



# PARAMETRI LDPC STAIRCASE

# Codice componente LDPC (1440, 1344):
# 2m = 1440
# 2m-r = 1344
m = 720
r = 96

# Decoder Staircase
w = 7
alpha_c = 0.75
alpha_n = 0.375
v_max = 8
v_I_max = 24
spa_max_iter = 20

# Lunghezza di UNA sequenza Monte Carlo.
# Manteniamo la stessa configurazione usata nei test precedenti:
# 122 blocchi informativi + 2 blocchi di terminazione.
num_info_blocks = 122
num_termination_blocks = 2

info_bits_per_block = m * (m - r)
N_bits = num_info_blocks * info_bits_per_block


# PARAMETRI 64-QAM

M = 64
bits_per_symbol = int(np.log2(M))  # 6



# PARAMETRI CANALE FSO

# Turbolenza Gamma-Gamma
FSO_ALPHA = 10.0
FSO_BETA = 5.0

# block_size è espresso in SIMBOLI QAM, non in bit.
#
# Con 64-QAM:
#     1440 simboli * 6 bit/simbolo = 8640 coded bits per fade.
#
# Se invece stessa correlazione rispetto ai coded bits della BPSK con block_size=1440:
#     FSO_BLOCK_SIZE = 240

FSO_BLOCK_SIZE = 240



# PUNTI Eb/N0

Eb_N0_dB_range = np.array([
    13.325,
    13.35,
    13.375,
    13.425,
    13.45,
    13.475,
])



# MODALITÀ MONTE CARLO

# "exploratory" -> early stopping su sequenze fallite
# "fixed"       -> stesso numero di sequenze per ogni Eb/N0
SIMULATION_MODE = "exploratory"


# Modalità esplorativa

MIN_SEQUENCES = 15
TARGET_FAILED_SEQUENCES = 10
MAX_SEQUENCES = 40



# Modalità fixed

FIXED_SEQUENCES = 100



# SEED E RISULTATI

# La sorgente usa Generator indipendente.
SOURCE_SEED = 12345

# FSO_channel.py usa np.random.* internamente.
# Questo seed controlla turbolenza, pointing error, fase e AWGN del canale.
CHANNEL_SEED = 67890

CHECKPOINT_EVERY = 1

RESULTS_DIR = Path("risultati_FSO_64QAM_MonteCarlo_2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# FUNZIONI DI SUPPORTO

def zero_event_upper_bound_95(num_trials: int) -> float:
    """
    Limite superiore unilaterale al 95% quando si osservano zero eventi:

        p_upper = 1 - 0.05^(1/N)

    Per N grande è circa 3/N.

    Lo usiamo soprattutto per la Sequence FER, dove l'unità statistica
    è la sequenza Monte Carlo.
    """
    if num_trials <= 0:
        return np.nan

    return float(-np.expm1(np.log(0.05) / float(num_trials)))


def release_decoder_state(decoder) -> None:
    """
    Libera le grandi memorie create da decode_sequence().

    Alla chiamata successiva decode_sequence() le ricrea automaticamente.
    """
    for name in ("L_I", "L_E_left", "L_E_right"):
        if hasattr(decoder, name):
            setattr(decoder, name, None)


def should_stop_exploratory(sequence_count: int, failed_sequences: int) -> bool:
    """
    Early stopping della modalità esplorativa.

    Fermiamo il punto solamente se:
        sequence_count >= MIN_SEQUENCES
    e
        failed_sequences >= TARGET_FAILED_SEQUENCES
    """
    return (sequence_count >= MIN_SEQUENCES and failed_sequences >= TARGET_FAILED_SEQUENCES)


def save_checkpoint(eb_n0_db: float, sequence_count: int, counters: dict[str, float | int],) -> None:
    """Salva i contatori cumulativi del punto corrente."""
    tag = f"{eb_n0_db:.2f}".replace(".", "p")

    np.savez(RESULTS_DIR / f"checkpoint_{tag}dB_2.npz", Eb_N0_dB=np.float64(eb_n0_db), sequence_count=np.int64(sequence_count), **counters)



# CARICAMENTO MATRICI

logger.info("Caricamento matrici LDPC...")

H_matrix = np.load("prova\\H_raw_1440_1344.npy", allow_pickle=True)
H_matrix = sp.csr_matrix(H_matrix)

G_matrix = np.load("prova\\G_sys_1440_1344.npy", allow_pickle=True)

perm = np.load( "prova\\perm_1440_1344.npy", allow_pickle=True)



# INIZIALIZZAZIONE ENCODER E DECODER

encoder = LDPC_Staircase_Encoder(m, r, G_matrix, ldpc_encode)

component_decoder = NumbaMinSumDecoder(H_matrix, alpha=0.75, dtype=np.float32)

# Warm-up dei kernel Numba prima dei timer Monte Carlo.
component_decoder.warmup_staircase_kernels(m, perm=perm)

decoder = LDPC_Staircase_Decoder(m, r, w, alpha_c, alpha_n, v_max, v_I_max, H_matrix, component_decoder, spa_max_iter=spa_max_iter, perm=perm, track_extrinsic_stats=False)


# GENERATORI CASUALI

source_rng = np.random.default_rng(SOURCE_SEED)

# Il file FSO_channel.py usa il generatore globale NumPy.
np.random.seed(CHANNEL_SEED)



# CONTENITORI RISULTATI

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

mean_irradiance_list = []
min_irradiance_list = []



# CICLO SUI PUNTI Eb/N0

simulation_start = time.perf_counter()

for point_index, Eb_N0_dB in enumerate(Eb_N0_dB_range, start=1):

    logger.info("=" * 90)
    logger.info("PUNTO %d/%d - Eb/N0 = %.3f dB", point_index, len(Eb_N0_dB_range), Eb_N0_dB)
    logger.info("Modalità Monte Carlo: %s", SIMULATION_MODE)

    if SIMULATION_MODE == "exploratory":
        logger.info("Stop esplorativo: min_seq=%d, target_failed_seq=%d, max_seq=%d", MIN_SEQUENCES, TARGET_FAILED_SEQUENCES, MAX_SEQUENCES)
        max_trials_this_point = MAX_SEQUENCES

    elif SIMULATION_MODE == "fixed":
        logger.info("Numero fisso di sequenze: %d", FIXED_SEQUENCES)
        max_trials_this_point = FIXED_SEQUENCES

    else:
        raise ValueError('SIMULATION_MODE deve essere "exploratory" oppure "fixed".')

    logger.info("=" * 90)


    # Contatori cumulativi del punto

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

    # Statistiche semplici sul canale FSO
    cumulative_irradiance_sum = 0.0
    cumulative_irradiance_samples = 0
    global_min_irradiance = np.inf

    point_start = time.perf_counter()


    # MONTE CARLO DEL PUNTO

    while sequence_count < max_trials_this_point:

        # In modalità esplorativa controlliamo l'early stopping
        # PRIMA di iniziare una nuova sequenza.
        if (SIMULATION_MODE == "exploratory" and should_stop_exploratory(sequence_count, total_wrong_sequences)):
            stop_reason = ("target di sequenze fallite raggiunto dopo il minimo di sequenze")
            break

        trial_number = sequence_count + 1
        trial_start = time.perf_counter()

        logger.info("Eb/N0 %.2f dB - sequenza Monte Carlo %d", Eb_N0_dB, trial_number)


        # 1. SORGENTE

        bits = source_rng.integers(0, 2, N_bits, dtype=np.uint8)


        # 2. ENCODER STAIRCASE

        # encode_sequence() resetta già B0, ma lo rendiamo esplicito.
        encoder.reset()

        if np.count_nonzero(encoder.B_prev) != 0:
            raise RuntimeError("B0 non è stato azzerato correttamente.")

        encode_start = time.perf_counter()
        encoded_blocks = encoder.encode_sequence(bits, num_termination_blocks=num_termination_blocks)
        encode_time = time.perf_counter() - encode_start

        # View senza copia.
        bits_stream = encoded_blocks.reshape(-1)

        R_code = N_bits / bits_stream.size


        # 3. MODULAZIONE 64-QAM

        mod_start = time.perf_counter()
        mod_data = qam_mod_opt(bits_stream, M,)
        mod_time = time.perf_counter() - mod_start


        # 4. CANALE FSO

        channel_start = time.perf_counter()
        received_data, h, I_tot, noise_var = fso_coherent_channel(mod_data, Eb_N0_dB, M, R_code, alpha=FSO_ALPHA, beta=FSO_BETA, block_size=FSO_BLOCK_SIZE)
        channel_time = time.perf_counter() - channel_start

        # Statistiche della realizzazione FSO
        cumulative_irradiance_sum += float(np.sum(I_tot, dtype=np.float64))
        cumulative_irradiance_samples += int(I_tot.size)

        seq_min_irradiance = float(np.min(I_tot))
        global_min_irradiance = min(global_min_irradiance, seq_min_irradiance)


        # 5. SOFT DEMAPPER 64-QAM CON PERFECT CSI

        demap_start = time.perf_counter()
        rx_llrs = qam_soft_demap_fso(received_data, h, M, noise_var)
        rx_llrs = np.asarray(rx_llrs, dtype=np.float32)
        demap_time = time.perf_counter() - demap_start


        # 6. BER PRE-FEC

        hard_bits_pre = rx_llrs < 0.0
        pre_fec_errors = int(np.count_nonzero(hard_bits_pre != bits_stream))


        # 7. DECODER LDPC STAIRCASE

        decode_start = time.perf_counter()
        decoded_bits = decoder.decode_sequence(rx_llrs, original_bits=bits, num_termination_blocks=num_termination_blocks)
        measured_decode_time = (time.perf_counter() - decode_start)

        decoded_bits = decoded_bits[:N_bits]

        post_fec_errors = int(np.count_nonzero(bits != decoded_bits))

        # Una sequenza fallisce se rimane almeno un bit errato.
        sequence_failed = int(post_fec_errors > 0)

        metrics = dict(decoder.last_metrics)

        processed_blocks = int(metrics["processed_blocks"])
        wrong_blocks = int(metrics["wrong_blocks"])
        failed_blocks = int(metrics.get("failed_blocks", 0))

        component_calls = int(metrics["component_decoder_calls"])

        average_sweeps = float(metrics["average_sweeps"])
        average_window = float(metrics["average_window_size"])

        decode_time = float(metrics.get("total_decode_time", measured_decode_time))


        # 8. AGGIORNAMENTO CONTATORI

        sequence_count += 1

        total_info_bits += N_bits
        total_coded_bits += bits_stream.size

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


        # 9. STIME CUMULATIVE

        pre_fec_ber_partial = (total_pre_fec_errors / total_coded_bits)
        ber_partial = (total_post_fec_errors / total_info_bits)
        bler_partial = (total_wrong_blocks / total_processed_blocks if total_processed_blocks > 0 else 0.0)
        sequence_fer_partial = (total_wrong_sequences / sequence_count)
        trial_time = (time.perf_counter() - trial_start)

        logger.info("Seq %d conclusa in %.2f min | encoder %.2fs | mod %.2fs | FSO %.2fs | demap %.2fs | decode %.2fs",
            sequence_count,
            trial_time / 60.0,
            encode_time,
            mod_time,
            channel_time,
            demap_time,
            decode_time,
        )

        logger.info("Seq %d: pre-FEC errors=%d | post-FEC errors=%d | wrong blocks=%d/%d | failed=%s",
            sequence_count,
            pre_fec_errors,
            post_fec_errors,
            wrong_blocks,
            processed_blocks,
            bool(sequence_failed),
        )

        logger.info("Canale seq %d: sigma2=%.6e | mean(I_tot)=%.6e | min(I_tot)=%.6e",
            sequence_count,
            float(noise_var),
            float(np.mean(I_tot)),
            seq_min_irradiance,
        )

        logger.info("CUMULATIVO: pre-BER=%.6e | BER=%.6e | BLER=%.6e | Sequence FER=%.6e (%d/%d)",
            pre_fec_ber_partial,
            ber_partial,
            bler_partial,
            sequence_fer_partial,
            total_wrong_sequences,
            sequence_count,
        )


        # 10. CHECKPOINT

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
                },
            )


        # 11. LIBERAZIONE MEMORIA

        del decoded_bits
        del metrics

        release_decoder_state(decoder)

        del hard_bits_pre
        del rx_llrs

        del received_data
        del h
        del I_tot
        del mod_data

        del bits_stream
        del encoded_blocks
        del bits

        gc.collect()

    else:
        if SIMULATION_MODE == "fixed":
            stop_reason = (f"completate {FIXED_SEQUENCES} sequenze fisse")
        else:
            stop_reason = (f"raggiunto MAX_SEQUENCES={MAX_SEQUENCES}")


    # RISULTATI FINALI DEL PUNTO

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
    mean_irradiance = (cumulative_irradiance_sum / cumulative_irradiance_samples)

    # Upper bound 95% della Sequence FER quando non osserviamo alcuna sequenza fallita.
    if total_wrong_sequences == 0:
        sequence_fer_upper95 = (zero_event_upper_bound_95( sequence_count))
    else:
        sequence_fer_upper95 = np.nan

    # Salvataggio nelle liste finali
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

    mean_irradiance_list.append(mean_irradiance)
    min_irradiance_list.append(global_min_irradiance)

    logger.info("=" * 90)
    logger.info("FINE Eb/N0 %.2f dB: %s", Eb_N0_dB, stop_reason)
    logger.info("Sequenze=%d | tempo punto=%.2f h", sequence_count, point_time / 3600.0)
    logger.info("Pre-FEC BER = %.6e (%d errori)", pre_fec_ber, total_pre_fec_errors)
    logger.info("Post-FEC BER = %.6e (%d errori)", ber, total_post_fec_errors)
    logger.info("BLER = %.6e (%d/%d blocchi)", bler, total_wrong_blocks, total_processed_blocks)
    logger.info("Sequence FER = %.6e (%d/%d sequenze)", sequence_fer, total_wrong_sequences, sequence_count)

    if total_wrong_sequences == 0:
        logger.info("Zero sequenze fallite: upper bound 95%% Sequence FER = %.6e", sequence_fer_upper95)

    logger.info("Failure probability decoder = %.6e", failure_probability)
    logger.info("Average calls/block = %.2f | avg sweeps = %.3f | avg window = %.3f", average_calls, average_sweeps_final, average_window_final)
    logger.info("mean(I_tot)=%.6e | min(I_tot)=%.6e", mean_irradiance, global_min_irradiance)
    logger.info("=" * 90)


    # Salvataggio progressivo della curva

    np.savez(
        RESULTS_DIR / "FSO_64QAM_MonteCarlo_progressivo2.npz",
        Eb_N0_dB=np.asarray(Eb_N0_dB_range[:len(ber_list)], dtype=float),
        pre_fec_ber=np.asarray(pre_fec_ber_list, dtype=float),
        ber=np.asarray(ber_list, dtype=float),
        bler=np.asarray(bler_list, dtype=float),
        sequence_fer=np.asarray(sequence_fer_list, dtype=float),
        sequence_fer_upper95=np.asarray(sequence_fer_upper95_list, dtype=float),
        failure_probability=np.asarray(failure_probability_list, dtype=float),
        num_sequences=np.asarray(num_sequences_list, dtype=np.int64),
        total_info_bits=np.asarray(total_info_bits_list, dtype=np.int64),
        total_coded_bits=np.asarray(total_coded_bits_list, dtype=np.int64),
        pre_fec_errors=np.asarray(pre_fec_errors_list, dtype=np.int64),
        post_fec_errors=np.asarray(post_fec_errors_list, dtype=np.int64),
        wrong_sequences=np.asarray(wrong_sequences_list, dtype=np.int64),
        wrong_blocks=np.asarray(wrong_blocks_list, dtype=np.int64),
        processed_blocks=np.asarray(processed_blocks_list, dtype=np.int64),
        failed_blocks=np.asarray(failed_blocks_list, dtype=np.int64),
        average_calls=np.asarray(average_calls_list, dtype=float),
        average_sweeps=np.asarray(average_sweeps_list, dtype=float),
        average_window_size=np.asarray(average_window_size_list, dtype=float),
        decode_time_s=np.asarray(decode_time_list, dtype=float),
        time_per_info_bit_s=np.asarray(time_per_info_bit_list, dtype=float),
        mean_irradiance=np.asarray(mean_irradiance_list, dtype=float),
        min_irradiance=np.asarray(min_irradiance_list, dtype=float),
    )



# CONVERSIONE FINALE IN ARRAY

Eb_N0_dB_array = np.asarray(Eb_N0_dB_range, dtype=float)
pre_fec_ber_array = np.asarray(pre_fec_ber_list, dtype=float)
ber_array = np.asarray(ber_list, dtype=float)
bler_array = np.asarray(bler_list, dtype=float)
sequence_fer_array = np.asarray(sequence_fer_list, dtype=float)
sequence_fer_upper95_array = np.asarray(sequence_fer_upper95_list, dtype=float)
failure_probability_array = np.asarray(failure_probability_list, dtype=float)
num_sequences_array = np.asarray(num_sequences_list, dtype=np.int64)
total_info_bits_array = np.asarray(total_info_bits_list, dtype=np.int64)
total_coded_bits_array = np.asarray(total_coded_bits_list, dtype=np.int64)
pre_fec_errors_array = np.asarray(pre_fec_errors_list, dtype=np.int64)
post_fec_errors_array = np.asarray(post_fec_errors_list, dtype=np.int64)
wrong_sequences_array = np.asarray(wrong_sequences_list, dtype=np.int64)
wrong_blocks_array = np.asarray(wrong_blocks_list, dtype=np.int64)
processed_blocks_array = np.asarray(processed_blocks_list, dtype=np.int64)
failed_blocks_array = np.asarray(failed_blocks_list, dtype=np.int64)
average_calls_array = np.asarray(average_calls_list, dtype=float)
average_sweeps_array = np.asarray(average_sweeps_list, dtype=float)
average_window_size_array = np.asarray(average_window_size_list, dtype=float)
decode_time_array = np.asarray(decode_time_list, dtype=float)
time_per_info_bit_array = np.asarray(time_per_info_bit_list, dtype=float)
mean_irradiance_array = np.asarray(mean_irradiance_list, dtype=float)
min_irradiance_array = np.asarray(min_irradiance_list, dtype=float)



# SALVATAGGIO FINALE

output_file = (RESULTS_DIR / f"FSO_64QAM_MonteCarlo_122_blocchi2_{SIMULATION_MODE}.npz")

np.savez(
    output_file,
    Eb_N0_dB=Eb_N0_dB_array,

    M=np.int64(M),
    bits_per_symbol=np.int64(bits_per_symbol),

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

    FSO_alpha=np.float64(FSO_ALPHA),
    FSO_beta=np.float64(FSO_BETA),
    FSO_block_size=np.int64(FSO_BLOCK_SIZE),

    simulation_mode=np.array(SIMULATION_MODE),

    min_sequences=np.int64(MIN_SEQUENCES),
    target_failed_sequences=np.int64(TARGET_FAILED_SEQUENCES),
    max_sequences=np.int64(MAX_SEQUENCES),
    fixed_sequences=np.int64(FIXED_SEQUENCES),

    source_seed=np.int64(SOURCE_SEED),
    channel_seed=np.int64(CHANNEL_SEED),

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

    mean_irradiance=mean_irradiance_array,
    min_irradiance=min_irradiance_array,
)

logger.info("Risultati salvati in: %s", output_file)



# PLOT 1: BER PRE-FEC E POST-FEC

plt.figure(figsize=(8, 6))

plt.semilogy(Eb_N0_dB_array, pre_fec_ber_array, marker="o", linestyle="--", label="Pre-FEC 64-QAM")

# BER post-FEC realmente misurate (>0)
positive_mask = ber_array > 0.0

if np.any(positive_mask):
    plt.semilogy(Eb_N0_dB_array[positive_mask], ber_array[positive_mask], marker="s", linestyle="-", label="Post-FEC Staircase")

# Nei punti con zero errori NON fingiamo di aver misurato una BER.
# Li mostriamo a 1/N solamente come riferimento grafico, con marker separato e senza collegarli alla curva BER.
zero_mask = ber_array == 0.0

if np.any(zero_mask):

    zero_display = (1.0 / total_info_bits_array[zero_mask])

    plt.semilogy(Eb_N0_dB_array[zero_mask], zero_display, marker="v", linestyle="None", label="0 errori osservati (marker a 1/N)")

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate")
plt.title("Monte Carlo 64-QAM FSO - LDPC Staircase")
plt.grid(True, which="both", linestyle=":")
plt.legend()
plt.tight_layout()

plt.savefig(RESULTS_DIR / "BER_64QAM_FSO_MonteCarlo.png", dpi=200)

plt.show()



# PLOT 2: SEQUENCE FER

plt.figure(figsize=(8, 6))

positive_fer_mask = (sequence_fer_array > 0.0)

if np.any(positive_fer_mask):
    plt.semilogy(Eb_N0_dB_array[positive_fer_mask], sequence_fer_array[positive_fer_mask], marker="o", linestyle="-", label="Sequence FER misurata")

zero_fer_mask = (sequence_fer_array == 0.0)

if np.any(zero_fer_mask):

    upper = (sequence_fer_upper95_array[zero_fer_mask])

    plt.semilogy(Eb_N0_dB_array[zero_fer_mask], upper, marker="v", linestyle="None", label=("95% upper bound (0 sequenze fallite)"))

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Sequence FER")
plt.title("Probabilità di fallimento della sequenza Staircase")
plt.grid(True, which="both", linestyle=":")
plt.legend()
plt.tight_layout()

plt.savefig(RESULTS_DIR / "Sequence_FER_64QAM_FSO_MonteCarlo.png", dpi=200)

plt.show()



# PLOT 3: BLER

plt.figure(figsize=(8, 6))

positive_bler_mask = (bler_array > 0.0)

if np.any(positive_bler_mask):
    plt.semilogy(Eb_N0_dB_array[positive_bler_mask], bler_array[positive_bler_mask], marker="o", linestyle="-")

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Block Error Rate")
plt.title("BLER 64-QAM FSO - LDPC Staircase")
plt.grid(True, which="both", linestyle=":")
plt.tight_layout()

plt.savefig(RESULTS_DIR / "BLER_64QAM_FSO_MonteCarlo.png", dpi=200)

plt.show()



# PLOT 4: COMPLESSITÀ MEDIA

plt.figure(figsize=(8, 6))

plt.plot(Eb_N0_dB_array, average_calls_array, marker="o", linestyle="-")

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Chiamate medie al decoder componente per blocco")
plt.title("Complessità media decoder Staircase")
plt.grid(True, linestyle=":")
plt.tight_layout()

plt.savefig(RESULTS_DIR / "CALLS_64QAM_FSO_MonteCarlo.png", dpi=200)

plt.show()



# FINE

total_simulation_time = (time.perf_counter() - simulation_start)

logger.info("=" * 90)
logger.info("SIMULAZIONE COMPLETATA in %.2f ore", total_simulation_time / 3600.0)
logger.info("File risultati: %s", output_file)
logger.info("=" * 90)
