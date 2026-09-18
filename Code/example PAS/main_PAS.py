# Test con PAS, LDPC (1440, 1344) e CCDM quaternario per 64-QAM

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from AWGN_PAS import awgn_channel
from mod_MQAM_optimized import qam_mod_opt
from demod_MQAM_optimized import qam_demod_opt
from ldpc_encoder import ldpc_encode
from spa_minsum_opt import spa_decoder
from soft_demap_PAS import qam_soft_demap

from pas64_utils import (
    pas64_codeword_to_qam_bits,
    pas64_qam_llrs_to_codeword_order,
    amplitudes_to_pas64_amp_bits,
    pas64_amp_bits_to_amplitudes,
)

from ccdm_4ary import ( 
    ccdm_4ary_encode,
    ccdm_4ary_decode, 
    ccdm_4ary_k_from_composition,
)

H_matrix = np.load("H_raw_1440_1344.npy", allow_pickle=True)
H_matrix = sp.csr_matrix(H_matrix)  # Converti in formato sparso
H_matrix_sys = np.load("H_sys_1440_1344.npy", allow_pickle=True)
H_matrix_sys = sp.csr_matrix(H_matrix_sys)  # Converti in formato sparso
G_matrix = np.load("G_sys_1440_1344.npy", allow_pickle=True)


M = 64

K, N = G_matrix.shape

if N % 6 != 0:
    raise ValueError("Per 64-QAM serve N multiplo di 6.")

n_symbols = N // 6
n_axes = 2 * n_symbols
n_amp_bits = 4 * n_symbols
n_extra_info_sign_bits = K - n_amp_bits

composition = [240, 120, 80, 40]  # Esempio di composizione per CCDM quaternario

amplitude_alphabet = np.array([1, 3, 5, 7])

def valid_composition(amplitudes, target_composition):
    amplitudes = np.asarray(amplitudes)

    observed_composition = np.array([
        np.count_nonzero(amplitudes == amplitude)
        for amplitude in amplitude_alphabet
    ])

    return np.array_equal(
        observed_composition,
        np.asarray(target_composition)
    )

k_dm = ccdm_4ary_k_from_composition(composition)

Eb_N0_dB_range = [8, 10, 12, 14, 15, 16, 18, 20]  # Esempio di Eb/N0 in dB
N_frames = 1000  # Numero di frame da simulare per ogni Eb/N0

ber_info_list = [] # BER post LDPC sui K bit informativi sistematici
ber_amp_list = [] # BER post LDPC sui bit di ampiezza (4 bit per simbolo)
ber_sign_list = [] # BER post LDPC sui bit informativi extra nei segni
fer_list = [] # FER post LDPC sui bit sistematici (frame error rate)
cer_list = [] # CER post LDPC sui N bit della codeword

dm_failure_rate_list = []
ber_dm_cond_list = [] # BER condizionale sul CCDM (solo quando il CCDM ha successo)
fer_dm_cond_list = [] # FER condizionale sul CCDM (solo quando il CCDM ha successo)

ber_pas_cond_list = [] # BER condizionale sul PAS (solo quando il CCDM ha successo)
fer_pas_cond_list = [] # FER condizionale sul PAS (solo quando il CCDM ha successo)

fer_pas_total_list = []


for Eb_N0_dB in Eb_N0_dB_range:  

    total_info_errors = 0
    total_amp_errors = 0
    total_codeword_errors = 0
    total_sign_errors = 0
    frame_errors = 0


    dm_failures = 0
    dm_valid_frames = 0

    dm_bit_errors_valid = 0
    dm_frame_errors_valid = 0

    pas_bit_errors_valid = 0
    pas_frame_errors_valid = 0

    ldpc_declared_failures = 0

    for frame in range(N_frames):


        u = np.random.randint(0, 2, k_dm)  # Bit uniformi per il CCDM

        # CCDM encoding
        amplitudes = ccdm_4ary_encode(u, composition)

        if K < n_amp_bits:
            raise ValueError("Rate LDPC troppo basso per PAS 64-QAM standard.")

        # Converte le ampiezze nei 2 bit di ampiezza per PAS 64-QAM es. 1 -> 10, 3 -> 11, 5 -> 01, 7 -> 00
        amp_bits = amplitudes_to_pas64_amp_bits(amplitudes)



        # Calcola degli eventuali bit informativi extra che andranno nei segni
        n_extra_info_sign_bits = K - n_amp_bits
        sign_info_bits = np.random.randint(0, 2, n_extra_info_sign_bits)



        # Input sistematico LDPC
        info_bits = np.concatenate([amp_bits, sign_info_bits])

        # Encoding LDPC sistematico
        codeword = ldpc_encode(info_bits, G_matrix).astype(int)

        # Riordina PAS per 64-QAM
        qam_bits = pas64_codeword_to_qam_bits(codeword, K) # i bit sono nella forma (I0_sign, I0_amp1, I0_amp2, Q0_sign, Q0_amp1, Q0_amp2, I1_sign, I1_amp1, I1_amp2, Q1_sign, Q1_amp1, Q1_amp2,...)

        #  Modulazione 64-QAM
        tx_symbols = qam_mod_opt(qam_bits, M)


        R_pas = (k_dm + n_extra_info_sign_bits) / N  # Rate effettivo del sistema PAS 64-QAM

        # Canale AWGN
        rx_symbols, noise_var = awgn_channel(tx_symbols, Eb_N0_dB, M, R_pas)

        # Demodulazione soft
        llr = qam_soft_demap(rx_symbols, M, noise_var, composition)

        # Riordina gli LLR nello stesso ordine della codeword LDPC
        llr_codeword_order = pas64_qam_llrs_to_codeword_order(llr, N, K) # riordina dalla forma (I0_sign, I0_amp1, I0_amp2, Q0_sign, Q0_amp1, Q0_amp2, I1_sign, I1_amp1, I1_amp2, Q1_sign, Q1_amp1, Q1_amp2,...) alla forma (amp_bits | sign_info_bits | parity_bits)

        # Decoder LDPC
        post_llr, success = spa_decoder(llr_codeword_order, H_matrix_sys, max_iter=20)

        decoded_codeword = (post_llr < 0).astype(int)
        # Recupero bit informativi
        decoded_info_bits = decoded_codeword[:K]
        # I bit di ampiezza recuperati sono quelli da dare al CCDM inverso
        decoded_amp_bits = decoded_info_bits[:n_amp_bits]
        decoded_sign_info_bits = decoded_info_bits[n_amp_bits:K]

        cw_errors = np.sum(codeword != decoded_codeword)
        info_errors = np.sum(info_bits != decoded_info_bits)
        amp_errors = np.sum(amp_bits != decoded_amp_bits)
        sign_errors = np.sum(sign_info_bits != decoded_sign_info_bits)

        total_codeword_errors += cw_errors
        total_info_errors += info_errors
        total_amp_errors += amp_errors
        total_sign_errors += sign_errors

        if info_errors > 0:
            frame_errors += 1 # ogni frame con almeno un errore informativo nei K bit viene contato come frame in errore
        

        if not success:
            ldpc_declared_failures += 1

        # Conversione dei bit di ampiezza stimati nelle ampiezze
        try:
            decoded_amplitudes = pas64_amp_bits_to_amplitudes(decoded_amp_bits)
        except ValueError: # salta al frame successivo se non è stato possibile ottenere una sequenza di ampiezze
            # Non è stato possibile ottenere una sequenza di ampiezze
            dm_failures += 1
            continue

        # Controllo della composizione
        if not valid_composition(decoded_amplitudes, composition):
            dm_failures += 1
            continue

        # La composizione è valida: provo l'inverse CCDM
        try:
            u_hat = ccdm_4ary_decode(decoded_amplitudes,composition)
            u_hat = np.asarray(u_hat, dtype=int)

        except ValueError:
            # Composizione formalmente valida, ma sequenza non
            # decodificabile, ad esempio rank fuori dal sottoinsieme usato
            dm_failures += 1
            continue


        # Da qui in poi il CCDM inverso ha prodotto un'uscita
        dm_valid_frames += 1


        # Errori sui bit originari entrati nel CCDM
        dm_errors = np.sum(u != u_hat)

        dm_bit_errors_valid += dm_errors

        if dm_errors > 0:
            dm_frame_errors_valid += 1


        # Errori informativi PAS complessivi:
        # bit CCDM + bit informativi di segno
        pas_errors = dm_errors + sign_errors

        pas_bit_errors_valid += pas_errors

        if pas_errors > 0:
            pas_frame_errors_valid += 1

      



    ber_info = total_info_errors / (N_frames * K) # BER post LDPC sui K bit informativi sistematici
    ber_amp = total_amp_errors / (N_frames * n_amp_bits) # BER post LDPC sui bit di ampiezza (4 bit per simbolo)
    ber_sign = total_sign_errors / (N_frames * n_extra_info_sign_bits) # BER post LDPC sui bit informativi extra nei segni
    fer = frame_errors / N_frames # FER post LDPC sui bit sistematici (frame error rate)
    cer = total_codeword_errors / (N_frames * N) # CER post LDPC sui N bit della codeword

    dm_failure_rate = dm_failures / N_frames
    if dm_valid_frames > 0:

        ber_dm_cond = (dm_bit_errors_valid / (dm_valid_frames * k_dm))
        fer_dm_cond = (dm_frame_errors_valid / dm_valid_frames)
        ber_pas_cond = (pas_bit_errors_valid / (dm_valid_frames * (k_dm + n_extra_info_sign_bits)))
        fer_pas_cond = (pas_frame_errors_valid / dm_valid_frames)

    else:
        ber_dm_cond = np.nan
        fer_dm_cond = np.nan
        ber_pas_cond = np.nan
        fer_pas_cond = np.nan

    fer_pas_total = (dm_failures + pas_frame_errors_valid) / N_frames
    

    ber_info_list.append(ber_info)
    ber_amp_list.append(ber_amp)
    ber_sign_list.append(ber_sign)
    fer_list.append(fer)
    cer_list.append(cer)

    dm_failure_rate_list.append(dm_failure_rate)
    ber_dm_cond_list.append(ber_dm_cond)
    fer_dm_cond_list.append(fer_dm_cond)
    ber_pas_cond_list.append(ber_pas_cond)
    fer_pas_cond_list.append(fer_pas_cond)
    fer_pas_total_list.append(fer_pas_total)

    print(f"Eb/N0 (dB): {Eb_N0_dB} dB")
    print(f"BER info: {ber_info:.6e}")
    print(f"BER amp: {ber_amp:.6e}")
    print(f"BER sign: {ber_sign:.6e}")
    print(f"FER: {fer:.6e}")
    print(f"CER: {cer:.6e}")
    print(f"DM failure rate: "f"{dm_failure_rate:.6e} "f"({dm_failures}/{N_frames})")
    print(f"DM valid frames: "f"{dm_valid_frames}/{N_frames}")
    print(f"BER DM | valid: "f"{ber_dm_cond:.6e}")
    print(f"FER DM | valid: "f"{fer_dm_cond:.6e}")
    print(f"BER PAS | valid: "f"{ber_pas_cond:.6e}")
    print(f"FER PAS | valid: "f"{fer_pas_cond:.6e}")
    print(f"FER PAS totale: "f"{fer_pas_total:.6e}")
    print(f"LDPC declared failures: "f"{ldpc_declared_failures}/{N_frames}")



ber_info = np.array(ber_info_list) 
ber_amp = np.array(ber_amp_list) 
ber_sign = np.array(ber_sign_list) 
fer = np.array(fer_list) 
cer = np.array(cer_list)
ber_dm = np.array(ber_dm_cond_list)
ber_pas_dm = np.array(ber_pas_cond_list)
Eb_N0_dB_range = np.array(Eb_N0_dB_range)

def remove_invalid_entries(metric):
    """
    Mantiene la stessa lunghezza della metrica, ma sostituisce:
    - valori uguali a zero;
    - valori negativi;
    - inf;
    - NaN

    con np.nan, che non viene disegnato nel grafico semilogaritmico.
    """
    metric = np.asarray(metric, dtype=float).copy()

    invalid = (metric <= 0) | (~np.isfinite(metric))
    metric[invalid] = np.nan

    return metric
                          


ber_info = remove_invalid_entries(ber_info)
ber_amp = remove_invalid_entries(ber_amp)
ber_sign = remove_invalid_entries(ber_sign)
fer = remove_invalid_entries(fer)
cer = remove_invalid_entries(cer)
ber_dm = remove_invalid_entries(ber_dm)
ber_pas_dm = remove_invalid_entries(ber_pas_dm)
Eb_N0_dB_range_plot = Eb_N0_dB_range

plt.figure(figsize=(8, 6))
plt.semilogy(Eb_N0_dB_range_plot, ber_info, marker='o', linestyle='-', color='r', label='BER info')
plt.semilogy(Eb_N0_dB_range_plot, ber_amp, marker='s', linestyle='--', color='g', label='BER amp')
plt.semilogy(Eb_N0_dB_range_plot, ber_sign, marker='^', linestyle=':', color='c', label='BER sign')
plt.semilogy(Eb_N0_dB_range_plot, fer, marker='x', linestyle='-', color='b', label='FER')
plt.semilogy(Eb_N0_dB_range_plot, cer, marker='d', linestyle='-.', color='m', label='CER')
plt.semilogy(Eb_N0_dB_range_plot, ber_dm, marker='*', linestyle='-', color='k', label='BER DM | valid')
plt.semilogy(Eb_N0_dB_range_plot, ber_pas_dm, marker='p', linestyle='--', color='y', label='BER PAS | valid')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER, FER, and CER vs Eb/N0 for PAS 64-QAM with LDPC (1440, 1344) and CCDM')
plt.grid()
plt.legend()
plt.show()