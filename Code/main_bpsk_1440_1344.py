import numpy as np
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.special import erfc
import time
import logging

from AWGN import awgn_channel
from LDPC_Staircase_Encoder import LDPC_Staircase_Encoder
from LDPC_Staircase_Decoder import LDPC_Staircase_Decoder
from ldpc_encoder import ldpc_encode
from spa_minsum_opt import spa_decoder

# Configurazione globale del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# =====================================================================
# DEFINIZIONE DEI PARAMETRI DELLA SCALA 
# =====================================================================
# Codice componente: 2m = 1440, 2m-r = 1344. (1440, 1344) LDPC
m = 720
r = 96
w = 7           # Dimensione sliding window
alpha_c = 0.75  # Fattore scala se SPA converge
alpha_n = 0.375 # Fattore scala se SPA NON converge
v_max = 8       # Iterazioni sliding window normali
v_I_max = 24     # Iterazioni sliding window per il primo blocco (B_1)

# =====================================================================
# Caricamento MATRICI G e H (Forma Sistematica)
# =====================================================================
H_matrix = np.load("H_raw_1440_1344.npy", allow_pickle=True)
H_matrix = sp.csr_matrix(H_matrix)  # Converti in formato sparso
H_matrix_sys = np.load("H_sys_1440_1344.npy", allow_pickle=True)
H_matrix_sys = sp.csr_matrix(H_matrix_sys)  # Converti in formato sparso
G_matrix = np.load("G_sys_1440_1344.npy", allow_pickle=True)

perm = np.load("perm_1440_1344.npy", allow_pickle=True)  # Carica la permutazione


def bpsk_modulate(bits):
    """Modula i bit in simboli BPSK."""
    return 1 - (2 * bits)  # Mappa 0 -> 1, 1 -> -1

def llr_bpsk(received_symbols, noise_variance):
    """Calcola i LLR per la modulazione BPSK."""
    return (2 / noise_variance) * received_symbols

# ========================================================================
# DEFINIZIONE PARAMETRI MODULAZIONE
# ========================================================================

M = 2  # ordine della modulazione BPSK (2-BPSK)
k = int(np.log2(M)) # numero di bit per simbolo
k_axis = k // 2 # numero di bit per asse I e Q


# Istanzia il codificatore
encoder = LDPC_Staircase_Encoder(m, r, G_matrix, ldpc_encode)
decoder = LDPC_Staircase_Decoder(m, r, w, alpha_c, alpha_n, v_max, v_I_max, H_matrix, spa_decoder, spa_max_iter=20, perm=perm)


# Crea un flusso di bit casuale per riempire 122 blocchi
# Info per blocco = 720 * (720 - 96) = 449.280 bit. 
# 122 blocchi di info + 2 blocchi di terminazione = 124 blocchi totali.
# 122 * 449.280 = 54.812.160 bit di info

N_bits = 54812160


Eb_N0_dB_range = np.arange(3.4, 3.5, 0.1)
ber_list = []

for Eb_N0_dB in Eb_N0_dB_range:
    # Sorgente
    bits = np.random.randint(0, 2, N_bits)
    
    # Codifica LDPC Staircase
    encoded_blocks = encoder.encode_sequence(bits, num_termination_blocks=2)

    # Concatenazione dei blocchi codificati in un unico flusso di bit
    bits_stream = np.concatenate([block.ravel() for block in encoded_blocks])
    logger.info(f"Eb/N0 (dB): {Eb_N0_dB}, Sequenza codificata")

    # Calcolo del Code Rate (R)
    R_code = len(bits) / len(bits_stream)
    logger.info(f"Eb/N0 (dB): {Eb_N0_dB}, Code Rate (R): {R_code:.4f}")


    # Modulazione
    mod_data = bpsk_modulate(bits_stream)
    logger.info(f"Eb/N0 (dB): {Eb_N0_dB}, Modulazione BPSK completata")


    # Canale
    received_data, noise_var = awgn_channel(mod_data, Eb_N0_dB, M, R_code)
    received_data = np.real(received_data)  # Prende solo la parte reale per BPSK


    rx_llrs = llr_bpsk(np.real(received_data), noise_var)
    #rx_llrs = 1000 * bpsk_modulate(bits_stream)

    # LDPC STAIRCASE DECODER (Da Probabilità a Bit utente)
    decoded_bits = decoder.decode_sequence(rx_llrs, original_bits = bits, num_termination_blocks=2)


    # Troncamento per eliminare padding e zeri di terminazione
    decoded_bits = decoded_bits[:N_bits]


    # BER CALCULATION
    demodulated_bits = decoded_bits
    cont = np.sum(bits != demodulated_bits)
    ber = cont / N_bits
    ber_list.append(ber)


    logger.info(f"Eb/N0 (dB): {Eb_N0_dB}, Bit Error Rate (BER): {ber:.4e}")


#  Calcolo Curva Teorica
Eb_N0_linear_range = 10 ** (Eb_N0_dB_range / 10)
# Argomento della funzione erfc
arg = np.sqrt( (3 * k * Eb_N0_linear_range) / ((M - 1) * 2) )

# Calcolo BER teorico
ber_teorico = (2 / k) * (1 - 1/np.sqrt(M)) * erfc(arg)

# Filtra i dati per rimuovere i punti non visualizzabili (BER = 0)
ber_array = np.asarray(ber_list)
valid_idx = ber_array > 0
Eb_N0_dB_plot = Eb_N0_dB_range[valid_idx]
ber_plot = ber_array[valid_idx]


plt.figure(figsize=(8, 6))
plt.semilogy(Eb_N0_dB_plot, ber_plot, marker='o', linestyle='-', color='b', label='Simulated BER')
plt.semilogy(Eb_N0_dB_range, ber_teorico, marker='x', linestyle='--', color='r', label='Theoretical BER')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER vs Eb/N0 for 2-BPSK')
plt.grid()
plt.legend()
plt.show()




