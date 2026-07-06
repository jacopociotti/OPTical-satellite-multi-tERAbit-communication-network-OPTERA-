# spa_decoder.py
# Implementazione del decoder SPA per codici LDPC

import numpy as np
from math import atanh

def spa_decoder(llr, H, max_iter=10):   
    """
    Parameters
    ----------
    llr : array_like
        Input LLR values (length N).
    H : sparse matrix or array
        Parity-check matrix (M x N), where M = H.shape[0], N = H.shape[1].
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    decoded_bits : array
        Decoded codeword.
    """

    H = H.tocsr()  # Converte to sparse row format per operazioni veloci
    M, N = H.shape # M = 64, N = 128
  
    # 1. Inizializzazione variable-to-check messages (LLR values).
    # extrinsic_v2c[i, j] = message from variable j to check i
    v2c = np.zeros((M, N))
    for i in range(M):
        variable_indices = H[i].nonzero()[1] #indici delle colonne (variabili) non zero della riga i (check i)
        v2c[i, variable_indices] = llr[variable_indices]
    #print("v2c iniz", v2c)

    for iteration in range(max_iter):
        # 2. ========== Check-to-Variable (C2V) messages ==========
        # c2v[i, j] = message from check i to variable j
        c2v = np.zeros((M, N))
        
        for i in range(M):
            # Ottiene gli indici dei nodi variabili collegati al check i
            var_indices = H[i].nonzero()[1] #indici delle colonne (variabili) non zero della riga i (check i)
            
            for j in var_indices:  # Per ogni variabile j connessa al check i
                prod = 1  # Inizializza il prodotto
                
                for j_primo in var_indices:  # Scansiona tutte le variabili connesse a check i
                    if j_primo != j:  # Escludendo la variabile j stessa
                        prod *= np.tanh(v2c[i, j_primo] / 2.0)  # Moltiplica tanh dei messaggi
                
                # prod può arrivare a ±1 → atanh(±1) = ∞
                prod = np.clip(prod, -0.999999, 0.999999)
                c2v[i, j] = 2 * np.arctanh(prod)
        
        #print("c2v", c2v)
        
        # 3. ========== Variable-to-Check (V2C) messages (update) ==========
        v2c_new = np.zeros((M, N))
        for i in range(M):
            variable_indices = H[i].nonzero()[1] #indici delle colonne (variabili) non zero della riga i (check i)
            v2c_new[i, variable_indices] = llr[variable_indices]
        #print("v2c_new iniz", v2c_new)  
      
        for i in range(M):  # Per ogni check node i
            var_indices = H[i].nonzero()[1]  # Variabili connesse a check i
            
            for j in var_indices:  # Per ogni variabile j connessa a check i
                # Sum c2v messages from all checks EXCEPT i
                for i_primo in range(M):  # Scansiona TUTTI i check node
                    if i_primo != i and H[i_primo, j] != 0:  # Se check i_primo è DIVERSO da i
                                                      # AND è connesso a variabile j
                        v2c_new[i, j] += c2v[i_primo, j]  # Aggiungi il messaggio c2v

        v2c = v2c_new  # Aggiorna i messaggi V2C
        
        # 4. ========== Decode (hard decision) ==========
        # LLR a posteriori di ogni nodo variabile
        post_llr = llr + np.sum(c2v, axis=0)  # Somma tutti i c2v messages che arrivano a ogni variabile
        decoded_bits = (post_llr < 0).astype(int) # array di 0/1, 0 se LLR positivo (bit 0), 1 se LLR negativo (bit 1)
        
        # 5. ========== Check convergence (syndrome = 0) ==========
        syndrome = (H @ decoded_bits) % 2
        converged = np.all(syndrome == 0)
        # if np.all(syndrome == 0):
        #     return decoded_bits
    
    return post_llr, converged 