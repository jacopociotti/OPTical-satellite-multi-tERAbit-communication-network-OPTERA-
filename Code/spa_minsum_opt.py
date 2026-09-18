# spa_opt.py
# Scaled Min-Sum Algorithm (SMSA) - 100% Vettorializzato (Zero Loop Interni)

import numpy as np

def spa_decoder(llr, H, max_iter=10, alpha=0.75):   
    """
    Decoder LDPC Scaled Min-Sum ad altissime prestazioni.
    alpha: Fattore di attenuazione per compensare la sovrastima del Min-Sum.
    """
    H_dense = H.toarray() if hasattr(H, 'toarray') else np.asarray(H, dtype=int)
    M, N = H_dense.shape
    
    rows, cols = np.nonzero(H_dense)
    num_edges = len(rows)
    
    v2c = llr[cols].astype(float)
    c2v = np.zeros(num_edges, dtype=float)
    
    # PRE-ALLOCAZIONE: Evita di allocare memoria ad ogni iterazione.
    # Queste matrici fungeranno da "maschere" per trovare i minimi e i prodotti.
    dense_signs = np.ones((M, N), dtype=float)
    dense_mags = np.full((M, N), np.inf, dtype=float)
    
    for iteration in range(max_iter):
        
        # ==========================================
        # 1. Check-to-Variable (C2V) Update
        # ==========================================
        signs = np.sign(v2c)
        signs[signs == 0] = 1.0
        mags = np.abs(v2c)
        
        # --- Calcolo dei Segni ---
        # Spalmiamo i segni sugli archi attivi, gli inattivi restano a 1
        dense_signs[rows, cols] = signs
        
        # Prodotto totale per ogni check node (moltiplicazione vettoriale su tutta la riga)
        row_signs = np.prod(dense_signs, axis=1)
        
        # Il segno del singolo arco (messaggio in uscita da CN senza includere il segno del messaggio in entrata)
        # è il prodotto totale della sua riga moltiplicato per sé stesso
        # (visto che 1*1=1 e -1*-1=1)
        edge_signs = row_signs[rows] * signs
        
        # --- Calcolo dei Minimi ---
        # Spalmiamo le magnitudini sugli archi attivi, gli inattivi restano a infinito
        dense_mags[rows, cols] = mags
        
        # Troviamo l'INDICE e il VALORE del primo minimo per ogni riga in un colpo solo
        min1_idx = np.argmin(dense_mags, axis=1)
        min1_val = dense_mags[np.arange(M), min1_idx]
        
        # "Nascondiamo" temporaneamente il primo minimo per svelare il secondo
        dense_mags[np.arange(M), min1_idx] = np.inf
        min2_val = np.min(dense_mags, axis=1)
        
        # Decidiamo quale magnitudine assegnare: se l'arco corrisponde alla colonna del minimo, 
        # gli assegniamo il secondo minimo, altrimenti il primo.
        is_min1 = (cols == min1_idx[rows])
        edge_mags = np.where(is_min1, min2_val[rows], min1_val[rows])
        
        # --- Assegnazione C2V con attenuazione (Scaled Min-Sum) ---
        c2v = alpha * edge_signs * edge_mags
        
        
        # ==========================================
        # 2. Variable-to-Check (V2C) Update
        # ==========================================
        sum_c2v = np.zeros(N, dtype=float)
        np.add.at(sum_c2v, cols, c2v)   # prende i valori di c2v e li somma dentro sum_c2v usando gli indici in cols
        
        v2c = llr[cols] + sum_c2v[cols] - c2v
        
        
        # ==========================================
        # 3. Hard Decision e Sindrome
        # ==========================================
        post_llr = llr + sum_c2v
        decoded_bits = (post_llr < 0).astype(int)
        
        syndrome = np.bitwise_and(np.dot(H_dense, decoded_bits), 1)
        if np.sum(syndrome) == 0:
            return post_llr, True
            
    return post_llr, False