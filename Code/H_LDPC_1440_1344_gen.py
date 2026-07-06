import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def generate_H_1440_1344():
    """
    Genera la matrice di parità H (96 x 1440) per il codice LDPC (1440, 1344) 
    dello standard IEEE 802.15.3c
    """
    N = 1440
    M = 96
    p = 15  # Periodo delle sottomatrici
    
    # Inizializza la matrice H tutta a zeri
    H = np.zeros((M, N), dtype=int)
    
    # Definisce gli indici di riga per le prime 15 colonne (i '1' della sottomatrice base)
    base_indices = {
        0: [0, 1, 4],       # colonna 0 righe 0, 1, 4
        1: [32, 34, 39],
        2: [64, 70, 78],
        3: [8, 18, 95],
        4: [31, 42, 54],
        5: [63, 76, 91],
        6: [14, 45, 94],
        7: [30, 47, 83],
        8: [17, 62, 80],
        9: [28, 48, 82],
        10: [22, 60, 81],
        11: [27, 49, 84],
        12: [7, 53, 77],
        13: [19, 44, 85],
        14: [6, 46, 75]
    }
    
    # Popola l'intera matrice 96x1440, itera per colonne
    for j in range(N):
        # Trova la colonna base (da 0 a 14)
        j_base = j % p  # operazione modulo p=15, resto della divisione intera, (sarà sempre un intero tra 0 e 14)
        
        # Calcola lo shift verso l'alto delle colonne
        shift = j // p  # quoziente della divisione intera tra j e p, numero di blocchi completi di 15 colonne trascorsi fino alla colonna j
        
        # Per ogni riga in cui c'era un 1 nella colonna base, j_base=0 -> r_base prende i valori del dizionario 0, 1, 4
        for r_base in base_indices[j_base]:
            # applica lo spostamento ciclico (modulo 96)
            r_new = (r_base - shift) % M
            H[r_new, j] = 1
            
    # Restituisce sia il formato denso che quello sparso (csr) per l'algoritmo SPA
    return H, sp.csr_matrix(H)


def gauss_jordan(A):
    """Riduzione di Gauss-Jordan in GF(2) per trovare i pivot."""
    A = A.copy().astype(int) % 2
    m, n = A.shape
    pivots = []
    row = 0
    for col in range(n):
        if row >= m: break
        
        # Cerca pivot
        pivot_rows = np.where(A[row:m, col] == 1)[0]
        if pivot_rows.size == 0: continue
        
        pivot = row + pivot_rows[0]
        A[[row, pivot]] = A[[pivot, row]] # Scambio righe
        
        # Elimina le altre righe
        for i in range(m):
            if i != row and A[i, col] == 1:
                A[i] ^= A[row]
        pivots.append(col)
        row += 1
    return A, pivots

def make_systematic(H_raw):
    """Trasforma H grezza in sistematica [P^T | I] e genera G = [I | P]."""
    # Riduci la matrice per trovare i pivot
    R, pivots = gauss_jordan(H_raw)
    m,n = H_raw.shape
    
    # Colonne libere (Informazione) e Pivot (Parità)
    free_cols = [i for i in range(n) if i not in pivots]
    k = len(free_cols)
    
    # Vettore di permutazione: Info a sinistra, Parità a destra
    perm = free_cols + pivots
    
    H_sys = R[:, perm]

    # verifica
    I = H_sys[:, -m:]

    if not np.array_equal(I, np.eye(m, dtype=np.uint8)):
        raise RuntimeError("La matrice NON è in forma sistematica.")

    Pt = H_sys[:, :k]
    P = Pt.T
    G_sys = np.concatenate((np.eye(k, dtype=np.uint8), P), axis=1)     
    
    return H_sys, G_sys, perm




# TEST
if __name__ == "__main__":
    r = 96  # Numero di righe (parità)
    N = 1440 # Numero di colonne (lunghezza codeword)
    
    H_raw = generate_H_1440_1344()[0]  # Ottieni la matrice H in formato sparso
    H_sys, G_sys, perm = make_systematic(H_raw)

    np.save("H_raw_1440_1344.npy", H_raw)
    np.save("H_sys_1440_1344.npy", H_sys)
    np.save("G_sys_1440_1344.npy", G_sys)
    np.save("perm_1440_1344.npy", perm)

    print("H:", H_raw.shape)
    print("Hsys:", H_sys.shape)
    print("G:", G_sys.shape)


    print("Verifica ortogonalità")

    print(np.all((G_sys @ H_sys.T) % 2 == 0))

    # # Visualizza la struttura della matrice H (i punti neri sono gli '1')
    # plt.figure(figsize=(10, 5))
    # plt.spy(H_sys, markersize=1)
    # plt.title("Struttura della Matrice H Sistematica")
    # plt.show()