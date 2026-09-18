import numpy as np
import scipy.io

class LDPC_Staircase_Encoder:
    def __init__(self, m, r, G_matrix, component_LDPC_encoder):
        """
        Inizializza il codificatore Staircase.
        m: dimensione del blocco (matrice m x m)
        r: numero di bit di parità per ogni riga
        G_matrix: matrice generatrice del codice LDPC componente
        component_LDPC_encoder: oggetto che modella il codice LDPC (2m, 2m-r).
                                Prende un array di (2m-r) bit e restituisce (2m) bit.
        """
        self.m = m
        self.r = r
        self.G_matrix = G_matrix
        self.K_info_per_row = (2 * m) - r
        self.N = 2 * m
        self.K_info_bits_per_block = m * (m - r)
        self.encode_row = component_LDPC_encoder
        
        # Inizializza il blocco B_0 (Reference state, tutto a zeri)
        self.B_prev = np.zeros((self.m, self.m), dtype=int)

    # Funzione per codificare un singolo blocco B_i
    def encode_block(self, info_bits):
        """
        Codifica un singolo blocco B_i della scala.
        """ 
        # Costruisce B_{i,Sx} inserendo l'informazione nelle prime (m-r) colonne
        B_i_Sx = info_bits.reshape((self.m, self.m - self.r))
        
        # Concatena la trasposta del blocco precedente: A = [B_{i-1}^T , B_{i,Sx}]
        # B_prev.T è m x m. B_i_Sx è m x (m-r). A diventerà m x (2m-r)
        A = np.hstack((self.B_prev.T, B_i_Sx))
        
        # Matrice per la parità
        B_i_Rx = np.zeros((self.m, self.r), dtype=int)
        
        # Codifica riga per riga con il codice LDPC componente
        for j in range(self.m):
            # Passa la riga (lunghezza 2m-r) al codificatore LDPC
            codeword = self.encode_row(A[j, :], self.G_matrix)  
            # Estrae solo gli ultimi 'r' bit (la parità)  VALE SOLO SE LA MATRICE G È IN FORMA SISTEMATICA
            parity_bits = codeword[-self.r:]
            B_i_Rx[j, :] = parity_bits
            
        # Il blocco finale B_i è l'unione di informazione e parità
        B_i = np.hstack((B_i_Sx, B_i_Rx))
        
        # Aggiorna il blocco precedente per lo step successivo della scala
        self.B_prev = B_i
        
        return B_i

    def encode_sequence(self, bit_stream, num_termination_blocks=2):
        """
        Codifica un intero flusso di bit, aggiungendo i blocchi di terminazione.
        """
        num_blocks = int(np.ceil(len(bit_stream) / self.K_info_bits_per_block))
        
        # Padding con zeri se il flusso non è un multiplo esatto del blocco
        pad_len = (num_blocks * self.K_info_bits_per_block) - len(bit_stream)
        if pad_len > 0:
            bit_stream = np.concatenate((bit_stream, np.zeros(pad_len, dtype=int)))
            
        encoded_blocks = []
        
        # Codifica i blocchi di informazione
        for i in range(num_blocks):
            start = i * self.K_info_bits_per_block
            end = start + self.K_info_bits_per_block
            block_info = bit_stream[start:end]
            
            B_i = self.encode_block(block_info)
            encoded_blocks.append(B_i)
            # print(f"Blocco {i+1} codificato.")
            
        # Aggiunge i blocchi di terminazione (T=2) con informazione tutti zeri
        zeros_info = np.zeros(self.K_info_bits_per_block, dtype=int)
        for t in range(num_termination_blocks):
            B_t = self.encode_block(zeros_info)
            encoded_blocks.append(B_t)
            # print(f"Blocco di Terminazione T{t+1} codificato.")
            
        return encoded_blocks



