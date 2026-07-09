import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class LDPC_Staircase_Decoder:
    def __init__(self, m, r, w, alpha_c, alpha_n, v_max, v_I_max, H_matrix, component_spa_decoder, spa_max_iter=10, perm = None):
        """
        Inizializza il Decodificatore Staircase a Finestra Scorrevole (Sliding Window SDD).
        
        m, r: Dimensioni del blocco e parità
        w: Dimensione della finestra scorrevole
        alpha_c: Fattore di scala se l'SPA converge (es. 0.75)
        alpha_n: Fattore di scala se l'SPA NON converge (es. 0.375)
        v_max: Iterazioni massime per una finestra normale
        v_I_max: Iterazioni massime per la PRIMA finestra (es. 24, B_1 è più debole)
        H_matrix: Matrice di parità del codice LDPC componente
        component_spa_decoder: Funzione decoder LDPC (SPA) per la singola riga (2m)
        spa_max_iter: Numero massimo di iterazioni per l'algoritmo SPA
        perm: Permutazione per i bit di informazione
        """
        self.m = m
        self.r = r
        self.w = w
        self.alpha_c = alpha_c
        self.alpha_n = alpha_n
        self.v_max = v_max
        self.v_I_max = v_I_max
        self.decode_row = component_spa_decoder
        self.info_bits_per_block = m * (m - r)
        self.H_matrix = H_matrix
        self.spa_max_iter = spa_max_iter
        self.perm = None 
        self.inv_perm = None 

        if perm is not None:
            self.perm = np.asarray(perm, dtype=np.int64)

            if len(self.perm) != 2 * self.m:
                raise ValueError(f"La permutazione deve avere lunghezza {2 * self.m}, ma ne ha {len(self.perm)}.")
       
            self.inv_perm = np.empty_like(self.perm)
            self.inv_perm[self.perm] = np.arange(len(self.perm)) 

    def decode_sequence(self, received_llrs, original_bits = None, num_termination_blocks=2):
        """
        Decodifica l'intera sequenza di LLR ricevuti dal canale.
        """
        # Calcola quanti blocchi m x m sono stati ricevuti
        bits_per_block = self.m * self.m
        num_received_blocks = len(received_llrs) // bits_per_block
        
        # INIZIALIZZAZIONE MEMORIA (Intrinseca ed Estrinseca)
        # L_I: Informazione intrinseca (dal canale)
        # L_E: Informazione estrinseca (dal decoder)
        self.L_I = []
        self.L_E = []
        
        # Il blocco B_0 non viene trasmesso, ma è tutti zeri.
        # Nelle LLR (convenzione LLR > 0 -> bit '0'), uno zero certo è un +infinito.
        B_0_LLR = np.full((self.m, self.m), 1000.0) 
        self.L_I.append(B_0_LLR)
        self.L_E.append(np.zeros((self.m, self.m)))
        
        # Riempie L_I con i blocchi ricevuti dal canale e inizializza L_E a 0
        for i in range(num_received_blocks):
            start = i * bits_per_block
            end = start + bits_per_block
            block_llr = received_llrs[start:end].reshape((self.m, self.m))
            self.L_I.append(block_llr)
            self.L_E.append(np.zeros((self.m, self.m)))
            
        total_blocks = len(self.L_I)
        num_info_blocks = total_blocks - 1 - num_termination_blocks # Sottrazione tra B_0 e le terminazioni
        
        decoded_bit_stream = []


        # Se abbiamo fornito i bit originali, aggiungiamo il padding per farli combaciare 
        # con i blocchi completi
        padded_original_bits = None
        if original_bits is not None:
            num_blocks = int(np.ceil(len(original_bits) / self.info_bits_per_block))
            pad_len = (num_blocks * self.info_bits_per_block) - len(original_bits)
            if pad_len > 0:
                padded_original_bits = np.concatenate((original_bits, np.zeros(pad_len, dtype=int)))
            else:
                padded_original_bits = original_bits



        # SLIDING WINDOW DECODING
        # i parte da 1 (perché B_0 è noto) fino all'ultimo blocco dati
        for i in range(1, num_info_blocks + 1):

            start_time = time.time()  # Inizio del timer per la finestra corrente
            logger.info(f"Inizio Decodifica di B_{i}...")

            # Imposta le iterazioni (più alte per la prima finestra, come da paper)
            current_v_max = self.v_I_max if i == 1 else self.v_max
            
            # Calcola la fine della finestra (si rimpicciolisce alla fine del flusso)
            window_end = min(i + self.w, total_blocks)
            
            # Iterazioni della Sliding Window
            for v in range(current_v_max):
                
                # Iterazione progressiva sui blocchi DENTRO la finestra (da i a window_end - 1)
                for j in range(i, window_end):
                    
                    # Decodifica le righe della matrice A = [B_{j-1}^T , B_j]
                    for k in range(self.m):
                        # Somma intrinseca ed estrinseca precedente e corrente
                        L_tot_prev = self.L_I[j-1] + self.L_E[j-1]
                        L_tot_curr = self.L_I[j] + self.L_E[j]
                        
                        # Costruisce la riga k-esima da decodificare in ordine SISTEMATICO. 
                        # ATTENZIONE: per B_{j-1}^T, la riga k è la COLONNA k di B_{j-1}!
                        row_llr_in_sys = np.concatenate((L_tot_prev[:, k], L_tot_curr[k, :]))

                        if self.perm is not None:
                            # Applica la permutazione per ottenere l'ordine corretto RAW 
                            row_llr_in = row_llr_in_sys[self.inv_perm]
                        else:
                            row_llr_in = row_llr_in_sys
                        
                        # Passa la riga all'algoritmo Sum-Product (SPA) - RAW
                        row_llr_out, converged = self.decode_row(row_llr_in, self.H_matrix, max_iter=self.spa_max_iter)
                        
                        # Riporta l'uscita del decoder dall'ordine RAW all'ordine SISTEMATICO
                        if self.perm is not None:
                            row_llr_out_sys = row_llr_out[self.perm]
                        else:
                            row_llr_out_sys = row_llr_out

                        # Esetrae l'informazione puramente estrinseca e la scala
                        extrinsic = row_llr_out_sys - row_llr_in_sys
                        scale = self.alpha_c if converged else self.alpha_n
                        extrinsic *= scale
                        
                        # AGGIORNAMENTO (Aggiorna la memoria L_E)
                        self.L_E[j-1][:, k] = extrinsic[:self.m]
                        self.L_E[j][k, :] = extrinsic[self.m:]


            
            
            L_tot_final_i = self.L_I[i] + self.L_E[i]   # contiene un LLR per ogni bit del blocco B_i
            
            # Hard decision: LLR > 0 significa bit '0', LLR < 0 significa bit '1'
            hard_block = (L_tot_final_i < 0).astype(int)
            
            # Estrae solo i bit di informazione (le prime m-r colonne)
            info_bits_i = hard_block[:, : (self.m - self.r)].flatten()
            decoded_bit_stream.extend(info_bits_i)

            end_time = time.time()  # Fine del timer per la finestra corrente
            duration = end_time - start_time  # Fine del timer per la finestra corrente

            # --- Calcolo e Log del BER per il singolo blocco ---
            if padded_original_bits is not None:
                # La dimensione dell'informazione in un blocco è m * (m - r)
                #info_size = self.m * (self.m - self.r)
                info_size = self.info_bits_per_block
                # Calcola gli indici di inizio e fine per il blocco originale corrispondente al blocco 'i'
                # Nota: i parte da 1, quindi per i=1, start_idx=0
                start_idx = (i - 1) * info_size
                end_idx = start_idx + info_size
                
                # Estrae i bit originali per questo blocco
                orig_block_bits = padded_original_bits[start_idx:end_idx]
                
                # Calcola gli errori confrontando il blocco decodificato con quello originale
                errors = np.sum(info_bits_i != orig_block_bits)
                block_ber = errors / info_size
                
                logger.info(f"Blocco {i}/{num_info_blocks} decodificato in {duration:.2f}s / {duration/60:.2f} min | Errori: {errors}/{info_size} | BER: {block_ber:.4e}")
            else:
                logger.info(f"Blocco {i}/{num_info_blocks} decodificato in {duration:.2f}s / {duration/60:.2f} min")


        return np.array(decoded_bit_stream)
