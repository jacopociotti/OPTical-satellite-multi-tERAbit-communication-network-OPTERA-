# LDPC Staircase Decoder con Adaptive Sliding Window e controllo della sindrome. 
# Versione 2: gestione separata dei messaggi estrinseci provenienti dai due vincoli adiacenti (L_E_left e L_E_right).

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
        self.H_matrix = H_matrix.toarray() if hasattr(H_matrix, 'toarray') else np.asarray(H_matrix, dtype=int)
        self.spa_max_iter = spa_max_iter
        self.perm = None 
        self.inv_perm = None 

        self.w_max = self.w + 2 # Larghezza massima permessa per la finestra
        self.delta_w = 1    # incremento della finestra se non converge

        if perm is not None:
            self.perm = np.asarray(perm, dtype=np.int64)

            if len(self.perm) != 2 * self.m:
                raise ValueError(f"La permutazione deve avere lunghezza {2 * self.m}, ma ne ha {len(self.perm)}.")
       
            self.inv_perm = np.empty_like(self.perm)
            self.inv_perm[self.perm] = np.arange(len(self.perm)) 

        logger.info(f"Decoder LDPC staircase: perm attiva = {self.perm is not None}")
        logger.info(f"H_matrix shape = {self.H_matrix.shape}")
        logger.info(f"spa_max_iter = {self.spa_max_iter}")

    def decode_sequence(self, received_llrs, original_bits = None, num_termination_blocks=2):
        """
        Decodifica l'intera sequenza di LLR ricevuti dal canale.
        """
        # Calcola quanti blocchi m x m sono stati ricevuti
        bits_per_block = self.m * self.m
        num_received_blocks = len(received_llrs) // bits_per_block
        
        # INIZIALIZZAZIONE MEMORIA
        # L_I: informazione intrinseca proveniente dal canale.
        # L_E_left[b]: contributo estrinseco prodotto dal vincolo A_b su B_b.
        # L_E_right[b]: contributo estrinseco prodotto dal vincolo A_{b+1} su B_b.
        #
        # Ogni blocco B_b appartiene infatti a due vincoli componente:
        #   A_b     = [B_{b-1}^T | B_b]
        #   A_{b+1} = [B_b^T     | B_{b+1}]
        # Le due memorie impediscono che un contributo sovrascriva l'altro.
        self.L_I = []
        self.L_E_left = []
        self.L_E_right = []
        
        # Il blocco B_0 non viene trasmesso, ma è tutti zeri.
        # Nelle LLR (convenzione LLR > 0 -> bit '0'), uno zero certo è un +infinito.
        B_0_LLR = np.full((self.m, self.m), 1000.0) 
        self.L_I.append(B_0_LLR)
        self.L_E_left.append(np.zeros((self.m, self.m), dtype=float))
        self.L_E_right.append(np.zeros((self.m, self.m), dtype=float))
        
        # Riempie L_I con i blocchi ricevuti dal canale e inizializza L_E a 0
        for i in range(num_received_blocks):
            start = i * bits_per_block
            end = start + bits_per_block
            block_llr = received_llrs[start:end].reshape((self.m, self.m))
            self.L_I.append(block_llr)
            self.L_E_left.append(np.zeros((self.m, self.m), dtype=float))
            self.L_E_right.append(np.zeros((self.m, self.m), dtype=float))
            
        total_blocks = len(self.L_I)
        num_info_blocks = total_blocks - 1 - num_termination_blocks # Sottrazione tra B_0 e le terminazioni
        
        decoded_bit_stream = []

        processed_blocks = 0
        failed_blocks = 0
        accepted_blocks = 0
        total_sweeps_sequence = 0
        total_window_size_sequence = 0


        # Memoria per i blocchi (hard decisions) precedentemente accettati
        # Serve per costruire \hat{A}_i = [(\hat{B}_{i-1}^{acc})^T | \hat{B}_i]
        B_acc = [np.zeros((self.m, self.m), dtype=int) for _ in range(total_blocks)]

        padded_original_bits = None


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

            conv_rows = 0
            total_rows = 0
            mean_abs_ext = 0.0

            start_time = time.time()  # Inizio del timer per la finestra corrente
            logger.info(f"Inizio Decodifica di B_{i}...")


            w_cur = self.w
            V_i = self.v_I_max if i == 1 else self.v_max
            v = 0

            total_sweeps = 0
            syndromes_ui = []

            block_accepted = False  # Flag per indicare se il blocco è stato accettato
            block_failed = False  # Flag per indicare se il blocco ha fallito la decodifica
     
        
            while True:
                # Calcola la fine della finestra (si rimpicciolisce alla fine del flusso)
                window_end = min(i + w_cur, total_blocks)
                
                # SWEEP SINGOLO---Iterazione progressiva sui blocchi DENTRO la finestra (da i a window_end - 1)
                for j in range(i, window_end):

                    # Decodifica le righe della matrice A_j = [B_{j-1}^T | B_j].
                    #
                    # Per evitare feedback positivo, il decoder di A_j riceve soltanto
                    # l'informazione estrinseca prodotta dall'ALTRO vincolo:
                    #   - per B_{j-1}: il contributo di A_{j-1}, cioè L_E_left[j-1];
                    #   - per B_j:     il contributo di A_{j+1}, cioè L_E_right[j].
                    # Il vecchio messaggio prodotto dallo stesso A_j viene escluso.
                    L_apriori_prev = self.L_I[j-1] + self.L_E_left[j-1]
                    L_apriori_curr = self.L_I[j] + self.L_E_right[j]
                    
                    for k in range(self.m):
                        # Per B_{j-1}^T, la riga k corrisponde alla colonna k di B_{j-1}.
                        row_llr_in_sys = np.concatenate((L_apriori_prev[:, k], L_apriori_curr[k, :]))

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

                        total_rows += 1
                        conv_rows += int(converged)
                        mean_abs_ext += np.mean(np.abs(extrinsic))

                        scale = self.alpha_c if converged else self.alpha_n
                        extrinsic *= scale
                        
                        # AGGIORNAMENTO DEI DUE MESSAGGI DISTINTI.
                        # La prima metà è il messaggio A_j -> B_{j-1}
                        # e viene memorizzata come contributo del vincolo destro.
                        self.L_E_right[j-1][:, k] = extrinsic[:self.m]

                        # La seconda metà è il messaggio A_j -> B_j
                        # e viene memorizzata come contributo del vincolo sinistro.
                        self.L_E_left[j][k, :] = extrinsic[self.m:]


                v += 1
                total_sweeps += 1

                # LLR a posteriori del blocco target: canale + entrambi i
                # contributi estrinseci provenienti dai due vincoli adiacenti.
                L_tot_final_i = (self.L_I[i] + self.L_E_left[i] + self.L_E_right[i])
                B_curr = (L_tot_final_i < 0).astype(int)
                    
                # Prende la hard decision del blocco (i-1) già accettata dal decoder
                B_prev_acc = B_acc[i-1]
                    
                # Costruzione \hat{A}_i e calcolo sindrome
                u_i = 0
                for k in range(self.m):
                    # B_prev_acc[:, k] corrisponde alla riga k di (B_prev_acc)^T
                    row_sys = np.concatenate((B_prev_acc[:, k], B_curr[k, :]))
                        
                    # Converti all'ordine RAW prima di moltiplicare per H se la permutazione è attiva
                    if self.perm is not None:
                        row_raw = row_sys[self.inv_perm]
                    else:
                        row_raw = row_sys
                        
                    # Calcolo sindrome (modulo 2) per la riga
                    s_k = np.mod(np.dot(row_raw, self.H_matrix.T), 2)
                    u_i += np.sum(s_k)
                        
                syndromes_ui.append(u_i)
                
                # --- Tabella delle Decisioni ---
                if u_i == 0:
                    # Early Stop: Sindrome zero
                    B_acc[i] = B_curr
                    block_accepted = True
                    logger.info(f"Blocco {i} ACCETTATO. w={w_cur}, sweeps_totali={total_sweeps}, seq_sindromi={[int(x) for x in syndromes_ui]}")
                    break
                else:
                    if v < V_i:
                        # Mantieni la finestra, fai un altro sweep (torna all'inizio del while)
                        continue
                    elif v == V_i:
                        if w_cur < self.w_max:
                            # Aumenta la finestra, resetta il counter locale v, NON resettare le LLR
                            w_cur = min(w_cur + self.delta_w, self.w_max)
                            v = 0
                            logger.debug(f"Blocco {i}: aumento finestra a {w_cur} (sindrome corrente {u_i})")
                            continue
                        else:
                            block_failed = True

                            # Decoding failure per B_i
                            B_acc[i] = B_curr  # Salviamo la migliore stima disponibile (best-effort)

                            logger.warning(f"Blocco {i} FALLITO. w={w_cur}, sweeps_totali={total_sweeps}, seq_sindromi={[int(x) for x in syndromes_ui]}")
                            break


            processed_blocks += 1
            failed_blocks += int(block_failed)
            accepted_blocks += int(block_accepted)

            total_sweeps_sequence += total_sweeps
            total_window_size_sequence += w_cur

            mean_abs_ext /= max(total_rows, 1)

            logger.info(f"Finestra {i}: righe convergenti {conv_rows}/{total_rows} "f"({conv_rows/total_rows:.3%}), mean|extrinsic|={mean_abs_ext:.3f}")
            
            # Estrae solo i bit di informazione (le prime m-r colonne)
            info_bits_i = B_acc[i][:, : (self.m - self.r)].flatten()
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


        if processed_blocks > 0:
            failure_probability = failed_blocks / processed_blocks
            average_sweeps = total_sweeps_sequence / processed_blocks
            average_window_size = total_window_size_sequence / processed_blocks
        else:
            failure_probability = 0.0
            average_sweeps = 0.0
            average_window_size = 0.0
        logger.info("=" * 70)
        logger.info("METRICHE DECODER A LIVELLO DI SEQUENZA")
        logger.info(f"Failure probability: {failure_probability:.6e} "f"({failed_blocks}/{processed_blocks})")
        logger.info(f"Numero medio di sweep per blocco: "f"{average_sweeps:.3f}")
        logger.info(f"Dimensione media della finestra: "f"{average_window_size:.3f}")
        logger.info("=" * 70)

        return np.array(decoded_bit_stream)
