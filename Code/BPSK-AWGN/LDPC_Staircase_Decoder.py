# LDPC Staircase decoder with fixed Sliding Window + one memory (L_E) for the extrinsic information

import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class LDPC_Staircase_Decoder:
    def __init__(self, m, r, w, alpha_c, alpha_n, v_max, v_I_max, H_matrix, component_spa_decoder, spa_max_iter=10, perm=None, track_extrinsic_stats=False):
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
        self.track_extrinsic_stats = bool(track_extrinsic_stats)

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
        total_decode_start_time = time.perf_counter()

        wrong_blocks = 0
        component_decoder_calls = 0
        total_sweeps_sequence = 0

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
        B_0_LLR = np.full((self.m, self.m), 1000.0, dtype=np.float32)
        self.L_I.append(B_0_LLR)
        self.L_E.append(np.zeros((self.m, self.m), dtype=np.float32))
        
        # Riempie L_I con i blocchi ricevuti dal canale e inizializza L_E a 0
        for i in range(num_received_blocks):
            start = i * bits_per_block
            end = start + bits_per_block
            block_llr = np.asarray(
                received_llrs[start:end], dtype=np.float32
            ).reshape((self.m, self.m))
            self.L_I.append(block_llr)
            self.L_E.append(np.zeros((self.m, self.m), dtype=np.float32))
            
        total_blocks = len(self.L_I)
        num_info_blocks = total_blocks - 1 - num_termination_blocks # Sottrazione tra B_0 e le terminazioni
        
        decoded_bit_stream = np.empty(
            num_info_blocks * self.info_bits_per_block, dtype=np.uint8
        )

        # Reusable row buffers: avoid millions of temporary allocations.
        row_llr_in_sys = np.empty(2 * self.m, dtype=np.float32)
        row_llr_in = np.empty(2 * self.m, dtype=np.float32)
        row_llr_out_sys = np.empty(2 * self.m, dtype=np.float32)
        extrinsic = np.empty(2 * self.m, dtype=np.float32)


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
            logger.info(f"OLD - Inizio Decodifica di B_{i}...")

            # Imposta le iterazioni (più alte per la prima finestra, come da paper)
            current_v_max = self.v_I_max if i == 1 else self.v_max
            
            # Calcola la fine della finestra (si rimpicciolisce alla fine del flusso)
            window_end = min(i + self.w, total_blocks)
            
            # Iterazioni della Sliding Window
            for v in range(current_v_max):
                
                # Iterazione progressiva sui blocchi DENTRO la finestra (da i a window_end - 1)
                for j in range(i, window_end):
                    
                    # Stessa identica decodifica delle m righe di A_j, ma il ciclo
                    # k=0,...,m-1 viene eseguito dentro un unico kernel Numba.
                    # L'ordine dei blocchi j, degli sweep e delle finestre non cambia.
                    if hasattr(self.decode_row, "decode_component_old"):
                        conv_batch, ext_batch = self.decode_row.decode_component_old(
                            self.L_I[j-1],
                            self.L_E[j-1],
                            self.L_I[j],
                            self.L_E[j],
                            self.inv_perm,
                            self.perm,
                            self.spa_max_iter,
                            self.alpha_c,
                            self.alpha_n,
                            self.track_extrinsic_stats,
                        )
                        total_rows += self.m
                        conv_rows += int(conv_batch)
                        if self.track_extrinsic_stats:
                            mean_abs_ext += float(ext_batch)
                    else:
                        # Fallback compatibile con un decoder componente generico.
                        L_tot_prev = self.L_I[j-1] + self.L_E[j-1]
                        L_tot_curr = self.L_I[j] + self.L_E[j]

                        for k in range(self.m):
                            row_llr_in_sys[:self.m] = L_tot_prev[:, k]
                            row_llr_in_sys[self.m:] = L_tot_curr[k, :]

                            if self.perm is not None:
                                np.take(row_llr_in_sys, self.inv_perm, out=row_llr_in)
                            else:
                                np.copyto(row_llr_in, row_llr_in_sys)

                            row_llr_out, converged = self.decode_row(
                                row_llr_in, self.H_matrix, max_iter=self.spa_max_iter
                            )

                            if self.perm is not None:
                                np.take(row_llr_out, self.perm, out=row_llr_out_sys)
                            else:
                                np.copyto(row_llr_out_sys, row_llr_out)

                            np.subtract(row_llr_out_sys, row_llr_in_sys, out=extrinsic)

                            total_rows += 1
                            conv_rows += int(converged)
                            if self.track_extrinsic_stats:
                                mean_abs_ext += float(np.mean(np.abs(extrinsic)))

                            scale = self.alpha_c if converged else self.alpha_n
                            extrinsic *= scale

                            self.L_E[j-1][:, k] = extrinsic[:self.m]
                            self.L_E[j][k, :] = extrinsic[self.m:]

            component_decoder_calls += total_rows
            total_sweeps_sequence += current_v_max

            if self.track_extrinsic_stats:
                mean_abs_ext /= max(total_rows, 1)

            logger.info(f"Finestra {i}: righe convergenti {conv_rows}/{total_rows} "f"({conv_rows/total_rows:.3%}), mean|extrinsic|={mean_abs_ext:.3f}")
 

            
            
            L_tot_final_i = self.L_I[i] + self.L_E[i]   # contiene un LLR per ogni bit del blocco B_i
            
            # Hard decision: LLR > 0 significa bit '0', LLR < 0 significa bit '1'
            hard_block = (L_tot_final_i < 0).astype(np.uint8)
            
            # Estrae solo i bit di informazione (le prime m-r colonne)
            info_bits_i = hard_block[:, : (self.m - self.r)].flatten()
            out_start = (i - 1) * self.info_bits_per_block
            out_end = out_start + self.info_bits_per_block
            decoded_bit_stream[out_start:out_end] = info_bits_i

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

                wrong_blocks += int(errors > 0)
                
                logger.info(f"Blocco {i}/{num_info_blocks} decodificato in {duration:.2f}s / {duration/60:.2f} min | Errori: {errors}/{info_size} | BER: {block_ber:.4e}")
            else:
                logger.info(f"Blocco {i}/{num_info_blocks} decodificato in {duration:.2f}s / {duration/60:.2f} min | BER: N/A (original bits not provided)")


        total_decode_time = time.perf_counter() - total_decode_start_time
        processed_blocks = num_info_blocks  # Solo i blocchi di informazione sono considerati per le metriche finali

        if processed_blocks > 0:
            bler = (wrong_blocks / processed_blocks) if padded_original_bits is not None else None
            average_component_calls = component_decoder_calls / processed_blocks
            average_time_per_block = total_decode_time / processed_blocks
            average_sweeps = total_sweeps_sequence / processed_blocks
            average_window_size = self.w  # La dimensione della finestra è costante, quindi non ha senso calcolare una media
        else:
            bler = 0.0
            average_component_calls = 0.0
            average_time_per_block = 0.0
            average_sweeps = 0.0

        logger.info("=" * 70)
        logger.info("METRICHE DECODER A LIVELLO DI SEQUENZA")
        logger.info(f"Numero medio di sweep per blocco: "f"{average_sweeps:.3f}")
        logger.info(f"Dimensione media della finestra: "f"{average_window_size:.3f}")
        logger.info(f"Numero totale di chiamate al decoder componente: {component_decoder_calls}")
        logger.info(f"Numero medio di chiamate al decoder componente per blocco: {average_component_calls:.3f}")
        logger.info(f"BLER (Block Error Rate): {bler:.6e} ({wrong_blocks}/{processed_blocks})")
        logger.info(f"Tempo totale di decodifica: {total_decode_time:.2f}s / {total_decode_time/60:.2f} min")
        logger.info(f"Tempo medio di decodifica per blocco: {average_time_per_block:.2f}s / {average_time_per_block/60:.2f} min")
        logger.info("=" * 70)



        self.last_metrics = {
            "processed_blocks": processed_blocks,
            "wrong_blocks": wrong_blocks,
            "processed_blocks": processed_blocks,
            "average_sweeps": average_sweeps,
            "total_sweeps_sequence": total_sweeps_sequence,
            "average_window_size": average_window_size,
            "component_decoder_calls": component_decoder_calls,
            "average_component_calls": average_component_calls,
            "bler": bler,
            "total_decode_time": total_decode_time,
            "average_time_per_block": average_time_per_block
        }

        return decoded_bit_stream
