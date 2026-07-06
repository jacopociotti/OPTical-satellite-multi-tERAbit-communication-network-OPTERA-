import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def qam_mod_opt(bits, M):

    k = int(np.log2(M))
    sqrtM = int(np.sqrt(M))

    norm_factor = np.sqrt((2/3)*(M-1)) # fattore di normalizzazione per avere potenza unitaria

    bits = bits.reshape((-1, k))

    k_axis = k//2 # divide i bit in due parti uguali per asse I e Q

    levels = np.arange(-(sqrtM-1), sqrtM, 2) / norm_factor # livelli di modulazione normalizzati

    bits_I = bits[:, :k_axis]
    bits_Q = bits[:, k_axis:]
    
    idx_dec_I = bits_I.dot(2**np.arange(k_axis-1,-1,-1))    # assegna ad ogni simbolo (k_axis bit) il suo indice decimale binario es [1, 0] -> 1*2 + 0*1 = 2 -> idx 2
    idx_dec_Q = bits_Q.dot(2**np.arange(k_axis-1,-1,-1))    # assegna ad ogni simbolo (k_axis bit) il suo indice decimale binario es [0, 1] -> 0*2 + 1*1 = 1 -> idx 1

    vect = np.array(range(sqrtM))
    gray_constellation = np.bitwise_xor(vect, np.floor(vect/2).astype(int)) # mapping da indici binari a indici Gray es [0, 1 , 2 , 3] -> [0, 1, 3, 2]
    inv_gray_constellation = np.argsort(gray_constellation) # look-up table, dato un valore Gray ottengo il suo indice es [0, 1, 3, 2, 6, 7, 5, 4] -> [0, 1, 3, 2, 7, 6, 4, 5]
 
    idx_I = inv_gray_constellation[idx_dec_I]   # converte il valore Gray nell'indice corretto es [0, 1 , 2 , 3] -> [0, 1, 3, 2]
    idx_Q = inv_gray_constellation[idx_dec_Q]   # mapping da indici binari a indici Gray es [0, 1 , 2 , 3] -> [0, 1, 3, 2]
 
    I = levels[idx_I]
    Q = levels[idx_Q]

    return I + 1j*Q


   # es mapping Binario                                    es mapping Gray
    # 00 -> 0*2 + 0*1 = 0 -> idx 0 -> livello -3            00 ->  idx 0 -> livello -3
    # 01 -> 0*2 + 1*1 = 1 -> idx 1 -> livello -1            01 ->  idx 1 -> livello -1
    # 10 -> 1*2 + 0*1 = 2 -> idx 2 -> livello +1            11 ->  idx 2 -> livello +1
    # 11 -> 1*2 + 1*1 = 3 -> idx 3 -> livello +3            10 ->  idx 3 -> livello +3









