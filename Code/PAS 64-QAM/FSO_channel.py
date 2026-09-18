# FSO Gamma-Gamma channel model with LDPC Staircase coding and decoding

import numpy as np


def gamma_gamma_turbulence(num_symbols, alpha, beta, block_size):
    """
    Genera l'irradianza I_tur con distribuzione Gamma-Gamma.

    I_tur = X * Y
    X ~ Gamma(alpha, scale=1/alpha)
    Y ~ Gamma(beta,  scale=1/beta)

    Con questa parametrizzazione:
        E[X] = E[Y] = 1
        E[I_tur] = 1

    Parametri:
    num_symbols : numero totale di simboli da generare
    alpha : parametro della distribuzione Gamma-Gamma
    beta : parametro della distribuzione Gamma-Gamma
    block_size : numero di simboli con lo stesso coefficiente di canale
    """
    num_blocks = int(np.ceil(num_symbols / block_size))

    X = np.random.gamma(shape=alpha, scale=1.0 / alpha, size=num_blocks)

    Y = np.random.gamma(shape=beta, scale=1.0 / beta, size=num_blocks)

    I_blocks = X * Y # calcola un valore di I_tur per ogni blocco

    I_tur = np.repeat(I_blocks, block_size) # ogni valore di I_tur viene ripetuto block_size volte
    I_tur = I_tur[:num_symbols] # taglia l'array per avere esattamente num_symbols elementi

    phase_blocks = np.random.uniform(0, 2 * np.pi, size=num_blocks)
    
    phase = np.repeat(phase_blocks, block_size)[:num_symbols]  # Ripete ogni fase per block_size simboli
  
    return I_tur, phase

def pointing_error(num_symbols, A0, w_eq, sigma_s, block_size):

    num_blocks = int(np.ceil(num_symbols / block_size))

    # Spostamento del fascio sui due assi
    x_p = np.random.normal(0, sigma_s,num_blocks)

    y_p = np.random.normal(0, sigma_s, num_blocks)

    # Distanza radiale al quadrato
    r2 = x_p**2 + y_p**2

    # Pointing loss per blocco
    I_pt_blocks = A0 * np.exp(-2 * r2 / w_eq**2)

    # Block fading
    I_pt = np.repeat(I_pt_blocks, block_size)[:num_symbols]

    return I_pt


def atmospheric_attenuation(distance_km, attenuation_db_km):
    """
    Attenuazione atmosferica.

    distance_km       : distanza [km]
    attenuation_db_km : attenuazione specifica [dB/km]

    ritorna I_att, guadagno di potenza
    """

    loss_db = attenuation_db_km * distance_km

    I_att = 10 ** (-loss_db / 10)

    return I_att


def noise_variance(ebn0_db, rate, M, Es):
    """
    Calcola sigma^2 assumendo:

        Eb/N0 = Es / (2 * rate * bits_per_symbol * sigma^2)

    Per BPSK:
        bits_per_symbol = 1
        Es = 1
    """
    bits_per_symbol = int(np.log2(M))   # k

    ebn0_linear = 10 ** (ebn0_db / 10)

    noise_variance = Es / (2 *rate * bits_per_symbol * ebn0_linear)

    return noise_variance


def fso_coherent_channel(symbols, Eb_N0_dB, M, rate, alpha, beta, block_size):
    """
    Canale FSO coerente con sola turbolenza:

        h = sqrt(I_tur * I_pt) * exp(1j * phase)
        y = h * x + n

    dove:
        n ~ CN(0, N0)
    """

    Es = np.mean(np.abs(symbols) ** 2)  # Energia media del simbolo
    noise_var = noise_variance(ebn0_db=Eb_N0_dB, rate=rate, M=M, Es=Es)

    # Generazione della turbolenza
    I_tur, phase = gamma_gamma_turbulence(num_symbols = len(symbols), alpha = alpha, beta = beta, block_size = block_size)

    I_pt = pointing_error(num_symbols = len(symbols), A0 = 0.8532, w_eq = 0.1773, sigma_s = 0.01, block_size = block_size)

    I_att = atmospheric_attenuation(distance_km = 1.0, attenuation_db_km = 0.2208)

    I_tot = I_tur * I_pt * I_att  # Irradianza totale

    h = np.sqrt(I_tot) * np.exp(1j * phase)  # Coefficiente di canale complesso

    noise = np.sqrt(noise_var) * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))

    y = h * symbols + noise

    return y, h, I_tot, noise_var

