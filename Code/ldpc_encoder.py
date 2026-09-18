import numpy as np
import scipy.io

def ldpc_encode(info_row, G):
    return (G.T @ info_row) % 2
