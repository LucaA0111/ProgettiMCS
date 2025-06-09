import numpy as np
from scipy.fftpack import dct, idct

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def apply_cutoff(dct_block, d):
    F = dct_block.shape[0]
    for k in range(F):
        for l in range(F):
            if k + l >= d:
                dct_block[k, l] = 0
    return dct_block

def normalize_block(block):
    block = np.rint(block)
    block[block < 0] = 0
    block[block > 255] = 255
    return block.astype(np.uint8)

def compress_image(img_array, F, d):
    h, w = img_array.shape
    h_crop = (h // F) * F
    w_crop = (w // F) * F
    img_array = img_array[:h_crop, :w_crop]
    compressed = np.zeros_like(img_array)

    for i in range(0, h_crop, F):
        for j in range(0, w_crop, F):
            block = img_array[i:i+F, j:j+F]
            c = dct2(block)
            c = apply_cutoff(c, d)
            ff = idct2(c)
            compressed[i:i+F, j:j+F] = normalize_block(ff)

    return compressed
