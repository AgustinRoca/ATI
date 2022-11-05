from numpy.lib.function_base import copy
from src.utils.image_utils import thresholding

import numpy as np


def global_threshold(img, t):
    delta_t = 255
    new_img = copy(img)

    while(delta_t > 2):
        g1 = np.count_nonzero(new_img < t)
        g2 = np.count_nonzero(new_img >= t)
        sum1 = np.sum(np.where(new_img < t, new_img, 0))
        sum2 = np.sum(np.where(new_img >= t, new_img, 0))
        m1 = sum1 / g1
        m2 = sum2 / g2
        new_t = 0.5 * (m1 + m2)
        delta_t = abs(int(new_t) - int(t))
        t = new_t
    
    return int(t)


def otsu_threshold(img):
    hist = get_hist(img)
    accum_hist = get_accum_hist(hist)
    m = get_accum_median(hist)
    mg = m.max()
    var = get_var(accum_hist, m, mg)
    
    return np.nanargmax(var)


def get_hist(img):
    if len(img.shape) > 2:
        hist = np.zeros((3, 256), dtype=int)
    else: 
        hist = np.zeros(256, dtype=int)

    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            if len(img.shape) > 2:
                 for k in range(0, len(img[0][0])):
                     value = img[i][j][k]
                     hist[k][value] += 1
            else:
                value = img[i][j]
                hist[value] += 1
    

    return hist / (len(img) * len(img[0]))


def get_accum_hist(hist):
    accum_hist = np.zeros(hist.shape)
    accum_hist[0] = hist[0]
    for i in range(1, len(hist)):
        accum_hist[i] = hist[i] + accum_hist[i - 1]

    return accum_hist


def get_accum_median(hist):
    accum_median = np.zeros(hist.shape)
    accum_median[0] = 0
    for i in range(1, len(hist)):
        accum_median[i] = (hist[i] * i) + accum_median[i - 1]

    return accum_median
    

def get_var(accum_hist, m, mg):
    a = np.power((mg * accum_hist) - m, 2)
    b = accum_hist * (1 - accum_hist)
    return np.divide(a,b)