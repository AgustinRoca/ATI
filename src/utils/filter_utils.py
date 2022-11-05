from tkinter.constants import N
import numpy as np
from src.utils.image_utils import normalize
from src.utils.mask_utils import fill_outer_matrix, weighted_mean
from numpy import copy, uint32, uint64
from math import pi
from scipy import ndimage


def isotropic_dif(img, t):
    new_img = copy(img)
    mask = gaussian_conv(t)

    if len(img.shape) > 2:
        for dim in range(img.shape[-1]):
            convolution = ndimage.convolve(new_img[:, :, dim], mask)
            new_img[:, :, dim] = convolution
    else:
        new_img = ndimage.convolve(new_img, mask)
    return normalize(new_img, 0,  weighted_mean(255*np.ones(mask.shape), mask))


def gaussian_conv(t):
    mask_side = 2*t + 1
    mask = np.ones((mask_side, mask_side))

    for i in range(0, mask_side):
        y = i - mask_side//2
        for j in range(0, mask_side):
            x = j - mask_side//2
            mask[i][j] = 1/(4 * pi * t) * np.exp(-(x**2 + y**2) / (4 * t))

    return mask
    

def anisotropic_diff(img, sigma, t_final, lam=0.25):
    new_img = copy(img).astype(float)
    aux_img = copy(img).astype(float)

    for t in range(0, t_final):
        for i in range(1, len(img) - 1):
            for j in range(1, len(img[0]) - 1):
                if len(img.shape) > 2:
                    for k in range(0, len(img[0][0])):
                        dn = aux_img[i + 1][j][k] - aux_img[i][j][k]
                        ds = aux_img[i - 1][j][k] - aux_img[i][j][k]
                        de = aux_img[i][j + 1][k] - aux_img[i][j][k]
                        dw = aux_img[i][j - 1][k] - aux_img[i][j][k]
                        cn = lorentz(dn, sigma)
                        cs = lorentz(ds, sigma)
                        ce = lorentz(de, sigma)
                        cw = lorentz(dw, sigma)
                        new_img[i][j][k] = aux_img[i][j][k] + lam * (dn * cn + ds * cs + de * ce + dw * cw)
                else:
                    dn = aux_img[i + 1][j] - aux_img[i][j]
                    ds = aux_img[i - 1][j] - aux_img[i][j]
                    de = aux_img[i][j + 1] - aux_img[i][j]
                    dw = aux_img[i][j - 1] - aux_img[i][j]
                    cn = lorentz(dn, sigma)
                    cs = lorentz(ds, sigma)
                    ce = lorentz(de, sigma)
                    cw = lorentz(dw, sigma)
                    new_img[i][j] = aux_img[i][j] + lam * (dn * cn + ds * cs + de * ce + dw * cw)
        aux_img = copy(new_img)
    
    return normalize(new_img, new_img.min(), new_img.max())

def lorentz(x, sigma):
    a = (x**2/sigma**2) + 1
    return 1 / a


def bilateral_filter(img, r, s, size, normalize_result = True):
    new_img = copy(img).astype(uint32)
    aux_img = fill_outer_matrix(img, size).astype(uint32)

    for i in range(0, len(img)):
        end_aux_col = i + size
        x = i + np.floor(size/2)
        for j in range(0, len(img[0])):
            end_aux_row = j + size
            y = j + np.floor(size/2)
            if len(img.shape) > 2:
                for k in range(0, len(img[0][0])):
                    new_img[i][j][k] = apply_bilateral_mask(aux_img[i: end_aux_col, j: end_aux_row, k], s, r, size, x, y)
            else:
                new_img[i][j] = apply_bilateral_mask(aux_img[i: end_aux_col, j: end_aux_row], s, r, size, x, y)

    if normalize_result:
        new_img = normalize(new_img, 0, new_img.max())
    return new_img

def apply_bilateral_mask(frame, s, r, size, x, y):
    sum_up = 0
    sum_down = 0
    min = int(-np.floor(size/2))
    max = int(np.ceil(size/2))
    for i in range(min, max):
        for j in range(min, max):
            w = w_bilateral(x, y, x + i, y + j, s, r, frame[-min][-min], frame[i - min][j - min])
            pixel = frame[i - min][j - min]
            sum_up += pixel * w
            sum_down += w
    
    return sum_up / sum_down


def w_bilateral(i, j, k, l, s, r, p1, p2):
    a = -((i-k)**2 + (j-l)**2)/(2 * (s**2))
    b = -abs(int(p1)-int(p2)) / (2 * (r**2))
    return np.exp(a + b)