# /usr/bin/python3

import numpy as np

def points_dist(pointB, pointA):
    return np.linalg.norm(pointB - pointA)


def random_2d_vector(seed=None,):
    """return a random 2d vector bounded in a circumnference
    of radius = 1"""
    if seed:
        np.random.seed = seed
    ro = np.random.rand()
    theta = np.random.rand() * 2 * np.pi
    x = ro * np.cos(theta)
    y = ro * np.sin(theta)
    return np.array([x, y])

def line_generation(pos1, pos2):
    m = (pos2[1] - pos1[1]) / (pos2[0] - pos1[0])
    q = pos1[1] - m * pos1[0]
    return m, q

def ort_line_gen(m, x, y):
    m_1 = -1 / m
    q = y - m_1 * x
    return m_1, q

def sat(val, a, b=None):
    if b is None:
        low, upp = -a, a
    else:
        low, upp = min(a, b), max(a, b)
    array_sat = np.clip(val, low, upp)
    return array_sat


def compute_pos_des(p, p_c, m, q, l):
    if abs(m) <= 1:
        x_des1 = (p[1] + l * m - q) / m
        x_des2 = (p[1] - l * m - q) / m
        if abs(x_des1 - p_c[0]) >= abs(x_des2 - p_c[0]):
            x_des = x_des1
        else:
            x_des = x_des2
        y_des = x_des * m + q
    else:
        y_des1 = (p[0] + l / m) * m + q
        y_des2 = (p[0] - l / m) * m + q
        if abs(y_des1 - p_c[1]) >= abs(y_des2 - p_c[1]):
            y_des = y_des1
        else:
            y_des = y_des2
        x_des = (y_des - q) / m
    return x_des, y_des

def limit_decimal_places(number, decimal_places):
    format_string = "{:." + str(decimal_places) + "f}"
    return float(format_string.format(number))