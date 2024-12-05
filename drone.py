import numpy as np
from utils import random_2d_vector, points_dist, line_generation, ort_line_gen, sat, compute_pos_des
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import scipy as sp
from math import exp
import sys
from numpy import arange
import matplotlib

matplotlib.use('TkAgg')


class Drone:
    def __init__(self, initial_area, center, simulation):
        self.P = random_2d_vector(center) * initial_area
        self.V = np.array([0, 0, 0])
        self.sim = simulation
        self.trail = []
        self.trailV = []

    def step(self, dt, C, r):
        self.P = self.P + self.V * dt
        self.trail.append(self.P)
        self.trailV.append(self.V)

    def in_circle(towers, C, r):
        distances = np.sqrt((towers[:, 0] - C[0]) ** 2 + (towers[:, 1] - C[1]) ** 2)
        return distances <= r


    def gauss_pdf(x, y, sigma, mean):
        xt = mean[0]
        yt = mean[1]
        temp = ((x - xt) ** 2 + (y - yt) ** 2) / (2 * sigma ** 2)
        val = exp(-temp)
        return val


    def compute_centroid(vertices, center, radius, sigma=0.2, mean=[0.8, 0.8], discretz_int=20):
        x_inf = np.min(vertices[:, 0])
        x_sup = np.max(vertices[:, 0])
        y_inf = np.min(vertices[:, 1])
        y_sup = np.max(vertices[:, 1])

        t_discretize = 1.0 / discretz_int

        dx = (x_sup - x_inf) / 2.0 * t_discretize
        dy = (y_sup - y_inf) / 2.0 * t_discretize
        dA = dx * dy
        A = 0
        Cx = 0
        Cy = 0

        for i in arange(x_inf, x_sup, dx):
            for j in arange(y_inf, y_sup, dy):
                if np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2) <= radius:
                    p = Path(vertices)
                    if p.contains_points([(i + dx, j + dy)])[0]:
                        A = A + dA * gauss_pdf(i, j, sigma, mean)
                        Cx = Cx + i * dA * gauss_pdf(i, j, sigma, mean)
                        Cy = Cy + j * dA * gauss_pdf(i, j, sigma, mean)

        Cx = Cx / A
        Cy = Cy / A

        return np.array([[Cx, Cy]])


    def bounded_voronoi(towers, center, radius):
        i = in_circle(towers, center, radius)
        points_center = towers[i, :]

        points_left = np.copy(points_center)
        points_left[:, 0] = center[0] - (points_left[:, 0] - center[0]) - 2 * radius

        points_right = np.copy(points_center)
        points_right[:, 0] = center[0] + (center[0] - points_right[:, 0]) + 2 * radius

        points_down = np.copy(points_center)
        points_down[:, 1] = center[1] - (points_down[:, 1] - center[1]) - 2 * radius

        points_up = np.copy(points_center)
        points_up[:, 1] = center[1] + (center[1] - points_up[:, 1]) + 2 * radius

        points = np.concatenate((points_center, points_left, points_right, points_down, points_up), axis=0)

        vor = sp.spatial.Voronoi(points)

        original_region_indices = vor.point_region[:len(points_center)]
        filtered_regions = [vor.regions[idx] for idx in original_region_indices if -1 not in vor.regions[idx]]

        vor.filtered_points = points_center
        vor.filtered_regions = filtered_regions

        return vor


    def centroid_region(vertices):
        A = 0
        C_x = 0
        C_y = 0
        for i in range(0, len(vertices) - 1):
            s = (vertices[i, 0] * vertices[i + 1, 1] - vertices[i + 1, 0] * vertices[i, 1])
            A = A + s
            C_x = C_x + (vertices[i, 0] + vertices[i + 1, 0]) * s
            C_y = C_y + (vertices[i, 1] + vertices[i + 1, 1]) * s
        A = 0.5 * A
        C_x = (1.0 / (6.0 * A)) * C_x
        C_y = (1.0 / (6.0 * A)) * C_y

        return np.array([[C_x, C_y]])


    def plot_gaussian_2d(mean, sigma, xlim=(-1, 2), ylim=(-1, 2), resolution=100):
        x = np.linspace(xlim[0], xlim[1], resolution)
        y = np.linspace(ylim[0], ylim[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.array([gauss_pdf(x, y, sigma, mean) for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = Z.reshape(X.shape)

        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label='Gaussian Value')