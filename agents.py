# /usr/bin/python3

import numpy as np
from utils import random_2d_vector, points_dist, line_generation, ort_line_gen, sat, compute_pos_des
import math


'''distanza tra i cani nella formazione --> l_formation
   distanza entro la quale i cani entrano in azione per recuperare le pecore --> limit_dist --> herd_dist_limit
   distzanza alla quale la pecore è considerata recuperata --> re_entered
   distnza entra la quale le pecore sentono la presenza dei cani --> d_limit'''

class Dog:
    k_repulsive = 50
    d_limit = 7./2. # distnza entra la quale le pecore sentono la presenza dei cani
    ########################################
    re_entered = 7.5 # distanza alla quale la pecore è considerata recuperata
    k_run = 25
    speed_limit = 80
    pursuit_l = 0.5
    l_formation = 7.5

    def __init__(self, P, simulation):
        self.formation_flag = False
        self.sim = simulation
        self.trail = []
        self.pd_trail = []
        self.color = "blue"
        self.color_trail = []
        self.marker = "X"
        self.trail_marker = []
        self.P = np.array(P)
        self.P_d = self.P
        self.V = np.array([0, 0])
        self.A = np.array([0, 0])
        ########################################
        self.chasing_dog = None
        self.chasing_sheep = None  # pecora che sta inseguendo
        self.int_point = False # intermediate point to generate chasing trajectory
        self.K = 50



    def step(self, dt):
        if not self.chasing_sheep: #Non sta inseguendo una pecora
            self.marker = "X"
            if self.formation_flag:
                self.V = self.dog_formation_control(self.l_formation)
            self.P = self.P + (self.V + self.sim.tracking_velocity) * dt
            #self.P = self.P + (self.V) * dt

            if self.sim.lose_sheeps and len(self.sim.dogs_form)>3:
                self.chasing_sheep = self.sim.lose_sheeps.pop(0)
                self.sim.dogs_form.remove(self)
                self.chasing_sheep.chased = True

        else:  #sta inseguendo una pecora
            self.marker = "*"
            sheep_pos = self.chasing_sheep.P
            if points_dist(sheep_pos, self.sim.center) <= self.re_entered:  #La pecora è rientrata
                self.chasing_sheep.chased = False
                self.int_point = False
                self.chasing_sheep = None
                self.sim.dogs_form.append(self)
                self.step(dt)
                return

            else:
                self.P_d = self.pure_pursuit(sheep_pos, self.P)  #La pecora non è ancora rientrata

            diff = self.P_d - self.P

            if diff.any():
                d = np.linalg.norm(diff)
                dir_diff = diff / d
                #k_lim = self.dog_to_dog_reaction()
                self.V = dir_diff * sat(self.k_run * d, self.speed_limit)
                self.P = self.P + self.V * dt
            else:
                self.P = self.P
        self.trail.append(self.P)
        #self.pd_trail.append(self.P_d)
        self.trail_marker.append(self.marker)
        self.color_trail.append(self.color)

    def dog_formation_control(self, l):
        dogs = self.sim.dogs_form
        n = len(dogs)
        if n == 6:
            # D for exagon formation
            r3 = np.sqrt(3)
            D =np.array([np.array([0., 1., r3, 2., r3, 1.]),
                              np.array([1., 0., 1., r3, 2., r3]),
                              np.array([r3, 1., 0., 1., r3, 2.]),
                              np.array([2., r3, 1., 0., 1., r3]),
                              np.array([r3, 2., r3, 1., 0., 1.]),
                              np.array([1., r3, 2., r3, 1., 0.])])
        elif n == 5:
            # D for pentagon formation
            diag5 = math.sin(math.radians(72)) / math.cos(math.radians(54))
            D =np.array([np.array([0., 1., diag5, diag5, 1.]),
                              np.array([1., 0., 1., diag5, diag5]),
                              np.array([diag5, 1., 0., 1., diag5]),
                              np.array([diag5, diag5, 1., 0., 1.]),
                              np.array([1., diag5, diag5, 1., 0.])])
        elif n == 4:
            # D for square formation
            r2 = np.sqrt(2)
            D = r2* np.array([np.array([0., 1, r2, 1.]),
                              np.array([1., 0, 1., r2]),
                              np.array([r2, 1., 0, 1.]),
                              np.array([1., r2, 1., 0])])
        elif n == 3:
            # D for triangle formation
            D = np.sqrt(3) * np.array([np.array([0., 1., 1.]),
                                  np.array([1., 0., 1.]),
                                  np.array([1., 1., 0.])])
        else:
            print('System Error')
        i = dogs.index(self)
        Ui = []
        for k in range(2):
            ui = 0  # init ctrl input
            xi = self.P[k]  # element i,k di t-1
            # for each neighbour
            for j in range(n):

                dog = dogs[j]
                xj = dog.P[k]  # neighbor at t-1
                diff = xi - xj
                dist = np.linalg.norm(self.P - dog.P)
                # noinspection PyUnboundLocalVariable
                wij = (dist - D[i, j]*l)
                ui += wij * diff
            Ui.append(-ui)
        return np.array(Ui)

    def pure_pursuit(self, pos, pos_d):
        m, q = line_generation(pos, self.sim.center)
        self.color = "yellow"
        if self.int_point is True and points_dist(pos_d, self.sim.center) <= points_dist(pos, self.sim.center):
            self.int_point = False
        if self.int_point is False:
            m, q = ort_line_gen(m, pos[0], pos[1])
            self.pursuit_l = 3
        else:
            self.pursuit_l = sat(points_dist(self.P, pos) / 3, 0.5, 50)
        x_d, y_d = compute_pos_des(pos, self.sim.center, m, q, self.pursuit_l)
        pd = np.array([x_d, y_d])
        if points_dist(pd, pos_d) <= 2:
            self.int_point = True
        return pd



class Sheep:
    ka = 50.
    c = 2.
    m = 1.
    speed_limit = 100

    def __init__(self, initial_area, center,  simulation) -> None:
        self.sim = simulation
        self.limit_dist = self.sim.herd_dist_limit

        self.chased = False

        self.trail = []
        self.trail_marker = []
        self.P = random_2d_vector() * initial_area + center
        print(self.P)
        self.V = np.array([0., 0.])
        self.A = np.array([0., 0.])
        self.trail.append(self.P)

    def step(self, dt, dogs_list):
        dist_from_cent = points_dist(self.sim.center, self.P)
        if self.chased:
            marker = "4"
        elif self not in self.sim.lose_sheeps:
            marker = "."
            if dist_from_cent > self.limit_dist:
                self.sim.lose_sheeps.append(self)
        else:
            marker = "^"
            if dist_from_cent <= self.limit_dist:
                self.sim.lose_sheeps.remove(self)
        self.trail_marker.append(marker)

        dog_repulsive = self.compute_dog_reaction(dogs_list)
        self.A = 1 / Sheep.m * (random_2d_vector() * Sheep.ka  # Random force
                                + dog_repulsive  # Repulsive dog force
                                - Sheep.c * self.V)  # Friction force
        self.V = sat(self.V  # Velocity saturation
                     + self.A * dt  # Acceleration
                     #+ self.sim.center_V * 0.12  # "Inertial principle"
                     , self.speed_limit)  # Velocity limit
        self.P = self.P + self.V * dt
        self.trail.append(self.P)
        '''if self.V[0] or self.V[1] >= 90:
            print("velocità troppo alta")'''

    def compute_dog_reaction(self, dogs_list):
        reaction = np.array([0., 0.])
        for dog in dogs_list:
            d_dog = self.P - dog.P
            d = np.linalg.norm(d_dog)
            if d <= dog.d_limit:
                k_linear = dog.d_limit - d
                k_hyper = 1. / d - 1. / dog.d_limit
                k = k_linear * 0.1 + k_hyper
                reaction += d_dog / d * k * dog.k_repulsive
        return reaction
