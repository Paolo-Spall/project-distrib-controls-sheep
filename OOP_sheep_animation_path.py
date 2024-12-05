#/urs/bin/python3
import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from agents import Sheep, Dog
from utils import limit_decimal_places
from drone import Drone
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class Simulation:

    def __init__(self):
        self.sheeps = []
        self.dogs = []
        self.dogs_form = []
        self.drones = []
        self.iter_steps_n = 0
        self.center = np.array([0., 0.])
        #self.center_V = np.array([0, 0])
        self.lose_sheeps = []
        self.herd_dist_limit = 8.5 #distanza alla quale una pecotra viene considerata persa
        self.center_trail = []
        self.trajectory_trail = []

    def initial_configurations(self, dt, speed):
        self.tracking_speed = speed
        self.tracking_velocity = np.zeros(2)
        a.formation_control_on()
        for t in range(0, 20):
            for dog in self.dogs:
                dog.step(dt)
        for dog in self.dogs:
            dog.trail.clear()
            dog.trail_marker.clear()
            dog.color_trail.clear()
            dog.pd_trail.clear()
        self.compute_dog_center()

        
        


    def enter_path(self, points):
        """biuld the path for the sheeps to follow"""
        self.original_path = self.center_trail + points + self.center_trail
        print(self.original_path[:][0])
        print(self.original_path)
        self.tracked_point = points.pop(0) # firt point to be tracked
        self.path =  points + [self.center_trail[0]]


    def add_sheeps(self, n, initial_area):
        for i in range(n):
            self.sheeps.append(Sheep(initial_area, self.center_trail[0], self))
    
    def add_dog(self, P):
        dog = Dog(P, self)
        self.dogs.append(dog)
        self.dogs_form.append(dog)

    def add_drones(self, n, initial_area, center):
        for i in range(n):
            self.drones.append(n, initial_area, center, self)




    def compute_dog_center(self):
        count = 0
        center = numpy.array([0, 0])
        for dog in self.dogs_form:
            center = center+dog.P
        self.center = center / len(self.dogs_form)
        self.center_trail.append(self.center)

    def compute_formation_trajectory(self, i, c):
        dist = c[i] - self.center
        return dist

    #def compute_center_velocity(self, dt):
    #    if len(self.center_trail) > 1:
    #        self.center_V = (self.center_trail[-1] - self.center_trail[-2])/dt

    def formation_control_on(self):
        for dog in self.dogs_form:
            dog.formation_flag = True

    def formation_control_off(self):
        for dog in self.dogs_form:
            dog.formation_flag = False
    
    def track_path(self):
        """compute the common control input for the dogs to track the path"""
        distance = self.tracked_point - self.center_trail[-1]
        dist_norm = np.linalg.norm(distance)
        direction = distance / dist_norm # direction unit vector
        self.tracking_velocity = direction * self.tracking_speed
        if dist_norm < 0.5:
            self.tracked_point = self.path.pop(0)
        

    def step(self, dt, i):
        self.compute_dog_center()
        self.track_path()
        #self.compute_center_velocity(dt)
        for dog in self.dogs:
            dog.step(dt)
        for sheep in self.sheeps:
            sheep.step(dt,self.dogs)

    def simulate(self, T, dt):
        
        self.dt = dt
        t = 0.
        while self.path != []:
            self.step(dt, int(t/dt))
            self.iter_steps_n += 1
            t += dt
        print("time",t)

    def entities(self):
        return self.sheeps + self.dogs

    def animate(self, grid_dimension=50):
        for i in range(self.iter_steps_n):
            plt.clf()
            plt.xlim(-grid_dimension/2,grid_dimension/2)
            plt.ylim(-grid_dimension/2,grid_dimension/2)
            '''plt.xlim(self.center_trail[i][0]-25., self.center_trail[i][0]+25.)
            plt.ylim(self.center_trail[i][1]-25., self.center_trail[i][1]+25.)'''
            plt.scatter(self.center_trail[i][0],self.center_trail[i][1],marker="*")
            plt.gca().add_patch(patches.Circle((self.center_trail[i][0],self.center_trail[i][1]),
                                               radius=7.5, edgecolor='green', facecolor='none', linewidth=2))
            plt.gca().add_patch(patches.Circle((self.center_trail[i][0], self.center_trail[i][1]),
                                               radius=self.herd_dist_limit, edgecolor='red', facecolor='none', linewidth=2))
            for agent in self.entities():
                plt.scatter(agent.trail[i][0], agent.trail[i][1], marker=agent.trail_marker[i])
            plt.text(0.95, 0.95, limit_decimal_places(i*self.dt,2), fontsize=12, bbox=dict(facecolor='white', alpha=0.5),
                     horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
            original_path_arr = np.array(self.original_path)
            plt.plot(original_path_arr[:, 0], original_path_arr[:, 1], 'r--')
            plt.pause(0.0001)

        for agent in self.entities():
            trail_arr = np.array(agent.trail)
            plt.plot(trail_arr[:, 0], trail_arr[:, 1])
            #plt.plot(self.trajectory_trail[:, 0], self.trajectory_trail[:, 1], 'r--')
            plt.plot(np.array(self.center_trail)[:, 0], np.array(self.center_trail)[:, 1], 'b-')
        plt.show()

    def animate_coppelia(self):
        client = RemoteAPIClient()
        sim = client.getObject('sim')
        sim.startSimulation()

        plane_handle = sim.createPrimitiveShape(sim.primitiveshape_plane, [500, 500, 0])  # piano 500x500
        # sim.setBoolParameter(sim.boolparam_floor_visible, False)  # Disabilita il piano a scacchi
        sim.setShapeColor(plane_handle, None, sim.colorcomponent_ambient_diffuse, [0.35, 0.55, 0.35])  # Colore verde
        sim.setObjectPosition(plane_handle, -1, [0, 0, 0.02])  # poco sotto origine
        dog_instances = []
        sheep_instances = []

        for dog in self.dogs:
            # Crea una sfera e impostala nella posizione iniziale
            dog_handle = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [0.4, 0.4, 0.4])
            sim.setShapeColor(dog_handle, None, sim.colorcomponent_ambient_diffuse, [1, 0, 0])  # Colore rosso
            dog_instances.append([dog, dog_handle])

        for sheep in self.sheeps:
            sheep_handle = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [0.2, 0.2, 0.2])
            sim.setShapeColor(sheep_handle, None, sim.colorcomponent_ambient_diffuse, [1, 1, 0])
            sheep_instances.append([sheep, sheep_handle])

        instances_list = dog_instances + sheep_instances

        for i in range(self.iter_steps_n - 1):
            for py_agent, copp_agent in instances_list:
                sim.setObjectPosition(copp_agent, -1, py_agent.trail[i].tolist() + [0])
            time.sleep(0.01)


if __name__ == '__main__':
    a = Simulation()
    T = 1000.
    dt = 0.025
    tracking_speed=1.1
    points = [[-10, 10.], [0., 0.]]#, (-100, -200), (-50, -100), (0, -50), (50, 50), (100, 100), (150, 150), (200, 200)]
    points = [np.array(i) for i in points]
    a.add_dog([9.9-20, 0.4])
    a.add_dog([10.3-20, 0.5])
    a.add_dog([10.4-20, 0.6])
    a.add_dog([10-20, 0])
    a.add_dog([-10.1-20, 0.2])
    a.add_dog([-10-20, 0.3])
    a.initial_configurations(dt, tracking_speed)
    print("center_trail:", a.center_trail)
    a.add_sheeps(20, 3)
    a.enter_path(points)
    #a.center_trail.clear()
    '''a.formation_control_on()
    a.compute_dog_center()
    a.add_sheeps(20, initial_area=5, center=a.center_trail[0])
    a.center_trail.pop(0)'''
    
    a.simulate(T, dt)

    print(len(a.lose_sheeps))

    a.animate_coppelia()
    #a.animate(grid_dimension=80)
