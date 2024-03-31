from random import uniform
import numpy as np
from math import cos
from math import radians
from math import sin

class organism():
    def __init__(self, settings, wih=None, who=None, name=None):

        self.x = uniform(settings['x_min'], settings['x_max'])  # position (x)
        self.y = uniform(settings['y_min'], settings['y_max'])  # position (y)

        self.r = uniform(0, 360)              # orientation  [0, 360]
        self.v = uniform(0, settings['v_max'])  # velocity     [0, v_max]
        self.dv = uniform(-settings['dv_max'], settings['dv_max'])  # dv

        self.d_food = settings["vision"] + 1  # distance to nearest food
        self.d_organism = settings["vision"] + 1
        self.d_organism_highest = settings["vision"] + 1
        self.d_food_highest = settings["vision"] + 1

        self.r_food = 0    # orientation to nearest food
        self.r_food_highest = 0 # orientation to nearest food with highest value
        self.r_organism = 0
        self.r_organism_highest = 0

        self.shortest_food_fitness = 0
        self.longest_food_fitness = 0

        self.shortest_org_fitness = 0
        self.highest_org_fitness = 0

        self.shortest_org_id = 0

        self.fitness = 1   # fitness (food count)

        self.wih = wih
        self.who = who

        self.name = name

    # NEURAL NETWORK
    def think(self, settings):

        # activation function
        af = lambda x: np.tanh(x)

        combined_data = (
            [self.d_food,
             self.r_food,
             self.d_food_highest,
             self.r_food_highest,

             self.shortest_food_fitness,
             self.longest_food_fitness,

             self.d_organism,
             self.r_organism,
             self.d_organism_highest,
             self.r_organism_highest,

             self.shortest_org_fitness,
             self.highest_org_fitness,

             self.r,
             self.v]
        )

        # NO 1
        hs = af(np.dot(self.wih["wih_0"], combined_data))  # hidden layer no 1

        for i in range(1, settings["hlayers"]):
            hs = af(np.dot(self.wih[f"wih_{i}"], hs))

        out = af(np.dot(self.who, hs))         # output layer

        # UPDATE dv AND dr WITH MLP RESPONSE
        self.nn_dv = float(out[0]) # [-1, 1] (accelerate=1, deaccelerate=-1)
        self.nn_dr = float(out[1]) # [-1, 1] (left=1, right=-1)
        self.k = float(out[2]) # [-1, 1] (kill=1, notkill=-1)

    # UPDATE HEADING
    def update_r(self, settings):
        self.r += self.nn_dr * settings['dr_max'] * settings['dt']
        self.r = self.r % 360

    # UPDATE VELOCITY
    def update_vel(self, settings):
        self.v += self.nn_dv * settings['dv_max'] * settings['dt']
        if self.v < 0: self.v = 0
        if self.v > settings['v_max']: self.v = settings['v_max']

    # UPDATE POSITION
    def update_pos(self, settings):
        dx = self.v * cos(radians(self.r)) * settings['dt']
        dy = self.v * sin(radians(self.r)) * settings['dt']

        if self.x <= settings["x_max"] or self.x >= settings["x_min"]:
            self.x += dx
        else:
            self.x = randint(settings["x_min"], settings["x_max"])

        if self.y <= settings["y_max"] or self.y >= settings["y_min"]:
            self.y += dy
        else:
            self.y = randint(settings["y_min"], settings["y_max"])

    def battle(self, settings, organisms):
        if self.k == 1:
            if self.d_organism <= settings["r2"] * 1.5:
                if self.fitness >= organisms[self.shortest_org_id].fitness:
                    self.fitness += organisms[self.shortest_org_id].fitness
                    organisms[self.shortest_org_id].fitness = 1
                else:
                    self.fitness = self.fitness * 0.5
                    organisms[self.shortest_org_id].fitness += self.fitness
