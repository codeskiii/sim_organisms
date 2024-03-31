#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division, print_function

import numpy as np

from math import atan2
from math import degrees
from math import sqrt
from random import uniform

from organism import organism
from food import food
from evolve import evolve

import pygame

import threading
from tqdm import tqdm

import json

#--- CONSTANTS ----------------------------------------------------------------+

with open("settings.json", "r") as file:
  settings = json.loads(file.read())

#--- FUNCTIONS ----------------------------------------------------------------+

def dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calc_heading(org, food):
    d_x = food.x - org.x
    d_y = food.y - org.y
    theta_d = np.degrees(np.arctan2(d_y, d_x)) - org.r
    if abs(theta_d) > 180:
        theta_d += 360
    return theta_d / 180


def simulate(settings, organisms, foods, gen, visualization=False):

    if visualization:
        # Initialize Pygame for live view
        pygame.init()
        screen = pygame.display.set_mode((int(abs(settings['x_min']) * 1.02), int(abs(settings['y_min']) * 1.02)))
        running = True

    total_time_steps = int(settings['gen_time'] / settings['dt'])

    #--- CYCLE THROUGH EACH TIME STEP ---------------------+
    # Consider using a more efficient progress bar library
    # for t_step in tqdm(total_time_steps / settings['dt'], desc="CYCLE"):
    for t_step in tqdm(range(total_time_steps)):
        if visualization:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not running:
                break

            # Clear the screen
            screen.fill((255, 255, 255))

        # UPDATE FITNESS FUNCTION
        for food in foods:
            for org in organisms:
                food_org_dist = dist(org.x, org.y, food.x, food.y)
                food_value = food.energy

                # DETERMINE IF THIS IS THE CLOSEST FOOD PARTICLE IN VISION
                if food_org_dist <= org.d_food and food_org_dist < settings["vision"]:
                    org.d_food = food_org_dist
                    org.r_food = calc_heading(org, food)
                    org.shortest_food_fitness = food_value

                # DETERMINE IF THIS IS FOOD PARTICLE WITH HIGHEST VALUE IN VISION
                if food_value >= org.r_food_highest and food_org_dist < settings["vision"]:
                    org.d_food_highest = food_org_dist
                    org.r_food_highest = calc_heading(org, food)
                    org.longest_food_fitness = food_value

                # UPDATE FITNESS FUNCTION
                if food_org_dist <= settings["r2"] * 0.80:
                    org.fitness += food.energy
                    food.respawn(settings)
                else:
                    if visualization:
                        # SHOW THEM
                        pygame.draw.circle(screen, (56, 178, 47), (int(food.x), int(food.y)), settings["r1"])

        id = 0
        for org_1 in organisms:
            for org_2 in organisms:
                org_dist = dist(org_1.x, org_1.y, org_2.x, org_2.y)
                org_1_value = org_1.fitness

                # DETERMINE IF THIS IS THE CLOSEST ORGANISM IN VISION
                if org_dist <= org.d_organism and org_dist < settings["vision"]:
                    org_2.d_organism = org_dist
                    org_2.r_organism = calc_heading(org_1, org_2)
                    org_2.shortest_org_fitness = org_1_value
                    org_2.shortest_org_id = id

                # DETERMINE IF THIS IS ORGANISM WITH HIGHEST VALUE IN VISION
                if org_1_value >= org.r_organism_highest and org_1_value < settings["vision"]:
                    org_2.d_organism_highest = org_dist
                    org_2.r_organism_highest = calc_heading(org_1, org_2)
                    org_2.highest_org_fitness = org_1_value
            id += 1

        # GET ORGANISM RESPONSE
        for org in organisms:
            org.think(settings)
            # UPDATE ORGANISMS POSITION AND VELOCITY
            org.update_r(settings)
            org.update_vel(settings)
            org.update_pos(settings)
            org.battle(settings, organisms)
            if visualization:
                pygame.draw.circle(screen, (255, 0, 0), (int(org.x), int(org.y)), settings["r2"])

        if visualization:
            # TAKE CARE OF LIVE VIEW
            pygame.display.update()

    if visualization:
        pygame.quit()

    return organisms

#--- MAIN ---------------------------------------------------------------------+

def run(settings):
    #--- POPULATE THE ENVIRONMENT WITH FOOD ---------------+
    foods = []
    for _ in range(settings['food_num']):
        foods.append(food(settings))

    #--- POPULATE THE ENVIRONMENT WITH ORGANISMS ----------+
    organisms = []
    for i in range(settings['pop_size']):
        # OLD --->> wih_init = np.random.uniform(-1, 1, (settings['hnodes'], settings['inodes']))     # mlp weights (input -> hidden)
        wih_init = {}
        for j in range(settings["hlayers"]+1):
            variance = 2.0 / (settings['inodes'] + settings['hnodes'])
            std_dev = np.sqrt(variance)
            wih_init[f"wih_{j}"] = np.random.normal(0, std_dev, size=(settings['hnodes'], settings['inodes']))

        variance = 2.0 / (settings['hnodes'] + settings['onodes'])
        std_dev = np.sqrt(variance)
        who_init = np.random.normal(0, std_dev, size=(settings['onodes'], settings['hnodes']))   # mlp weights (hidden -> output)

        organisms.append(organism(settings, wih_init, who_init, name=f'gen[{i}]'))

    gen_stats = {} # stats about every gen (gen num: best, organisms)
    evil_streak = 0 # sterak of worse best
    #--- CYCLE THROUGH EACH GENERATION --------------------+
    for gen in range(settings['gens']):
        # SIMULATE
        if gen != settings['gens'] - 1:
            threads = []
            for _ in range(settings["threads"]):
                t = threading.Thread(target=simulate, args=(settings, organisms.copy(), foods.copy(), gen))
                threads.append(t)
                t.start()

            # Join threads (wait for them to finish)
            for t in threads:
                t.join()
        else:
            organisms = simulate(settings, organisms, foods, gen, visualization=True)

        organisms, stats = evolve(settings, organisms, gen)
        print('\n >> GEN:', gen, 'BEST:', stats['BEST'], 'AVG:', stats['AVG'], 'WORST:', stats['WORST'])

        gen_stats[gen] = [stats["BEST"], organisms]
        if evil_streak <= settings["patient"]:
            bests = [gen_stat[0] for gen_stat in gen_stats.values()]
            global_best = max(bests)
            if gen_stats[gen][0] == global_best:
                evil_streak = 0
            else:
                evil_streak += 1
        else:
            bests = [gen_stat[0] for gen_stat in gen_stats.values()]
            best_index = bests.index(max(bests))
            print(len(gen_stats))
            organisms = gen_stats[best_index][1]
            evil_streak = 0
    pass


#--- RUN ----------------------------------------------------------------------+
if __name__ == '__main__':
    run(settings)
#--- END ----------------------------------------------------------------------+
