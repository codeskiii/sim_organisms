from random import uniform, sample, random, randint, choices
from math import floor
from collections import defaultdict
import operator
from organism import organism

def evolve(settings, organisms_old, gen):

    elitism_num = int(floor(settings["initial_elitism"] * settings['pop_size']))
    new_orgs = settings['pop_size'] - elitism_num

    # Get statistics from the current generation
    stats = defaultdict(int)
    for org in organisms_old:
        if org.fitness > stats['BEST']:
            stats['BEST'] = org.fitness

        if org.fitness < stats['WORST']:
            stats['WORST'] = org.fitness

        stats['SUM'] += org.fitness
        stats['COUNT'] += 1

    stats['AVG'] = stats['SUM'] / stats['COUNT']

    # settings["initial_elitism"]: Keep best performing organisms
    orgs_sorted = sorted(organisms_old, key=operator.attrgetter('fitness'), reverse=True)
    organisms_new = [organism(settings, wih=orgs_sorted[i].wih, who=orgs_sorted[i].who, name=orgs_sorted[i].name) for i in range(elitism_num)]

    candidates = range(elitism_num)

    # Selection (truncation selection)
    random_indices = sample(candidates, 2)
    org_1, org_2 = orgs_sorted[random_indices[0]], orgs_sorted[random_indices[1]]

    # Generate new organisms
    for _ in range(new_orgs):

        # Mutation
        mutate = random()
        if mutate <= settings["initial_mutate"]:
            # mat_pick = randint(0, 1)  # Pick which weight matrix to mutate
            mutate_times = randint(0, settings["hlayers"] + 2) # n times matrices are mutated
            for _ in range(0, mutate_times):
                mat_pick = randint(0,1)
                layer = randint(0, settings["hlayers"])

                # Crossover
                crossover_weight = random()
                org_1.wih[f"wih_{layer}"] = (crossover_weight * org_1.wih[f"wih_{layer}"]) + ((1 - crossover_weight) * org_2.wih[f"wih_{layer}"])
                parrent = randint(0,1)
                wg = uniform(0.7, 1.3)
                if org_1.fitness * wg > org_2.fitness: wih_new = org_1.wih
                elif org_2.fitness * wg > org_1.fitness: wih_new = org_2.wih
                else:
                    parrent = randint(0,1)
                    if parrent == 1: wih_new = org_1.wih
                    elif parrent == 0: wih_new = org_2.wih
                who_new = (crossover_weight * org_1.who) + ((1 - crossover_weight) * org_2.who)

                if mat_pick == 0:
                    index_row = randint(0, settings['hnodes'] - 1)
                    for i in range(settings['inodes']):
                        wih_new[f"wih_{layer}"][index_row][i] = wih_new[f"wih_{layer}"][index_row][i] * uniform(0.9, 1.1)
                        wih_new[f"wih_{layer}"][index_row][i] = max(min(wih_new[f"wih_{layer}"][index_row][i], 1), -1)
                else:
                    index_row = randint(0, settings['onodes'] - 1)
                    index_col = randint(0, settings['hnodes'] - 1)
                    who_new[index_row][index_col] = who_new[index_row][index_col] * uniform(0.9, 1.1)
                    who_new[index_row][index_col] = max(min(who_new[index_row][index_col], 1), -1)

                organisms_new.append(organism(settings, wih=org_1.wih, who=who_new, name=f'gen[{gen}]-org[{_}]'))

    return organisms_new, stats
