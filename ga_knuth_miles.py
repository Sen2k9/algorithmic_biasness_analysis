import networkx as nx
import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import graph
import pdb
global start
global finish
global dis

start = 0
finish = 0
global node_mapping
global reverse_mapping
node_mapping = {}
reverse_mapping = {}


def print_pop(population):
    for i in population:
        print(i)


def initialize_map(p_zero, N):
    # first thing is to create the map that you're trying to navigate.  I will do this randomly.
    # This will be of the form of a adjacency matrix...
    # In other words, an NxN matrix where each row and column correspond to an intersection on a map
    # X_ij, then is equal to the amount of time that it takes to get from position i to position j
    # could also be considered a distance measure... but whatever is easier to think about.
    # practically, then we need a matrix that has numeric values in it...
    # there should be some paths that don't exist.  I will assign these a 0.
    # For instance, if you can't get directly from i to j, then X_ij = 0

    # The initialization needs some tuning parameters.  One is the proportion of 0's in the final result

    the_map = np.zeros((N, N))

    for i in range(0, N):
        for j in range(0, i):
            if random.random() > p_zero:
                the_map[i][j] = random.random()
                the_map[j][i] = the_map[i][j]

    return the_map

# Let's make a more complicated map that has at least 10 stops that have to be made and see what happens.


def initialize_complex_map(p_zero, N, groups):

    # the_map = np.zeros((N,N))

    # for i in range(0, N):
    #     for j in range(0, i):
    #         group_i = int(i/(N/groups))
    #         group_j = int(j/(N/groups))

    #         if random.randint(0,100) > (100*p_zero) and abs(group_i - group_j) <= 1:
    #             the_map[i][j] = random.randint(0,100)
    #             the_map[j][i] = the_map[i][j]
    import knuth_miles as km
    global node_mapping

    global reverse_mapping
    global dist
    dist = eval(
        input("At most how many miles of road network between cities you are interested? : "))

    the_map, node_mapping, _ = km.main(dist)
    for k, v in node_mapping.items():
        reverse_mapping[v] = k

    plot_best(the_map, [], dist)
    u = input("Input the start node : ")
    v = input("Input the finish node: ")
    global start
    global finish
    start = node_mapping[u]
    finish = node_mapping[v]
    """      
    the_map= np.loadtxt("map.txt")
    _, revised_map = graph.draw_graph(len(the_map), the_map)
    plot_best(revised_map, [])
    """

    #ax = sns.heatmap(the_map)

    # G = nx.Graph()
    # G = graph.draw_graph(len(the_map), the_map)

    # return revised_map
    return the_map


def create_starting_population(size, the_map):

    # this just creates a population of different routes of a fixed size.  Pretty straightforward.

    population = []

    for i in range(0, size):
        population.append(create_new_member(the_map))

    return population


def fitness(route, the_map):

    score = 0

    for i in range(1, len(route)):
        if (the_map[route[i-1]][route[i]] == 0) and i != finish:
            print("WARNING: INVALID ROUTE")
            print(route)
            print(the_map)
        score = score + the_map[route[i-1]][route[i]]

    return score


def crossover(a, b):

    # I initially made an error here by allowing routes to crossover at any point, which obviously won't work
    # you have to insure that when the two routes cross over that the resulting routes produce a valid route
    # which means that crossover points have to be at the same position value on the map

    common_elements = set(a) & set(b)

    if len(common_elements) == 2:
        return (a, b)
    else:
        common_elements.remove(start)
        common_elements.remove(finish)
        value = random.sample(common_elements, 1)

    cut_a = np.random.choice(np.where(np.isin(a, value))[0])
    cut_b = np.random.choice(np.where(np.isin(b, value))[0])

    new_a1 = copy.deepcopy(a[0:cut_a])
    new_a2 = copy.deepcopy(b[cut_b:])

    new_b1 = copy.deepcopy(b[0:cut_b])
    new_b2 = copy.deepcopy(a[cut_a:])

    new_a = np.append(new_a1, new_a2)
    new_b = np.append(new_b1, new_b2)

    return (new_a, new_b)

# this function need to be changed


def mutate(route, probability, the_map):

    new_route = copy.deepcopy(route)
    #new_route = np.zeros(1,dtype=int)

    for i in range(1, len(new_route)):
        if random.random() < probability:

            go = True

            while go:

                possible_values = np.nonzero(the_map[new_route[i-1]])
                proposed_value = random.randint(0, len(possible_values[0])-1)
                #new_route= np.append(new_route, possible_values[0][proposed_value])
                route = np.append(
                    new_route, possible_values[0][proposed_value])

                if new_route[i] == finish:
                    go = False
                else:
                    i += 1

    return new_route


def create_new_member(the_map):
    # here we are going to create a new route
    # the new route can have any number of steps, so we'll select that randomly
    # the structure of the route will be a vector of integers where each value is the next step in the route
    # Everyone starts at 0, so the first value in the vector will indicate where to attempt to go next.
    # That is, if v_i = 4, then that would correspond to X_0,4 in the map that was created at initialization

    # N is the size of the map, so we need to make sure that
    # we don't generate any values that exceed the size of the map

    N = len(the_map)
    route = np.array([start])

    #route = np.zeros(1, dtype=int)

    go = True

    i = 1
    # this loop creates all the route
    # while go:

    #     possible_values = np.nonzero(the_map[route[i-1]])
    #     proposed_value = random.randint(0,len(possible_values[0])-1)
    #     route = np.append(route, possible_values[0][proposed_value])

    #     if route[i] == N-1:
    #         go = False
    #     else:
    #         i += 1
    while go:
        possible_values = np.nonzero(the_map[route[i-1]])
        proposed_value = random.randint(0, len(possible_values[0])-1)
        route = np.append(route, possible_values[0][proposed_value])
        if route[i] == finish:
            go = False
        else:
            i += 1

    return route


def score_population(population, the_map):

    scores = []

    for i in range(0, len(population)):
        scores += [fitness(population[i], the_map)]

    return scores


def pick_mate(scores):

    array = np.array(scores)
    temp = array.argsort()
    # The empty_like() function is used to create a new array with the same shape and type as a given array.
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))

    fitness = [len(ranks) - x for x in ranks]

    cum_scores = copy.deepcopy(fitness)

    for i in range(1, len(cum_scores)):
        cum_scores[i] = fitness[i] + cum_scores[i-1]

    probs = [x / cum_scores[-1] for x in cum_scores]

    rand = random.random()

    for i in range(0, len(probs)):
        if rand < probs[i]:

            return i


def main():

    # parameters
    sparseness_of_map = 0.80
    size_of_map = 500
    population_size = 50
    number_of_iterations = 100
    number_of_couples = population_size//4  # 9
    number_of_winners_to_keep = 10
    mutation_probability = 0.1
    number_of_groups = 1

    # initialize the map and save it
    the_map = initialize_complex_map(
        sparseness_of_map, size_of_map, number_of_groups)

    # create the starting population
    population = create_starting_population(population_size, the_map)

    last_distance = 1000000000
    # for a large number of iterations do:

    for i in range(0, number_of_iterations):
        new_population = []

        # evaluate the fitness of the current population
        scores = score_population(population, the_map)
        # chose the best route in the population based on fitness

        best = population[np.argmin(scores)]

        number_of_moves = len(best)
        # calculated the distance of the best route
        distance = fitness(best, the_map)
        print('Iteration %i: Best so far is %i cities for a total distance of %f' % (
            i, number_of_moves, distance))

        print("Node/City coverage {}%".format((number_of_moves / 100) * 100))
        print()
        print("Node/City route is : [ ", end='')
        for j in range(len(best) - 1):
            print("{} => ".format(reverse_mapping[best[j]]), end='')
        print("{} ]".format(reverse_mapping[best[-1]]))
        print()

        # print(distance)
        # plot of the best route
        if distance != last_distance:
            # print('Iteration %i: Best so far is %i steps for a distance of %f' % (i, number_of_moves, distance))
            # print("Node route is : [ ", end='')
            # for k in range(len(best)-1):
            #     print("{} =>".format(reverse_mapping[best[k]]),end = '')
            # print("{} ]".format(reverse_mapping[best[-1]]))

            # print("Node coverage {}%".format((number_of_moves/100)*100))
            plot_best(the_map, best, dist)

        # allow members of the population to breed based on their relative score;
            # i.e., if their score is higher they're more likely to breed

        # randomly choose two member each time for crossover, and thereby created 9*2=18 members in new_population

        for j in range(0, number_of_couples):
            new_1, new_2 = crossover(
                population[pick_mate(scores)], population[pick_mate(scores)])
            new_population = new_population + [new_1, new_2]

        # mutate
        for j in range(0, len(new_population)):
            new_population[j] = np.copy(
                mutate(new_population[j], mutation_probability, the_map))

        # keep members of previous generation
        new_population += [population[np.argmin(scores)]]
        for j in range(1, number_of_winners_to_keep):
            keeper = pick_mate(scores)
            new_population += [population[keeper]]

        # add new random members
        while len(new_population) < population_size:
            new_population += [create_new_member(the_map)]

        # replace the old population with a real copy
        population = copy.deepcopy(new_population)

        last_distance = distance

    # plot the results


def plot_best(the_map, route, distance):
    global count

    G = nx.Graph()
    import knuth_miles as km

    _, _, G = km.main(distance)

    pos = nx.get_node_attributes(G, 'pos')
    population = nx.get_node_attributes(G, 'population')

    #weight = nx.get_edge_attributes(G, 'weight')
    if len(route) > 0:
        for each in route:
            G.add_node(reverse_mapping[each], pos=pos[reverse_mapping[each]],
                       population=population[reverse_mapping[each]], status='visited')
    #attr = {}
        for i in range(0, len(route) - 1):
            u = route[i]
            v = route[i+1]
            G.add_edge(reverse_mapping[u], reverse_mapping[v], weight=the_map[u]
                       [v], relation='visited')
            # print(the_map[u][v])
        #attr[(route[i],route[i+1])]= {'relation':'visited'}
    relation = nx.get_edge_attributes(G, 'relation')
    status = nx.get_node_attributes(G, 'status')
    # print(len(status))
    edge_dic = {'not_visited': 'silver', 'visited': 'darkblue'}

    node_dic = {'not_visited': 'yellow', 'visited': 'red'}

    #nx.draw_networkx(G, pos, arrows = True, edge_color=[edge_dic[x] for x in relation.values()], node_color=[node_dic[x] for x in status.values()])

    # draw with matplotlib/pylab
    plt.figure(figsize=(8, 8))
    # with nodes colored by degree sized by population
    node_color = [float(G.degree(v)) for v in G]
    nx.draw(G, pos, arrows=True,
            edge_color=[edge_dic[x] for x in relation.values()],
            node_size=[population[v] for v in G],
            node_color=[node_dic[x] for x in status.values()],
            with_labels=True)

    # scale the axes equally
    plt.xlim(-5000, 500)
    plt.ylim(-2000, 3500)
    count += 1
    figname = "roadmap" + str(count) + ".jpg"
    # savefig(fname, dpi=None, facecolor='w', edgecolor='w',
    #         orientation='portrait', papertype=None, format=None,
    #         transparent=False, bbox_inches=None, pad_inches=0.1,
    #         frameon=None, metadata=None)
    #         orientation='portrait', format='pdf'
    #plt.figure(figsize=(25, 15))
    plt.savefig(figname, orientation='landscape')
    plt.show()


if __name__ == "__main__":

    count = 0
    main()
