
def cal_fitness(weight, value, population, threshold):
    fitness = np.empty(population.shape[0])  # creating array of fitness based on number of chromosome
    
    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * value)
        S2 = np.sum(population[i] * weight)
        if S2 <= threshold:
            fitness[i] = S1
        else:
            fitness[i] = 0
    return fitness.astype(int)


def selection(fitness, num_parents, population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))

        parents[i, :] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents


def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1] / 2)  # one-point crossover in the middle
    
    crossover_rate = 0.5
    i = 0
    while i < num_offsprings:
        
        parent1_index = i % parents.shape[0]
        parent2_index = (i+1) % parents.shape[0]
        x = rd.random()
        if x > crossover_rate:
            continue
        
        parent1_index = i % parents.shape[0]
        parent2_index = (i+1) % parents.shape[0]
        offsprings[i, 0:crossover_point] = parents[parent1_index,
                                                   0:crossover_point]
        offsprings[i, crossover_point:] = parents[parent2_index,
                                                  crossover_point:]
        i +=1
    return offsprings


def mutation(offsprings):
    mutants = np.empty((offsprings.shape))
    mutation_rate = 0.5
    for i in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[i, :] = offsprings[i, :]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0, offsprings.shape[1]-1)
        if mutants[i, int_random_value] == 0:
            mutants[i, int_random_value] = 1
        else:
            mutants[i, int_random_value] = 0
    return mutants


# passed argument as weight list, value list, initial_population(#chromosome, #genes), population size(#chromosome,#genes), #iteration, maximum weight
 
def optimize(weight, value, population, pop_size, num_generations, threshold):
    parameters, fitness_history = [], []
    num_parents = int(pop_size[0] / 2)  # dividing total #chromose(here 8) by 2
    
    num_offsprings = pop_size[0] - num_parents  # keeping other portion (here 8-4=4) of new population as offsprings
    total_coverage = 0
    count = 0
    
    for i in range(num_generations):  # for each generation/iteration

        
        fitness = cal_fitness(weight, value, population, threshold)  # calculate fitness for each population
        
        fitness_history.append(fitness)  # store the calculated fitness value for each generation
        
        parents = selection(fitness, num_parents, population) # selecting top fittest(here 4) individuals/chromosome

        offsprings = crossover(parents, num_offsprings) # creates offspring

        mutants = mutation(offsprings)  # creates random mutation
        
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0] :, :] = mutants
        #print('Last generation: \n{}\n'.format(population))
        fitness_last_gen = cal_fitness(weight, value, population, threshold)
        
        #print('Fitness of the {} generation: \n{}\n'.format(i,fitness_last_gen))
        max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
        coverage = sum(
            list(population[max_fitness[0][0], :]))/population.shape[1]
        
        for each in parents:
            total_coverage += sum(each) / population.shape[1]
            count +=1
            
        
        # print("maximum value {}".format(np.max(fitness_last_gen)))
        # print(population[max_fitness[0][0], :])

        # print("node coverage {}".format(coverage))
        count +=1
        total_coverage += coverage
        
        # if np.max(fitness_last_gen) > f_best:
        #         change = 1
       
        # if change == 1:
        #     #i = 0
        #     change = 0
        


    #print('Last generation: \n{}\n'.format(population))
    fitness_last_gen = cal_fitness(weight, value, population, threshold)
    #print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    print("Optimized maximum value :{}".format(np.max(fitness_last_gen)))
    print("Node coverage for optimized result :{}%".format(int(sum(list(population[max_fitness[0][0],:]))/population.shape[1]*100)))
    #print(population[max_fitness[0][0],:])
    parameters.append(population[max_fitness[0][0], :])
    return parameters, fitness_history, total_coverage, count


import numpy as np
import pandas as pd
import random as rd
from random import randint, Random
import matplotlib.pyplot as plt


def run(n, value, weight, threshold,itr):


    # Problem initializaton
    # np.random.seed(5113)
    item_number = np.arange(1, n + 1)

    # weight = np.random.randint(1, 15, size=n)
    # value = np.random.randint(10, 5*n, size=n)
    knapsack_threshold = threshold  # Maximum weight that the bag of thief can hold

    # Sample Input Visualization


    # print('The list is as follows:')
    # print('Item No.   Weight   Value')
    # for i in range(item_number.shape[0]):
    #     print('{0}          {1}         {2}\n'.format(
    #         item_number[i], weight[i], value[i]))


    solutions_per_pop = n  # Chromosome of possible solution size

    # (number of chromosome, number of genes)
    pop_size = (solutions_per_pop, item_number.shape[0])


    print('Population size = {}'.format(pop_size))

    # randomly creating (8,10) matrix with 1/0 value
    initial_population = np.random.randint(2, size=pop_size)

    initial_population = initial_population.astype(int)  # converting the values as integer

    num_generations = itr  # number of iteration

    #print('Initial population: \n{}'.format(initial_population))
    print("\nGenetic Algorithm")

    #GA Algorithm called
    parameters, fitness_history, total_coverage, count = optimize(
        weight, value, initial_population, pop_size, num_generations, knapsack_threshold)

    print("Weight : {} ".format(np.sum(parameters[0]*weight)))
    #print('The optimized parameters for the given inputs are: \n{}'.format(parameters[0]))
    selected_items = item_number * parameters
    #print('\nSelected items that will maximize the knapsack without breaking it:')
    # for i in range(selected_items.shape[1]):
    #     if selected_items[0][i] != 0:
    #         print('{} '.format(selected_items[0][i]), end="")
    
    

    print("Average node coverage for {} generation : {}%".format(
        num_generations, int(total_coverage/count*100)))





