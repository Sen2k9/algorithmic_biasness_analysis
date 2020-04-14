

# ------------------------------------------------------------------------------

# Student name:
# Date:

# need some python libraries
import copy
from random import Random, random
from math import exp

import math
import numpy as np
global biased_node
biased_node = 0
global maxWeight
maxWeight = None
global n
n = 0
global value
global weights
value = None
weights = None
global item_number
item_number = 0

def evaluate(x):
    a = np.array(x)
    b = np.array(value)
    c = np.array(weights)
    #print(x[biased_node])


    totalValue = np.dot(a, b)  # compute the value of the knapsack selection
    # compute the weight value of the knapsack selection
    totalWeight = np.dot(a, c)
    #print(maxWeight)
    if totalWeight > maxWeight or not x[biased_node]:
        # return [totalWeight - maxWeight, totalWeight]
        return [0, totalWeight]
    else:
        # returns a list of both total value and total weight
        return [totalValue, totalWeight]


def OneflipNeighborhood(x):
    nbrhood = []

    for i in range(0, n):
        temp = list(x)
        nbrhood.append(temp)
        if nbrhood[i][i] == 1:
            nbrhood[i][i] = 0
        else:
            nbrhood[i][i] = 1

        # print(nbrhood)

    return nbrhood


# initial prop of items is proportion of items in initial solution
def Initial_solution(initial_prop_of_items):
    np.random.seed(1)  # seed
    x = np.random.binomial(1, initial_prop_of_items, size=n)
    #print("Random Initial Solution with one_prop: ", initial_prop_of_items )
    # print(x)
    # print("\n\n")
    return x


def print_results(solutionsChecked, f_best, x_best, coverage, best_count):
    #print("\nFinal number of solutions checked: ", solutionsChecked)
    print("Best value found: ", f_best[0])
    print("Weight is: ", f_best[1])
    #print("Total number of items selected: ", np.sum(x_best))
    print("Node coverage for the best soluton {}% ".format(
        int(np.sum(x_best)/n*100)))
    #print("Best solution: ", x_best)
    #print(coverage, best_count)
    print("Average Node coverage : {}%".format(
        int(coverage / best_count * 100)))
    print("\n")


def hill_climbing_with_random_walk(initial_prop_of_items, random_walk_prob,
                                   max_super_best_steps):

    print("Hill Climbing with random walk\n")
    # print("Initial Prop of items:", initial_prop_of_items)
    # print("Random walk probability", random_walk_prob)
    # print("Max No. of steps without improvement", max_super_best_steps)

    import random
    solutionsChecked = 0
    coverage = 0

    # x_curr will hold the current solution
    x_curr = Initial_solution(initial_prop_of_items)
    #print(x_curr)

    coverage = np.sum(x_curr)/n
    best_count = 1

    x_super_best = x_curr[:]  # x_best will hold the best solution

    # f_curr will hold the evaluation of the current soluton
    f_curr = evaluate(x_curr)[:]
    f_best = f_curr[:]  # Best solution in neighbourhood
    f_super_best = f_curr[:]
    # begin local search overall logic ----------------
    count = 0  # number of iteration with out improvement
    change = 0

    while (count < max_super_best_steps):

        # create a list of all neighbors in the neighborhood of x_curr
        Neighborhood = OneflipNeighborhood(x_curr)

        eeta = random.uniform(0, 1)
        if (eeta >= random_walk_prob):

            f_best[0] = 0
            for s in Neighborhood:  # evaluate every member in the neighborhood of x_curr
                solutionsChecked = solutionsChecked + 1

                # and (evaluate(s)[1]< maxWeight):
                if (evaluate(s)[0] > f_best[0]):
                    # find the best member and keep track of that solution
                    x_curr = s[:]
                    f_best = evaluate(s)[:]  # and store its evaluation

                    coverage += np.sum(x_curr)/n
                    best_count += 1
        else:
            x_curr = Neighborhood[random.randint(0, len(Neighborhood) - 1)]

        if (evaluate(x_curr)[0] > f_super_best[0]):  # to remember best solution
            f_super_best = evaluate(x_curr)[:]  # best solution so far
            x_super_best = x_curr[:]
            change = 1  # To record change

        count = count + 1  # counting number of iterations without improvement

        if(change == 1):  # Reseting count and change
            count = 0
            change = 0
    print_results(solutionsChecked, f_super_best,
                  x_super_best, coverage, best_count)
    # print("\n\n\n")


def simulated_Annealing(initial_prop_of_items, initial_temp,
                        iter_per_temp, final_temp):

    print("Simulated Annealing")
    # print("Initial Prop of items:", initial_prop_of_items)
    # print("Initial temp", initial_temp)
    # print("Final temp", final_temp)
    # print("Iteration per temperature", iter_per_temp)

    import random
    solutionsChecked = 0
    total_improvements = 0  # No of improving moves
    total_randomsteps = 0  # No of random moves

    # x_curr will hold the current solution
    x_curr = Initial_solution(initial_prop_of_items)
    x_best = x_curr[:]  # x_best will hold the best solution

    # f_curr will hold the evaluation of the current soluton
    f_curr = evaluate(x_curr)[:]
    f_best = f_curr[:]
    coverage = np.sum(x_curr)/n

    k = 0
    best_count = 1
    while (initial_temp/(k+1) > final_temp):  # Temp check

        m = 0     # Counting iteration in current temp

        improvements = 0   # improvements in current iteration
        randomsteps = 0  # Random steps in current iteration

        while (m < iter_per_temp):
            solutionsChecked = solutionsChecked + 1

            Neighborhood = OneflipNeighborhood(x_curr)

            # Selecting random neighbour
            s = Neighborhood[random.randint(0, len(Neighborhood)-1)]

            if (evaluate(s)[0] > f_curr[0]):  # choosing good value
                x_curr = s[:]
                f_curr = evaluate(s)[:]
                improvements += 1

            else:
                delta = evaluate(x_curr)[0] - \
                    evaluate(s)[0]  # choosing bad value
                eeta = random.uniform(0, 1)
                randomness = math.exp(-1 * delta * (k+1) / (initial_temp))
                #print(delta,"    ",randomness, eeta<randomness)
                if (eeta < randomness):
                    x_curr = s[:]
                    f_curr = evaluate(s)[:]
                    randomsteps = randomsteps + 1
            best_count += 1
            coverage += np.sum(x_curr)/n

            if(f_curr[0] > f_best[0]):  # Recording best value found so far
                x_best = x_curr[:]
                f_best = f_curr[:]

            m = m+1

        total_improvements = total_improvements + improvements  # total improvements
        total_randomsteps = total_randomsteps + randomsteps  # total random steps
        k = k + 1

    print(
        "Value :", f_best[0],
        "\nWeight :", f_best[1])
    #print("Total number of items selected: ", np.sum(x_best))
    print("Node coverage for the best soluton {}% ".format(
        int(np.sum(x_best)/n*100)))

    # print("Total random steps:", total_randomsteps,
    #       "Total improvements", total_improvements)

    print("Average Node Coverage : {}% ".format(int(coverage/best_count*100)))

    #print("Solutions checked", solutionsChecked)
    # print("\n\n\n")


def taboo_search(initial_prop_of_items, taboo_tenure,
                 max_super_best_steps):
    print("\nTaboo Search")
    # print("Initial Prop of items:", initial_prop_of_items)
    # print("taboo tenure", taboo_tenure)
    # print("Max super best steps", max_super_best_steps)

    solutionsChecked = 0

    # x_curr will hold the current solution
    x_curr = Initial_solution(initial_prop_of_items)
    x_best = x_curr[:]  # x_best will hold the best solution

    # f_curr will hold the evaluation of the current soluton
    f_curr = evaluate(x_curr)
    f_best = f_curr[:]  # Best solution in neighbourhood
    f_super_best = f_curr[:]  # Best solution so far

    taboo_list = [0]*n  # taboo status of each element in solution
    count = 0  # counting number of non improving steps

    coverage = np.sum(x_curr)/n
    best_count = 1

    while (count < max_super_best_steps):

        # create a list of all neighbors in the neighborhood of x_curr
        Neighborhood = OneflipNeighborhood(x_curr)
        neighbor = 0  # Number of element changed in current step
        f_best[0] = 0  # Reseting best neighbour value to zero
        for s in Neighborhood:  # evaluate every member in the neighborhood of x_curr
            solutionsChecked = solutionsChecked+1

            # and (evaluate(s)[1]< maxWeight):
            if (evaluate(s)[0] > f_best[0]) and (taboo_list[neighbor] == 0):  # tabu term added
                # find the best member and keep track of that solution
                x_curr = s[:]
                f_best = evaluate(s)[:]  # Best solution in neighbourhood
                neighbor_selected = neighbor  # neighbour selected in current step

                coverage += np.sum(x_curr)/n
                best_count += 1

            # Updating best solution fourd so far
            if (evaluate(s)[0] > f_super_best[0]):
                x_curr = s[:]
                f_best = evaluate(s)[:]
                f_super_best = evaluate(s)[:]
                x_super = s[:]
                neighbor_selected = neighbor
                change = 1
                # print(neighbor_selected)

            neighbor = neighbor + 1

        count = count + 1  # Counting number of steps with our improvement

        if(change == 1):  # Recording change status
            count = 0
            change = 0

        for i in range(0, len(taboo_list)-1):  # Updating taboo status of each item
            xx = taboo_list[i]
            if(xx > 0):
                taboo_list[i] = xx-1

        # Updating taboo status of selected item
        taboo_list[neighbor_selected] = taboo_tenure
        # print(taboo_list)

        # print(x_curr)
        # print(taboo_list)
        # print("Neighbor selected:", neighbor_selected, "Best_value", f_best[0])
        # print("Highest value found:", f_super_best[0])
        # print("\n")
    #print("\nFinal number of solutions checked: ", solutionsChecked)
    print("Best value found: ", f_super_best[0])
    print("Weight is: ", f_super_best[1])
    #print("Total number of items selected: ", np.sum(x_best))
    print("Node coverage for the best soluton {}% ".format(
        int(np.sum(x_super)/n*100)))
    #print("Best solution: ", x_best)
    #print(coverage, best_count)
    print("Average Node coverage : {}%".format(
        int(coverage / best_count * 100)))
    print("\n")

    # print("Final best", x_super, "\n",
    #       "Solutions checed", solutionsChecked,
    #       "Value:", f_best[0],
    #       "Weight:", f_best[1],
    #       "Highest value found:", f_super_best[0],
    #       "Best Solution:", x_super)
    # print("\n\n\n")


def run(number, v, w, threshold, itr, node):
    
    global biased_node
    biased_node = node
    global n
    n = number

    # to setup a random number generator, we will specify a "seed" value
    # need this for the random number generation -- do not change
    # seed = 5113
    # myPRNG = Random(seed)

# to get a random number between 0 and 1, use this:             myPRNG.random()
# to get a random number between lwrBnd and upprBnd, use this:  myPRNG.uniform(lwrBnd,upprBnd)
# to get a random integer between lwrBnd and upprBnd, use this: myPRNG.randint(lwrBnd,upprBnd)

# number of elements in a solution


# create an "instance" for the knapsack problem

    # for i in range(0, n):
    #     #value.append(myPRNG.uniform(10, 100))
    #     value.append(myPRNG.randint(10, 5*n))

    global value
    value = v

    # for i in range(0, n):
    #     #weights.append(myPRNG.uniform(5, 20))
    #     weights.append(myPRNG.randint(1, 15))

    global weights
    weights = w
    global maxWeight

    
    maxWeight = threshold
    global item_number

    item_number = np.arange(1, n + 1)

    # Sample Input Visualization

    # print('The list is as follows:')
    # print('Item No.   Weight   Value')
    # for i in range(item_number.shape[0]):
    #     print('{0}          {1}         {2}\n'.format(
    #         item_number[i], weights[i], value[i]))

    #print("values : {}".format(value))
    #print("weights : {}".format(weights))

    # change anything you like below this line ------------------------------------
    global solutionsChecked

    # monitor the number of solutions evaluated
    solutionsChecked = 0

    # 500 steps with out change in max value
    hill_climbing_with_random_walk(0.5, 0.1, itr)

    simulated_Annealing(initial_prop_of_items=0.5, initial_temp=100,
                        iter_per_temp=itr, final_temp=1)
    # Tabu tenure(how many times an already visited solution should be stopped) 30, #max steps without improvement 10,000
    taboo_search(0.1, 10, itr)
