import knapsack_with_ga
import all_heuristics
import numpy as np
import random
from six.moves import cPickle as pickle  # for performance
from data_generator import load_dict

n = 100
data = load_dict('data/data_{}.pkl'.format(n))


itr = 200
item_number = data["item"]

weight = data["weight"]
value = data["value"]
threshold = data["threshold"]  # Maximum weight that the bag of thief can hold

#print(item_number, weight, value, threshold)

knapsack_with_ga.run(n, value, weight, threshold, itr)
print("\n")
all_heuristics.run(n, value, weight, threshold, itr)
