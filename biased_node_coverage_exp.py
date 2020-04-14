import biased_knapsack_with_ga
import biased_all_heuristics
import numpy as np
import random
from six.moves import cPickle as pickle  # for performance
from data_generator import load_dict

n = 225
data = load_dict('data/data_{}.pkl'.format(n))


itr = 200
item_number = data["item"]

weight = data["weight"]
value = data["value"]
threshold = data["threshold"]  # Maximum weight that the bag of thief can hold
#max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
# index = None
# copy_value = value[:]
# copy_weight = weight[:]

# while True:
#     value_index = np.where(copy_value == np.min(copy_value))[0][0]

#     weight_index = np.where(copy_weight == np.max(copy_weight))[0][0]
#     #print(value_index, weight_index)
#     if value_index == weight_index:
#         index = value_index
#         break
#     else:
#         copy_value[value_index], copy_value[-1] = copy_value[-1], copy_value[value_index]
#         copy_value = np.delete(copy_value, -1)

#         copy_weight[weight_index], copy_weight[-1] = copy_weight[-1], copy_weight[weight_index]
#         copy_weight = np.delete(copy_weight, -1)


min_value = np.where(value == np.min(value))
#max_weight = np.where(weight == np.max(weight))
biased_node = min_value[0][0]
#print(weight, value, biased_node)
# print(value[biased_node])
#print(item_number, weight, value, threshold)

biased_knapsack_with_ga.run(n, value, weight, threshold, itr, biased_node)
print("\n")
biased_all_heuristics.run(n, value, weight, threshold, itr, biased_node)
