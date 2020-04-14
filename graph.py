import networkx as nx
import matplotlib.pyplot as plt
import random
import math
 # or DiGraph, MultiGraph, MultiDiGraph, etc

# for i in range(100):
#     G.add_node(i)

# for i in range(100):
#     for j in range(100):
#         if i != j:
#             G.add_edge(i, j, weight=random.randint(1, 100))

# nx.draw(G)
# # plt.savefig("simple_path.png")  # save as png
# plt.show()  # display


def initialize_complex_map(p_zero, N, groups):
    import random
    import numpy as np
    #f = open("map.txt","w+")
    the_map = np.zeros((N, N))

    for i in range(0, N):
        for j in range(0, i):
            group_i = int(i/(N/groups))
            group_j = int(j/(N/groups))
            if random.randint(0, 100) > (100*p_zero) and abs(group_i - group_j) <= 1:
                rand = random.randint(0, 100)
                the_map[i][j] = rand
                the_map[j][i] = the_map[i][j]
                #G.add_edge(i, j, weight=rand)
                #f.write("%d %d %d\n"%(rand))

    np.savetxt("map.txt", the_map)
    new_data = np.loadtxt("map.txt")
    print(new_data)
    #nx.draw(G, with_labels=True)

    # plt.show()
    G = draw_graph(len(new_data), new_data)
    nx.draw(G)
    plt.show()


#ax = sns.heatmap(the_map)

# plt.show()
    return the_map


def draw_graph(N, the_map):
    import numpy as np
    G = nx.Graph()
    position=[]
    for i in range(N):
        random.seed(i)
        x = random.randint(1,100)
        random.seed(i+50)
        y = random.randint(1,100)
        G.add_node(i, pos=(x,y), status = 'not_visited')
        position.append((x,y))
    for i in range(0, N):
        for j in range(0, i):
            if the_map[i][j] > 0:
                x1, y1 = position[i]
                x2, y2 = position[j]
                dis = math.sqrt(((x1-x2)**2) + ((y1-y2)**2))
                G.add_edge(i, j, weight= dis, relation='not_visited')
                the_map[i][j] = dis
	
    #weight = nx.get_edge_attributes(G, 'weight')
    #nx.draw_networkx_edge_labels(weight)
	#nx.get_node_attributes
    np.savetxt("map.txt", the_map)
    return G, the_map


# if __name__ == "__main__":
#initialize_complex_map(0.8, 100, 1)
