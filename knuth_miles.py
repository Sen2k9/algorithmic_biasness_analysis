

import re
import sys

import matplotlib.pyplot as plt
import networkx as nx


def miles_graph():
    """ Return the cites example graph in miles_dat.txt
        from the Stanford GraphBase.
    """
    # open file miles_dat.txt.gz (or miles_dat.txt)
    import gzip
    fh = gzip.open('knuth_miles.txt.gz', 'r')

    G = nx.Graph()
    G.position = {}
    G.population = {}

    cities = []
    for line in fh.readlines():
        line = line.decode()
        if line.startswith("*"):  # skip comments
            continue

        numfind = re.compile("^\d+")

        if numfind.match(line):  # this line is distances
            dist = line.split()
            for d in dist:
                G.add_edge(city, cities[i], weight=int(d))
                i = i + 1
        else:  # this line is a city, position, population
            i = 1
            (city, coordpop) = line.split("[")
            cities.insert(0, city)
            (coord, pop) = coordpop.split("]")
            (y, x) = coord.split(",")

            pos = (-int(x) + 7500, int(y) - 3000)
            G.add_node(city, pos=pos)
            # assign position - flip x axis for matplotlib, shift origin
            G.position[city] = pos
            G.population[city] = float(pop) / 1000.0
    return G


def main(distance):

    G = miles_graph()

    # print("Loaded miles_dat.txt containing 128 cities.")
    # print("digraph has %d nodes with %d edges"
    #       % (nx.number_of_nodes(G), nx.number_of_edges(G)))

    # make new graph of cites, edge if less then 300 miles between them
    H = nx.Graph()
    pos = nx.get_node_attributes(G, 'pos')
    weight = nx.get_edge_attributes(G, 'weight')
    #print(pos['Richfield, UT'])
    node_mapping = {}
    import numpy as np
    N = len(G.nodes)
    the_map = np.zeros((N, N))

    i = 0
    for v in G:
        H.add_node(v, pos=pos[v],
                   population=G.population[v], status='not_visited')
        node_mapping[v] = i
        i += 1
    # print(node_mapping)

    for (u, v, d) in G.edges(data=True):
        if d['weight'] < distance:
            H.add_edge(u, v, weight=d['weight'], relation='not_visited')
            i = node_mapping[u]
            j = node_mapping[v]
            the_map[i][j] = d['weight']
            the_map[j][i] = the_map[i][j]
    # print(the_map)
    # # draw with matplotlib/pylab
    # plt.figure(figsize=(8, 8))
    # # with nodes colored by degree sized by population
    # node_color = [float(H.degree(v)) for v in H]
    # nx.draw(H, G.position,
    #         node_size=[G.population[v] for v in H],
    #         node_color=node_color,
    #         with_labels=False)

    # # scale the axes equally
    # plt.xlim(-5000, 500)
    # plt.ylim(-2000, 3500)

    # plt.show()
    return the_map, node_mapping, H


# main()

# reference:
# Author: Aric Hagberg (hagberg@lanl.gov)

#    Copyright (C) 2004-2019 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
