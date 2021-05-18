import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import sys


import pandas as pd  # for convinience
import seaborn as sns
from gurobiDinges import Gurobi
import csv
import pickle
import copy

def read_data():
    average_connections = 0

    connections = dict()
    opinions = dict()
    with open("reddit_twitter_data/Twitter/edges_twitter.txt", "r") as f:
        for line in f.readlines():
            u, v = line.replace("\n", '').split("\t")
            if u not in connections:
                connections[u] = set()
            if v not in connections:
                connections[v] = set()
            connections[u].add(v)
            connections[v].add(u)
            average_connections += 1

    with open("reddit_twitter_data/Twitter/twitter_opinion.txt", "r") as f:
        for line in f.readlines():
            u, v, w = line.replace("\n", '').split("\t")
            if u not in opinions:
                opinions[u] = []
            opinions[u].append(float(w))
    original_con = copy.deepcopy(connections)
    original_opinions = copy.deepcopy(opinions)
    outside_actor = True
    actors = 100

    if outside_actor:
        average_connections = 25
        key = len(connections) + 1
        original_length = len(connections)
        for i in range(actors):
            if i < (actors * 0.7):
                val = 0
            else:
                val = 1
            opinions[str(key)] = [val]

            new_connections = []
            while len(new_connections) < average_connections:
                random = np.random.randint(1, original_length+1)
                if random not in new_connections and random != 0:
                    if get_inate_opinion(original_con, str(random), original_opinions) < 0 and val == 0:
                        new_connections.append(random)
                        if str(key) not in connections:
                            connections[str(key)] = set()
                        # connections[str(key)].add(str(random))
                        connections[str(random)].add(str(key))

                    if get_inate_opinion(original_con, str(random), original_opinions) >= 0 and val == 1:
                        new_connections.append(random)
                        if str(key) not in connections:
                            connections[str(key)] = set()
                        # connections[str(key)].add(str(random))
                        connections[str(random)].add(str(key))

            key += 1


    return connections, opinions


def to_hex(rgba_color):
    red = int(rgba_color[0])
    green = int(rgba_color[1])
    blue = int(rgba_color[2])
    return '#%02x%02x%02x' % (red, green, blue)


def plot_graph(conn, op):
    G = nx.Graph()
    colors = []
    to_sort = []
    for key in conn:
        color = get_inate_opinion(conn, key, op)
        if color > 0:
            to_sort.append(((key, {"color": pltc.to_hex([color, 0, 0])}), color))

        else:
            to_sort.append(((key, {"color": pltc.to_hex([0, 0, abs(color)])}), color))
    to_sort.sort(key=lambda x: x[1])
    pos = dict()
    keys = []
    x = []
    x_connection = dict()
    x_lens = list()
    average_conn_list = list()
    diff_conn_list = list()
    for item in to_sort:
        if np.random.rand() >= 0:
            G.add_node(item[0][0])
            colors.append(item[0][1]["color"])
            pos[item[0][0]] = [np.random.rand(), item[1]]
            keys.append(item[0][0])
            x.append(item[1])
            x_connection[item[0][0]] = [item[1], len(conn[item[0][0]])]
            x_lens.append(len(conn[item[0][0]]))

            own_val=get_inate_opinion(conn,item[0][0], op)

            diff_counter=0
            if(own_val>0):
                positive=True
            else:
                positive=False
            # average_conn = 0
            # for connected in conn[item[0][0]]:
            #     connect_op = get_inate_opinion(conn, connected, op)
            #     if(connect_op>0 and not positive):
            #         diff_counter+=1
            #     if (connect_op <= 0 and positive):
            #         diff_counter += 1
            #     average_conn += connect_op
            # average_conn = average_conn / (len(conn[item[0][0]]))
            # average_conn_list.append(average_conn)

            diff_conn_list.append(diff_counter)
    for key in keys:
        for node in conn[key]:
            if node in keys:
                G.add_edge(key, node, weight=1)
    nx.draw(G, node_color=colors, pos=pos)
    plt.show()

    plt.hist(np.array(x), density=True, bins=20,label="Weighted amount of nodes with the given opinion")
    sns.kdeplot(x)
    plt.legend()
    plt.show()
    lijstje = list()
    for idx, item in enumerate(x):
        for i in range(x_lens[idx]):
            lijstje.append(item)
    plt.hist(np.array(lijstje), density=True, bins=20,label="Weighted amount of connections per opinion bin")
    plt.legend()
    plt.show()

    # plt.hist(np.array(average_conn_list), density=True, bins=20)
    # plt.show()

    lijstje = list()
    for idx, item in enumerate(x):
        for i in range(diff_conn_list[idx]):
            lijstje.append(item)
    plt.hist(np.array(lijstje), density=True, bins=20,label="Weighted amount of connections with differing opinions")
    plt.legend()
    plt.show()


def friedkin_johnson(conn, op):
    current_op = op
    new_op = dict()
    for i in range(1):
        for key in conn:
            new_val = 0
            own_opinion = get_inate_opinion(conn, key, op)
            new_val += own_opinion
            tmp = conn[key]
            w = 0
            for connected in conn[key]:
                if connected != key:
                    connect_op = get_inate_opinion(conn, connected, current_op)
                    w += connect_op
                    new_val += connect_op
            new_val = new_val /  (len(conn[key]) + 1.0)
            new_op[key] = new_val
        current_op = new_op
        new_op = dict()
    val = current_op['1']
    return current_op


def get_inate_opinion(conn, key, op):
    mean = np.mean(op[str(key)])
    opinion = np.minimum(np.maximum(mean, 0), 1)
    if mean == 0.0 or mean == 1.0:
        opinion = mean * 2 - 1
    else:
        other_mean = [np.mean(op[tmp]) for tmp in conn[str(key)] if int(tmp) <= 548]
        opinion = mean * (len(other_mean)+1)
        for item in other_mean:
            opinion -= item
        opinion = np.minimum(np.maximum(opinion, 0), 1) *2 -1
    return opinion


def plot_graph2(opinions, adjacency):
    G = nx.Graph()
    colors = []
    to_sort = []
    for index, op in enumerate(opinions):
        if op > 0:
            op = np.minimum(op, 1)
            to_sort.append(((index, {"color": pltc.to_hex([op, 0, 0])}), op))
        else:
            op = np.maximum(op, -1)
            to_sort.append(((index, {"color": pltc.to_hex([0, 0, abs(op)])}), op))

    to_sort.sort(key=lambda x: x[1])
    pos = dict()
    keys = []


    for item in to_sort:
        if np.random.rand() >= 0:

            G.add_node(item[0][0])
            colors.append(item[0][1]["color"])
            pos[item[0][0]] = [np.random.rand(), item[1]]
            keys.append(item[0][0])

    rounded = np.round(adjacency,5)
    t = np.nonzero(np.round(adjacency,5))
    for index in range(len(t[0])):
            G.add_edge(t[0][index], t[1][index], weight=rounded[t[0][index], t[1][index]])
    nx.draw(G, node_color=colors, pos=pos)
    plt.show()

if __name__ == '__main__':
    # old_stdout = sys.stdout
    # log_file = open("message.log", "w")
    # sys.stdout = log_file
    #
    #
    epsilon=[0.4]
    # g=Gurobi()
    # conn, op = read_data()
    # plot_graph(conn, op)
    # adj_matrix = np.zeros([len(conn), len(conn)])
    # inate_op = np.zeros([len(conn)])
    # for i in conn:
    #     temp_op=get_inate_opinion(conn, i, op)
    #     inate_op[int(i)-1]=temp_op
    # for i in conn:
    #     for j in conn[i]:
    #         adj_matrix[int(i)-1,int(j)-1]=1
    # temp = g.min_z(adj_matrix,inate_op)
    #
    # plot_graph2(temp,adj_matrix)

    # for idx,i in enumerate(epsilon):
    #     conn, op = read_data()
    #
    #     # plot_graph(conn,op)
    #
    #     adj_matrix = np.zeros([len(conn), len(conn)])
    #     for i in conn:
    #         for j in conn[i]:
    #             adj_matrix[int(i)-1,int(j)-1]=1
    #
    #
    #     inate_op = np.zeros([len(conn)])
    #     for i in range(len(conn)):
    #         temp_op=get_inate_opinion(conn, i+1, op)
    #         inate_op[int(i)]=temp_op
    #     temp=Gurobi()
    #
    #
    #     op = temp.min_z2(adj_matrix, inate_op, inate_op)
    #     for i in range(50):
    #         op = temp.min_z2(adj_matrix, inate_op, op)
    #
    #
    #
    #     plot_graph2(op, adj_matrix)
    #
    #     # temp.min_w_gurobi(op,0.4,conn,gam=0,existing=False,reduce_pl
    #     # s=False)
    #     pls, disaggs, z, W=temp.am(adj_matrix,inate_op, i,reduce_pls=False, gam=0.0, max_iters=7)
    #     #
    #     #
    #     pickle.dump([pls, disaggs, z, W],open('dump_external_with_regularization_aaaaa_'+str(idx)+'.dump','wb'))
    for idx, i in enumerate(epsilon):
        dinges=pickle.load(open('dump_external_with_regularization_aaaaa_'+str(idx)+'.dump', 'rb'))
        plot_graph2(dinges[2], dinges[3])

    # dinges = pickle.load(open('dump_external2.dump', 'rb'))
    # plot_graph2(dinges[2], dinges[3])
    # sys.stdout = old_stdout
    # log_file.close()

