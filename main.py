import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltc

import pandas as pd  # for convinience
import seaborn as sns
from gurobiDinges import Gurobi
import csv
import pickle
def read_data():
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

    with open("reddit_twitter_data/Twitter/twitter_opinion.txt", "r") as f:
        for line in f.readlines():
            u, v, w = line.replace("\n", '').split("\t")
            if u not in opinions:
                opinions[u] = []
            opinions[u].append(float(w))
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
            average_conn = 0
            for connected in conn[item[0][0]]:
                connect_op = get_inate_opinion(conn, connected, op)
                if(connect_op>0 and not positive):
                    diff_counter+=1
                if (connect_op <= 0 and positive):
                    diff_counter += 1
                average_conn += connect_op
            average_conn = average_conn / (len(conn[item[0][0]]))
            average_conn_list.append(average_conn)
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
    for i in range(10):
        new_val = 0
        for key in conn:
            own_opinion = get_inate_opinion(conn, key, op)
            new_val += own_opinion
            w = 1.0 / (len(conn[key]) + 1.0)
            for connected in conn[key]:
                connect_op = get_inate_opinion(conn, connected, current_op) * w
                new_val += connect_op
            new_val = new_val / (len(conn[key]) + 1.0)
            new_op[key] = new_val
        current_op = new_op
        new_op = dict()
    return current_op


def get_inate_opinion(conn, key, op):
    mean = np.mean(op[key])
    other_mean = [np.mean(op[tmp]) for tmp in conn[key]]
    opinion = mean * (len(other_mean)+1)
    for item in other_mean:
        opinion -= item
    opinion = np.minimum(np.maximum(opinion, 0), 1)
    return opinion


if __name__ == '__main__':

    conn, op = read_data()
    adj_matrix = np.zeros([len(conn), len(conn)])
    for i in conn:
        for j in conn[i]:
            adj_matrix[int(i)-1,int(j)-1]=1

    z = [np.mean(op[str(i+1)]) for i in range(len(conn))]
    z = np.array(z)
    L = np.diag(np.sum(adj_matrix, 0)) - adj_matrix
    inate_op2 = (L + np.eye(len(conn))).dot(z)
    inate_op2 = np.minimum(np.maximum(inate_op2, 0), 1)

    inate_op = np.zeros([len(conn)])
    for i in conn:
        temp_op=get_inate_opinion(conn, i, op)
        inate_op[int(i)-1]=temp_op
    temp=Gurobi()
    # temp.min_w_gurobi(op,0.2,conn,gam=0,existing=False,reduce_pls=False)
    pls, disaggs, z, W=temp.am(adj_matrix,inate_op2, 0.1,reduce_pls=False, gam=0, max_iters=1)
    # plot_graph(conn, op)
    # temp = friedkin_johnson(conn, op)

    #
    # plot_graph(conn, temp)
    pickle.dump([pls, disaggs, z, W],open('dump.dump','wb'))