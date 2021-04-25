import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
def read_data():
    connections = dict()
    opinions = dict()
    with open("reddit_twitter_data/Twitter/edges_twitter.txt", "r") as f:
        for line in f.readlines():
            u,v = line.replace("\n",'').split("\t")
            if u not in connections:
                connections[u] = []
            if v not in connections:
                connections[v] = []
            connections[u].append(v)
            connections[v].append(u)

    with open("reddit_twitter_data/Twitter/twitter_opinion.txt", "r") as f:
        for line in f.readlines():
            u,v,w = line.replace("\n",'').split("\t")
            if u not in opinions:
                opinions[u] = []
            opinions[u].append(float(w))
    return connections, opinions

def to_hex(rgba_color):
    red = int(rgba_color[0])
    green = int(rgba_color[1])
    blue = int(rgba_color[2])
    return '#%02x%02x%02x' % (red, green, blue)

if __name__ == '__main__':
    conn, op = read_data()

    G = nx.Graph()
    colors = []

    to_sort = []

    for key in conn:
        mean = np.mean(op[key])
        other_mean = [np.mean(op[tmp]) for tmp in conn[key]]
        color = mean*(len(other_mean)+1)
        for item in other_mean:
            color -= item
        color = (np.minimum(np.maximum(color,0),1)*2)-1
        if color > 0:
            to_sort.append(((key, {"color": pltc.to_hex([color, 0,0])}), color))

        else:
            to_sort.append(((key, {"color": pltc.to_hex([0, 0, abs(color)])}), color))

    to_sort.sort(key = lambda x: x[1])
    pos = dict()
    for item in to_sort:
        G.add_node(item[0][0])
        colors.append(item[0][1]["color"])
        pos[item[0][0]] = [np.random.rand(), item[1]]
    for key in conn:
        for node in conn[key]:
            G.add_edge(key, node, weight=1)
    nx.draw(G, node_color=colors, pos=pos)
    plt.show()