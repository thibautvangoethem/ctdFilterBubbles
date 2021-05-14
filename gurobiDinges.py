from gurobipy import *
import numpy as np


from scipy.linalg import solve
from gurobipy import *

import networkx as nx

from time import time
from pprint import pprint
import pickle
import csv
class Gurobi():
    def min_w_gurobi(self,z, lam, W0, reduce_pls, gam, existing):
        n = z.shape[0]
        m = Model("qcp")

        if existing:
            inds = [(i, j) for i in range(n) for j in range(n) if i > j and W0[i, j] > 0]
        else:
            inds = [(i, j) for i in range(n) for j in range(n) if i > j]
        x = m.addVars(inds, lb=0.0, name="x")

        # obj is min \sum_{i,j} wij (zi-zj)^2
        if existing:
            w = {(i, j): (z[i] - z[j]) ** 2 for i in range(n) for j in range(n) if i > j and W0[i, j] > 0}
        else:
            w = {(i, j): (z[i] - z[j]) ** 2 for i in range(n) for j in range(n) if i > j}

        obj_exp = x.prod(w)
        if reduce_pls:
            obj_exp += gam * x.prod(x)
        m.setObjective(obj_exp, GRB.MINIMIZE)
        print('added variables')

        # add constraints sum_j x[i,j] = di
        d = np.sum(W0, 0)
        for i in range(n):
            if existing:
                m.addConstr(quicksum(
                    [x[(j, i)] for j in range(i + 1, n) if W0[i, j] > 0] + [x[(i, j)] for j in range(i) if W0[i, j] > 0]) ==
                            d[i])
            else:
                m.addConstr(quicksum([x[(j, i)] for j in range(i + 1, n)] + [x[(i, j)] for j in range(i)]) == d[i])
        print('added first constraint')

        # add constraint \sum_{i,j} (wij - w0ij) < lam*norm(w0)**2
        t1=np.linalg.norm(W0)
        t2=lam
        rhs = (t1 * t2) ** 2

        if existing:
            m.addQConstr(quicksum(
                [x[(i, j)] * x[(i, j)] - 2 * x[(i, j)] * W0[i, j] + W0[i, j] * W0[i, j] for i in range(n) for j in range(n)
                 if i > j and W0[i, j] > 0]) <= rhs)
        else:
            m.addQConstr(quicksum(
                [x[(i, j)] * x[(i, j)] - 2 * x[(i, j)] * W0[i, j] + W0[i, j] * W0[i, j] for i in range(n) for j in range(n)
                 if i > j]) <= rhs)
        print('added second constraint')
        print('starting to optimize')
        m.optimize()

        W = np.zeros([n, n])
        for u in range(n):
            for v in range(n):
                if u > v:
                    if (existing and W0[u, v] > 0) or (not existing):
                        W[u, v] = x[(u, v)].X
                        W[v, u] = W[u, v]
        return W

    def min_z(self,W, s):
        # subset_s=s[0:]
        D = np.diag(np.sum(W, 0))
        L = D - W
        n = L.shape[0]
        temp=solve(L + np.eye(n), s)
        length=len(temp)
        for i in range(1, 51):
            temp[length - i] = s[length - i]
        return temp

    def min_z2(self, W, s, z):
        D = np.diag(np.sum(W, 0))
        n = D.shape[0]

        p1 = D + np.eye(n)
        p2 = np.matmul(W,z)+ s
        result = solve(p1,p2)
        length = len(result)
        for i in range(1,301):
            result[length-i] = s[length-i]
        return result

    def am(self,A, s, lam, reduce_pls=False, gam=0, max_iters=100, existing=False):
        # alternating minimization
        W = np.copy(A)
        z = self.min_z(W, s)  # minimize z first

        # polarization
        pls = [self.compute_pls(z)]

        # disagreement
        L = np.diag(np.sum(W, 0)) - W
        disaggs = [z.T.dot(L).dot(z)]

        # LOOP: first minimize W, then minimize z
        # then decide if we should exit
        i = 0
        flag = True
        while flag:
            print('iteration: {}'.format(i))
            # minimize W
            Wnew = self.min_w_gurobi(z, lam, A, reduce_pls=reduce_pls, gam=gam, existing=existing)

            # minimize z
            znew = self.min_z(Wnew, s)

            # exit condition
            if np.maximum(np.linalg.norm(z - znew), np.linalg.norm(Wnew - W)) < 5e-1 or i > max_iters - 1:
                flag = False

            # update z,W,i,pls
            z = znew
            W = Wnew
            i = i + 1
            pls.append(self.compute_pls(z))
            L = np.diag(np.sum(W, 0)) - W
            disaggs.append(z.T.dot(L).dot(z))
        return pls, disaggs, z, W

    def compute_pls(self,z):
        z_centered = z - np.mean(z)
        return z_centered.dot(z_centered)
if __name__ == '__main__':
    n_twitter = 548
    A = np.zeros([n_twitter, n_twitter])
    z_dict = {i: [] for i in range(n_twitter)}

    with open('reddit_twitter_data/Twitter/edges_twitter.txt', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for u, v in reader:
            A[int(u) - 1, int(v) - 1] = 1
            A[int(v) - 1, int(u) - 1] = 1

    # load opinions
    with open('reddit_twitter_data/Twitter/twitter_opinion.txt', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for u, v, w in reader:
            z_dict[int(u) - 1].append(float(w))

    # remove nodes not ocnnected in graph
    # not_connected = np.argwhere(np.sum(A, 0) == 0)
    # A = np.delete(A, not_connected, axis=0)
    # A = np.delete(A, not_connected, axis=1)
    # n_twitter = n_twitter - len(not_connected)

    # choose z, derive s
    z = [np.mean(z_dict[i]) for i in range(n_twitter)]
    z = np.array(z)
    lijstje = list()
    for i in range(len(A)):
        if (A[i, i] != A[0, 0]):
            print(A[i, i])

    L = np.diag(np.sum(A, 0)) - A
    s = (L + np.eye(n_twitter)).dot(z)
    s = np.minimum(np.maximum(s, 0), 1)

    Ltemp = L.transpose()

    # seems to run faster than Reddit, so we can include all lambdas in one list
    lam_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # results are dicts of the form
    # lambda -> (polarization at each iteration, disagreement at each iter, expressed opinions after NA dynamics, adjacency matrix after NA dynamics)
    g=Gurobi()
    res_dict = {}  # results for NA dynamics
    res2_dict = {}  # results for regularized NA dynamics

    max_iter = 7
    gam = 0.2
    for lam in lam_list:
        print('no fix')
        print('lam: {}'.format(lam))
        pls, disaggs, z, W = g.am(A, s, lam, reduce_pls=False, gam=0, max_iters=max_iter)
        print('with fix')
        print('lam: {}'.format(lam))
        pls2, disaggs2, z2, W2 = g.am(A, s, lam, reduce_pls=True, gam=gam, max_iters=max_iter)

        res_dict[lam] = (pls, disaggs, z, W)
        res2_dict[lam] = (pls2, disaggs2, z2, W2)
