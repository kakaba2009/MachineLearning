#!/bin/python3
import sys
def DFS(graph, start):
    L = 0
    S = []
    Q = [start]
    while Q:
        current = Q.pop()
        if current not in S:
            S.append(current)
            L +=1
            if graph[current]:
                for n in graph[current]:
                    Q.append(n)
    return S, L
q = int(input().strip())
for a0 in range(q):
    n,m,x,y = input().strip().split(' ')
    n,m,x,y = [int(n),int(m),int(x),int(y)]
    graph_list = [[] for a in range(n)]
    cities = set(range(n))
    for a1 in range(m):
        city_1,city_2 = input().strip().split(' ')
        city_1,city_2 = [int(city_1),int(city_2)]
        graph_list[city_1-1].append(city_2-1)
        graph_list[city_2-1].append(city_1-1)
    #print(graph_list)
    subgraphs_list = []
    while len(cities) > 0:
        subgraph = 0
        b, subgraph = DFS(graph_list, next(iter(cities)))
        #print(b)
        cities = cities - set(b)
        subgraphs_list.append(subgraph)
    cost = 0
    for sub in subgraphs_list:
        cost += (min(y * (sub - 1) + x, x * (sub)))
    print(cost)
