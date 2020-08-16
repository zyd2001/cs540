import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import csv

data = {}

with open('time_series_covid19_deaths_global.csv', newline='') as file:
    file.readline()
    reader = csv.reader(file)
    ca = []
    for row in reader:
        if (row[1] in data):
            data[row[1]] += np.array(row[4:], int)
        else:
            data[row[1]] = np.array(row[4:], int)

length = len(data['US'])

print(data['US'])
print(data['Canada'])
diff = {'US':[], 'Canada': []}
for i in range(1, length):
    diff['US'].append(data['US'][i] - data['US'][i - 1])
for i in range(1, length):
    diff['Canada'].append(data['Canada'][i] - data['Canada'][i - 1])
print(diff)

def f(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

def fit(d):
    (a, b, c), _ = opt.curve_fit(f, x, d, p0=[max(d), 1, length // 2], method='dogbox')
    return (a, b, c)

x = range(length)

para = {}

for key, value in data.items():
    try:
        para[key] = (fit(value))
    except RuntimeError as e:
        pass

for i in para.values():
    print(', '.join('{:.2f}'.format(x) for x in i))

def dist(p1, p2):
    return np.sqrt(sum((p1[i] - p2[i])**2 for i in range(len(p1))))

def single(c1, c2):
    res = float('inf')
    for i1 in c1:
        for i2 in c2:
            d = dist(para[i1], para[i2])
            if d < res:
                res = d
    return res

def complete(c1, c2):
    res = 0.
    for i1 in c1:
        for i2 in c2:
            d = dist(para[i1], para[i2])
            if d > res:
                res = d
    return res

k = 8

def print_clusters(c):
    l = []
    for i in para.keys():
        for j in range(len(c)):
            if (i in c[j]):
                l.append(j)
    print(', '.join(map(str, l)))

clusters = [{d} for d in para.keys()]

for _ in range(len(para) - k):
    d = float('inf')
    best_pair = (None, None)
    for i in range(len(clusters)-1):
        for j in range(i+1, len(clusters)):
            if single(clusters[i], clusters[j]) < d:
                d = single(clusters[i], clusters[j])
                best_pair = (i,j)
    new_clu = clusters[best_pair[0]] | clusters[best_pair[1]]
    clusters = [clusters[i] for i in range(len(clusters)) if i not in best_pair]
    clusters.append(new_clu)

print(clusters)
print_clusters(clusters)

clusters = [{d} for d in para.keys()]

for _ in range(len(para) - k):
    d = float('inf')
    best_pair = (None, None)
    for i in range(len(clusters)-1):
        for j in range(i+1, len(clusters)):
            if complete(clusters[i], clusters[j]) < d:
                d = complete(clusters[i], clusters[j])
                best_pair = (i,j)
    new_clu = clusters[best_pair[0]] | clusters[best_pair[1]]
    clusters = [clusters[i] for i in range(len(clusters)) if i not in best_pair]
    clusters.append(new_clu)

print(clusters)
print_clusters(clusters)

import copy
countries = list(para.keys())
def center(cluster):
    return np.average([para[c] for c in cluster], axis=0)
np.random.seed(456) # in some case there may be empty cluster and will cause exception
init_num = np.random.choice(len(countries) - 1, k)
clusters = [{countries[i]} for i in init_num]
while True:
    new_clusters = [set() for _ in range(k)]
    centers = [center(cluster) for cluster in clusters]
    for c in countries:
        clu_ind = np.argmin([dist(para[c], centers[i]) for i in range(k)])
        new_clusters[clu_ind].add(c)
    if all(new_clusters[i] == clusters[i] for i in range(k)):
        break
    else:
        clusters = copy.deepcopy(new_clusters)


print(clusters)
print_clusters(clusters)

for i in clusters:
    print(','.join('{:.4f}'.format(x) for x in center(i)))

distortion = 0.
for i in clusters:
    c = tuple(round(x, 4) for x in center(i))
    print(c) 
    distortion = distortion + sum(map(lambda x: dist(tuple(round(y, 2) for y in para[x]), c) ** 2, i))

print(distortion)