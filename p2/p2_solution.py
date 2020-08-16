import numpy as np
from math import log2
import copy

'''
This script is using all 9 features (2,3,...,10) to create a tree, which serves as a template.
Todo: you need to modify this by using the several specified features to create your own tree 
Todo: you need to do the pruning yourself
Todo: you need to get all the output including the test results.
Todo: you also need to generate the tree of such the format in the writeup: 'if (x3 <= 6) return 2 .......'
'''

with open('breast-cancer-wisconsin.data', 'r') as f:
    a = [l.strip('\n').split(',') for l in f if '?' not in l]


a = np.array(a).astype(int)   # training data


def entropy(data):
    count = len(data)
    p0 = sum(b[-1] == 2 for b in data) / count
    if p0 == 0 or p0 == 1: return 0
    p1 = 1 - p0
    return -p0 * log2(p0) - p1 * log2(p1)


def infogain(data, fea, threshold):  # x_fea <= threshold;  fea = 2,3,4,..., 10; threshold = 1,..., 9
    count = len(data)
    d1 = data[data[:, fea - 1] <= threshold]
    d2 = data[data[:, fea - 1] > threshold]
    if len(d1) == 0 or len(d2) == 0: return 0
    return entropy(data) - (len(d1) / count * entropy(d1) + len(d2) / count * entropy(d2))


def find_best_split(data):
    c = len(data)
    c0 = sum(b[-1] == 2 for b in data)
    if c0 == c: return (2, None)
    if c0 == 0: return (4, None)
    ig = [[infogain(data, f, t) for t in range(1, 10)] for f in (2,4,5,6,7)]
    ig = np.array(ig)
    max_ig = max(max(i) for i in ig)
    if max_ig == 0:
        if c0 >= c - c0:
            return (2, None)
        else:
            return (4, None)
    ind = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
    fea, threshold = (2,4,5,6,7)[ind[0]], ind[1] + 1
    return (fea, threshold)


def split(data, node):
    fea, threshold = node.fea, node.threshold
    d1 = data[data[:,fea-1] <= threshold]
    d2 = data[data[:, fea-1] > threshold]
    return (d1,d2)


class Node:
    def __init__(self, fea, threshold):
        self.fea = fea
        self.threshold = threshold
        self.left = None
        self.right = None

def create_tree(data, node):
    d1,d2 = split(data, node)
    f1, t1 = find_best_split(d1)
    f2, t2 = find_best_split(d2)
    if t1 == None: 
        node.left = (f1, len(d1))
    else:
        node.left = Node(f1,t1)
        create_tree(d1, node.left)
    if t2 == None: 
        node.right = (f2, len(d2))
    else:
        node.right = Node(f2,t2)
        create_tree(d2, node.right)

def gen_tree(node, depth):
    if (type(node) is tuple):
        print("return ", node[0])
        return
    else:
        print()
        print(f"if (x{node.fea} <= {node.threshold}) ", end="")
    gen_tree(node.left, depth + 1)
    print("else ", end="")
    gen_tree(node.right, depth + 1)

def cut(node):
    if (type(node) is tuple):
        return node
    left = cut(node.left)
    right = cut(node.right)
    num = right[1] + left[1]
    if (left[1] > right[1]):
        return (left[0], num)
    else:
        return (right[0], num)

def prune_tree(node, depth):
    if (type(node) is tuple):
        return
    if (depth == 6):
        node.right = cut(node.right)
        node.left = cut(node.left)
    else:
        prune_tree(node.left, depth + 1)
        prune_tree(node.right, depth + 1)

def predict(data, node):
    while (type(node) is not tuple):
        if (data[node.fea - 1] <= node.threshold):
            node = node.left
        else:
            node = node.right
    return node[0]

ig = [infogain(a, 10, t) for t in range(1,10)]
ig = np.array(ig)
print(len(a[a[:,-1] == 2]))
print(len(a[a[:,-1] == 4]))
print(entropy(a))
print(max(ig))
print(len(a[np.logical_and(a[:,-2] <= np.argmax(ig) + 1, a[:,-1] == 2)]))
print(len(a[np.logical_and(a[:,-2] <= np.argmax(ig) + 1, a[:,-1] == 4)]))

ig = [[infogain(a, fea, t) for t in range(1,10)] for fea in (2,4,5,6,7)]
ig = np.array(ig)
ind = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
root = Node((2,4,5,6,7)[ind[0]], ind[1] + 1)
create_tree(a, root)
gen_tree(root, 0)

with open('test.data', 'r') as f:
    test = [l.strip('\n').split(',') for l in f if '?' not in l]
test = np.array(test).astype(int)

prediction = list(map(lambda x: predict(x, root), test))
print(prediction)
print(len(prediction))

prune_tree(root, 0)
gen_tree(root, 0)

prediction = list(map(lambda x: predict(x, root), test))
print(prediction)
print(len(prediction))