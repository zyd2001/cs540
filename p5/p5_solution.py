import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# %matplotlib inline
import copy
import math
import heapq

'''
The below script is based on a 55 * height - 1 maze. 
Todo:
	1. Plot the maze and solution in the required format.
	2. Implement DFS algorithm. (I've given you the BFS below)
	3. Implement A* with Euclidean distance. (I've given you the one with Manhattan distance)

'''



width, height = 57, 58
X, Y = 14, 2

ori_img = mpimg.imread('1.png')
img = ori_img[:,:,0]

class Cell:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.succ = ''
        self.action = ''  # which action the parent takes to get this cell'
        self.marked = False
        self.searched = 0
cells = [[Cell(i,j) for j in range(width)] for i in range(height)]

def plot(cells):
    ceil = '+--' * (width // 2) + '+  ' + '+--' * (width // 2) + '+'
    s = ceil + '\n'
    for row in cells:
        for c in row:
            if 'L' in c.succ:
                s = s + '   '
            else:
                s = s + '|  '
        s = s + '|\n'
        for c in row:
            if 'D' in c.succ:
                s = s + '+  '
            else:
                s = s + '+--'
        s = s + '+\n'
    print(s)

def plot_solution(cells):
    ceil = '+--' * (width // 2) + '+##' + '+--' * (width // 2) + '+'
    s = ceil + '\n'
    for j in range(len(cells) - 1):
        row = cells[j]
        if row[0].marked:
            s = s + '|##'
        else:
            s = s + '|  '
        for i in range(1, len(row)):
            if 'L' in row[i].succ:
                if row[i - 1].marked and row[i].marked:
                    s = s + '#'
                else:
                    s = s + ' '
            else:
                s = s + '|'
            if row[i].marked:
                s = s + '##'
            else:
                s = s + '  '
        s = s + '|\n'
        for i in range(len(row)):
            if 'D' in row[i].succ:
                if cells[j + 1][i].marked and row[i].marked:
                    s = s + '+##'
                else:
                    s = s + '+  '
            else:
                s = s + '+--'
        s = s + '+\n'

    row = cells[-1]
    if row[0].marked:
        s = s + '|##'
    else:
        s = s + '|  '
    for i in range(1, len(row)):
        if 'L' in row[i].succ:
            if row[i - 1].marked and row[i].marked:
                s = s + '#'
            else:
                s = s + ' '
        else:
            s = s + '|'
        if row[i].marked:
            s = s + '##'
        else:
            s = s + '  '
    s = s + '|\n'
    s = s + ceil
    print(s)

def mark(cells, action):
    x, y = 0, width // 2
    for i in action:
        cells[x][y].marked = True
        if i == 'U':
            x = x - 1
        if i == 'D':
            x = x + 1
        if i == 'R':
            y = y + 1
        if i == 'L':
            y = y - 1
    cells[height - 1][width // 2].marked = True

for i in range(height):
    succ = []
    for j in range(width):
        s = ''
        c1, c2 = i * 16 + 8, j * 16 + 8
        if img[c1-8, c2] == 1: s += 'U'
        if img[c1+8, c2] == 1: s += 'D'
        if img[c1, c2-8] == 1: s += 'L'
        if img[c1, c2+8] == 1: s += 'R'
        cells[i][j].succ = s
        succ.append(s)
# 2    

cells_to_plot = copy.deepcopy(cells)
plot(cells_to_plot)

cells[0][width // 2].succ = cells[0][width // 2].succ.replace('U', '')
cells[height - 1][width // 2].succ = cells[height - 1][width // 2].succ.replace('D', '')

for row in cells:
    print(','.join([x.succ for x in row]))

# bfs
visited = set()
s1 = {(0,width // 2)}
s2 = set()
while (height - 1,width // 2) not in visited:
    for a in s1:
        visited.add(a)
        i, j = a[0], a[1]
        cells[i][j].searched = 1
        succ = cells[i][j].succ
        if 'U' in succ and (i-1,j) not in (s1 | s2 | visited): 
            s2.add((i-1,j))
            cells[i-1][j].action = 'U'
        if 'D' in succ and (i+1,j) not in (s1 | s2 | visited): 
            s2.add((i+1,j))
            cells[i+1][j].action = 'D'
        if 'L' in succ and (i,j-1) not in (s1 | s2 | visited): 
            s2.add((i,j-1))
            cells[i][j-1].action = 'L'
        if 'R' in succ and (i,j+1) not in (s1 | s2 | visited): 
            s2.add((i,j+1))
            cells[i][j+1].action = 'R'     
    s1 = s2
    s2 = set()
    
cur = (height - 1,width // 2)
s = ''
seq = []
while cur != (0,width // 2):
    seq.append(cur)
    i, j = cur[0], cur[1]
    t = cells[i][j].action
    s += t
    if t == 'U': cur = (i+1, j)
    if t == 'D': cur = (i-1, j)
    if t == 'L': cur = (i, j+1)
    if t == 'R': cur = (i, j-1)
action = s[::-1]
seq = seq[::-1]
# 3 
print(action)
mark(cells_to_plot, action)
plot_solution(cells_to_plot)
for row in cells:
    print(','.join([str(x.searched) for x in row]))
print()

for row in cells:
    for c in row:
        c.searched = 0

# dfs
stack = []
visited = set()
stack.append((0, width // 2))
while (height - 1, width // 2) not in stack:
    i, j = stack.pop()
    cell = cells[i][j]
    visited.add((i,j))
    cell.searched = 1
    if 'U' in cell.succ and (i - 1, j) not in visited:
        stack.append((i - 1, j))
    if 'D' in cell.succ and (i + 1, j) not in visited:
        stack.append((i + 1, j))
    if 'R' in cell.succ and (i, j + 1) not in visited:
        stack.append((i, j + 1))
    if 'L' in cell.succ and (i, j - 1) not in visited:
        stack.append((i, j - 1))
cells[height - 1][width // 2].searched = 1

for row in cells:
    print(','.join([str(x.searched) for x in row]))

for row in cells:
    for c in row:
        c.searched = 0

## Part2
man = {(i,j): abs(i-(height - 1)) + abs(j-(width // 2)) for j in range(width) for i in range(height)}
euc = {(i,j): math.sqrt((i-(height - 1))**2 + (j-(width // 2))**2 ) for j in range(width) for i in range(height)}

l = list(man.values())
l = np.array([l[i:i + height] for i in range(0, len(l), height)]).T
for row in l:
    print(','.join([str(x) for x in row]))

# manhattan   use man
g = {(i,j): float('inf') for j in range(width) for i in range(height)}
g[(0,width // 2)] = 0

queue = [(0,width // 2)]
visited = set()

while queue and (height - 1,width // 2) not in visited:
    queue.sort(key=lambda x: g[x] + man[x])
    point = queue.pop(0)
    if point not in visited:
        visited.add(point)
        i, j = point[0], point[1]
        cells[i][j].searched = 1
        succ = cells[i][j].succ
        if 'U' in succ and (i-1,j) not in visited:
            if (i-1,j) not in queue: queue += [(i-1,j)]
            g[(i-1,j)] = min(g[(i-1,j)], g[(i,j)]+1)
        if 'D' in succ and (i+1,j) not in visited:
            if (i+1,j) not in queue: queue += [(i+1,j)]
            g[(i+1,j)] = min(g[(i+1,j)], g[(i,j)]+1)
        if 'L' in succ and (i,j-1) not in visited:
            if (i,j-1) not in queue: queue += [(i,j-1)]
            g[(i,j-1)] = min(g[(i,j-1)], g[(i,j)]+1)
        if 'R' in succ and (i,j+1) not in visited:
            if (i,j+1) not in queue: queue += [(i,j+1)]
            g[(i,j+1)] = min(g[(i,j+1)], g[(i,j)]+1)     

for row in cells:
    print(','.join([str(x.searched) for x in row]))
print()

for row in cells:
    for c in row:
        c.searched = 0

g = {(i,j): float('inf') for j in range(width) for i in range(height)}
g[(0,width // 2)] = 0

queue = [(0,width // 2)]
visited = set()

while queue and (height - 1,width // 2) not in visited:
    queue.sort(key=lambda x: g[x] + euc[x])
    point = queue.pop(0)
    if point not in visited:
        visited.add(point)
        i, j = point[0], point[1]
        cells[i][j].searched = 1
        succ = cells[i][j].succ
        if 'U' in succ and (i-1,j) not in visited:
            if (i-1,j) not in queue: queue += [(i-1,j)]
            g[(i-1,j)] = min(g[(i-1,j)], g[(i,j)]+1)
        if 'D' in succ and (i+1,j) not in visited:
            if (i+1,j) not in queue: queue += [(i+1,j)]
            g[(i+1,j)] = min(g[(i+1,j)], g[(i,j)]+1)
        if 'L' in succ and (i,j-1) not in visited:
            if (i,j-1) not in queue: queue += [(i,j-1)]
            g[(i,j-1)] = min(g[(i,j-1)], g[(i,j)]+1)
        if 'R' in succ and (i,j+1) not in visited:
            if (i,j+1) not in queue: queue += [(i,j+1)]
            g[(i,j+1)] = min(g[(i,j+1)], g[(i,j)]+1)   

for row in cells:
    print(','.join([str(x.searched) for x in row]))
