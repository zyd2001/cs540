import numpy as np
import re

file = open("script.txt", "r")

lines = file.readlines()

p1 = re.compile('[^a-z ]')
p2 = re.compile(' +')

lines = list(map(lambda i: p1.sub('', i.strip().lower()), lines))
   
line = ' '.join(lines)
line = p2.sub(' ', line)

bi = np.zeros((27,27))
bi_lap = np.zeros((27,27))
tri = np.zeros((27,27,27))

s = ' abcdefghijklmnopqrstuvwxyz'

for i in range(27):
    c = line.count(s[i])
    for j in range(27):
        bi[i][j] = line.count(s[i] + s[j]) / c
        bi_lap[i][j] = (line.count(s[i] + s[j]) + 1) / (c + 27)
        if (bi_lap[i][j] < 0.0001):
            bi_lap[i][j] = 0.0001

for i in range(27):
    for j in range(27):
        c = line.count(s[i] + s[j])
        for k in range(27):
            tri[i][j][k] = (line.count(s[i] + s[j] + s[k]) + 1) / (c + 27)

for i in range(27):
    print(','.join(map(lambda c: '{:.4f}'.format(c), bi[i])))
print()
for i in range(27):
    print(','.join(map(lambda c: '{:.4f}'.format(c), bi_lap[i])))
print()
print(','.join(map(lambda c: '{:.4f}'.format(line.count(c) / len(line)), s)))

def gen_bi(c1):
    r = np.random.uniform()
    accum = 0
    count = 0
    for i in bi[s.index(c1)]:
        if (r <= accum + i and r > accum):
            return s[count]
        else:
            count += 1
            accum = accum + i
    return 'z'

def gen_tri(c1, c2):
    r = np.random.uniform()
    accum = 0
    count = 0
    for i in tri[s.index(c1)][s.index(c2)]:
        if (r <= accum + i and r > accum):
            return s[count]
        else:
            count += 1
            accum = accum + i
    return 'z'

sentences = []
for i in range(26):
    sentence = ''
    c = s[i + 1]
    sentence += c + gen_bi(c)
    for j in range(998):
        if (line.count(sentence[-2:]) == 0):
            sentence += gen_bi(sentence[-1])
        else:
            sentence += gen_tri(sentence[-2], sentence[-1])
    sentences.append(sentence)
    print(sentence)


#part2
file = open("script2.txt", "r")

ps1 = np.array(list(map(lambda c: line.count(c) / len(line), s)))
line = p2.sub(' ', ' '.join(file.readlines()))
ps2 = np.array(list(map(lambda c: line.count(c) / len(line), s)))
ps1 = np.round(ps1, 4)
ps2 = np.round(ps2, 4)
post_ps1 = ps1 / (ps1 + ps2)
post_ps2 = ps2 / (ps1 + ps2)
print(','.join(map(lambda c: '{:.4f}'.format(c), ps2)))
print(','.join(map(lambda c: '{:.4f}'.format(c), post_ps2)))

for i in sentences:
    p1 = 0
    p2 = 0
    for c in i:
        p1 += np.log(ps1[s.index(c)])
        p2 += np.log(ps2[s.index(c)])
    if (p1 > p2): 
        print(0, end=',')
    else:
        print(1, end=',')
print()