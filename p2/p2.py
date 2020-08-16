import numpy as np

file = open("breast-cancer-wisconsin.data", "r")

samples = []
labels = []

for line in file:
    if (line.find('?') == -1):
        s = list(map(int, line.split(',')))
        samples.append(s[1:-1])
        labels.append(s[-1] != 2)

samples = np.array(samples)
labels = np.array(labels)

def entropy(data):
    count = len(data)
    p0 = sum(b[-1] == 2 for b in data) / count
    if p0 == 0 or p0 == 1: return 0
    p1 = 1 - p0
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


def infogain(data, fea, threshold):  # x_fea <= threshold;  fea = 2,3,4,..., 10; threshold = 1,..., 9
    count = len(data)
    d1 = data[data[:, fea - 1] <= threshold]
    d2 = data[data[:, fea - 1] > threshold]
    if len(d1) == 0 or len(d2) == 0: return 0
    return entropy(data) - (len(d1) / count * entropy(d1) + len(d2) / count * entropy(d2))


num = len(labels)
T = labels == True
F = labels == False
Tnum = len(labels[T])
Fnum = len(labels[F])

H_Y = -(Tnum / num * np.log2(Tnum / num) + Fnum / num * np.log2(Fnum / num))
max_gain = 0
max_f = -1
H_YX = 0
Ts = samples[T, -1]
Fs = samples[F, -1]
for j in range(1, 11):
    Tsnum = len(Ts[Ts == j])
    Fsnum = len(Fs[Fs == j])
    x = samples[:, -1]
    xnum = len(x[x == j])
    H_YX += (Tsnum / num * np.log2(Tsnum / xnum) + Fsnum / num * np.log2(Fsnum / xnum))
gain = H_Y - H_YX