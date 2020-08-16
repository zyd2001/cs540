import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def diff_sigmoid(x):
    return x * (1 - x)

def diff_relu(x):
    y = x
    y[x>0] = 1
    y[x<=0] = 0
    return y

file = open("mnist_train.csv", "r")

samples = []
labels = []
alpha = 0.2

for line in file:
    if (line[0] == '9'):
        samples.append(list(map(lambda x: int(x) / 255, line[2:].split(','))))
        labels.append(0)
    if (line[0] == '6'):
        samples.append(list(map(lambda x: int(x) / 255, line[2:].split(','))))
        labels.append(1)

np.random.seed(0)
weightL1 = np.random.uniform(-1, 1, (392, 785))
weightLO = np.random.uniform(-1, 1, (393))
[i.append(1) for i in samples]
y = np.array(labels)
x = np.array(samples)
aL1 = np.zeros(393)
aL1[-1] = 1
pred = np.zeros(len(samples))

c = 1
while True:
    index = list(range(len(x)))
    np.random.shuffle(index)
    for i in index:
        aL1[:-1] = sigmoid(weightL1 @ x[i])
        pred[i] = sigmoid(aL1 @ weightLO)

        deltaO = diff_sigmoid(pred[i]) * (pred[i] - y[i])
        delta = deltaO * weightLO * diff_sigmoid(aL1)
        weightL1 -= alpha * (delta[:-1, np.newaxis] @ x[i][np.newaxis, :])
        weightLO -= alpha * deltaO * aL1
    cost = 0.5 * ((y - pred) ** 2).sum()
    num_correct = sum((pred > 0.5).astype(int) == y)
    # print('cost = ', cost, ' accuracy = ', num_correct / len(y))
    if (num_correct / len(y) > 0.999):
        print('Iteration= ', c)
        break
    c += 1

print('first layer weights')
for w in weightL1.T:
    print(",".join("{:.4f}".format(i) for i in w))
print('second layer weights')
print(",".join("{:.4f}".format(i) for i in weightLO))

test = open("test.txt", "r")
tests = []
for line in test:
    tests.append(list(map(lambda x: int(x) / 255, line.split(','))))
[i.append(1) for i in tests]
tests = np.array(tests)
pred = np.zeros(len(tests))
for i in range(len(tests)):
    aL1[:-1] = sigmoid(weightL1 @ tests[i])
    pred[i] = sigmoid(aL1 @ weightLO)
print('activition')
print(",".join("{:.2f}".format(i) for i in pred))
print('prediction')
print(",".join(map(lambda x: str(int(x > 0.5)), pred)))