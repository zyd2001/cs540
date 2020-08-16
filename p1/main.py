import random, math

file = open("mnist_train.csv", "r")

samples = []
labels = []
alpha = 0.0001

random.seed(5657)

for line in file:
    if (line[0] == '9'):
        samples.append(list(map(lambda x: int(x) / 255, line[2:].split(','))))
        labels.append(0)
    if (line[0] == '6'):
        samples.append(list(map(lambda x: int(x) / 255, line[2:].split(','))))
        labels.append(1)

print(samples.__len__())

weights = [];
bias = random.uniform(-1, 1);
predictions = [0] * len(samples);
for i in range(784):
    weights.append(random.uniform(-1, 1))

def log(i):
    if (i == 0):
        return -math.inf
    else:
        return math.log(i)

def update():
    for i in range(len(samples)):
        predictions[i] = func(samples[i])
        
def func(data):
    return 1 / (1 + math.exp(-sum(map(lambda w, x: w * x, weights, data), bias)))

def loss():
    loss = -sum(map(lambda x, y: y * log(x) + (1 - y) * log(1 - x), predictions, labels))
    if (math.isnan(loss)):
        return math.inf
    else:
        return loss

def geti(i):
    return list(map(lambda list: list[i], samples))

update()
last_loss = loss()
print(predictions)
while True:
    for i in range(len(weights)):
        weights[i] = weights[i] - alpha * sum(map(lambda x, y, a: (a - y) * x, geti(i), labels, predictions))
    bias = bias - alpha * sum(map(lambda y, a: (a - y), labels, predictions))
    update()
    if (abs(last_loss - loss()) < 0.0001):
        break
    last_loss = loss()
    print(last_loss)
print(last_loss)
# print(weights)