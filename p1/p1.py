import numpy as np

file = open("mnist_train.csv", "r")

samples = []
labels = []
alpha = 0.1
thresh = 1e-4

for line in file:
    if (line[0] == '9'):
        samples.append(list(map(lambda x: int(x) / 255, line[2:].split(','))))
        labels.append(0)
    if (line[0] == '6'):
        samples.append(list(map(lambda x: int(x) / 255, line[2:].split(','))))
        labels.append(1)

np.random.seed(0)
weights = np.random.uniform(-1, 1, 784)
bias = np.random.uniform(-1, 1)

# weights = list(map(lambda x: float(x), wfile.readline().split(",")))
# bias = float(wfile.readline())
# print(weights)

weights = np.array(weights)
predictions = np.array(predictions, dtype=np.float64)
labels = np.array(labels)
samples = np.array(samples)

print(",".join("{:.2f}".format(i) for i in samples[0]))

def loss():
    cost = np.zeros(len(labels))
    idx = (labels==0) & (predictions > 1 - thresh) | (labels == 1) & (predictions < thresh)
    cost[idx] = 1000
    
    predictions[predictions<thresh] = thresh
    predictions[predictions> 1-thresh] = thresh
    
    inv_idx = np.invert(idx)
    cost[inv_idx] = - labels[inv_idx] * np.log(predictions[inv_idx]) - (1-labels[inv_idx]) * np.log(1-predictions[inv_idx])
    return cost.sum()

last_loss = 0
while True:
    predictions = 1 / (1 + np.exp(-(np.matmul(weights, np.transpose(samples)) + bias)))
    weights -= alpha * np.matmul(predictions - labels, samples)
    bias = bias - alpha * (predictions - labels).sum()
    if (abs(last_loss - loss()) < 0.0001):
        break
    last_loss = loss()
    # print(last_loss)

for i in weights:
    print("{:.4f}".format(i), end=",")
print(bias)

test = open("test.txt", "r")
tests = []
for line in test:
    tests.append(list(map(lambda x: int(x) / 255, line.split(','))))
tests = np.array(tests)

results = 1 / (1 + np.exp(-(np.matmul(weights, np.transpose(tests)) + bias)))
print(",".join("{:.2f}".format(i) for i in results))
print(",".join(map(lambda x: str(int(x)), results)))