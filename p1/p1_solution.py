import numpy as np 
# import matplotlib.pyplot as plt

file = open("mnist_train.csv", "r")
wfile = open("weights", "r")

samples = []
labels = []

for line in file:
    if (line[0] == '9'):
        samples.append(list(map(lambda x: int(x) / 255, line[2:].split(','))))
        labels.append(0)
    if (line[0] == '6'):
        samples.append(list(map(lambda x: int(x) / 255, line[2:].split(','))))
        labels.append(1)
x = np.array(samples)
y = np.array(labels)

# Todo: you may need to change some hyper-paramter like num_epochs and alpha, etc
num_epochs = 10000
m = x.shape[1]
n = x.shape[0]
alpha = 0.1

large_num = 100
epsilon = 1e-6
thresh = 1e-4

np.random.seed(0)
w = np.random.rand(m)
b = np.random.rand()

c = np.zeros(num_epochs)


for epoch in range(num_epochs):
    a = 1 / (1 + np.exp(-(np.matmul(w, np.transpose(x)) + b)))
    
    w -= alpha * np.matmul(a-y, x)
    
    b -= alpha * (a-y).sum()
    
    cost = np.zeros(len(y))
    idx = (y==0) & (a > 1 - thresh) | (y == 1) & (a < thresh)
    cost[idx] = large_num
    
    a[a<thresh] = thresh
    a[a> 1-thresh] = thresh
    
    inv_idx = np.invert(idx)
    cost[inv_idx] = - y[inv_idx] * np.log(a[inv_idx]) - (1-y[inv_idx]) * np.log(1-a[inv_idx])
    c[epoch] = cost.sum()
    
    if epoch % 3 == 0:
        print('epoch = ', epoch + 1, 'cost = ', c[epoch])
    
    if epoch > 0 and abs(c[epoch -1] - c[epoch]) < epsilon:
        print('done')
        for i in w:
            print("{:.4f}".format(i), end=",")
        print(b)
        break
    


# Todo: new test
# new_test = np.loadtxt('test.txt', delimiter=',')
# new_x = new_test / 255.0


    
