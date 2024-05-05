from neural_network import *
import matplotlib.pyplot as plt
import pandas as pd

X = pd.read_csv('./iris/iris_in.csv', header=None)
Y = pd.read_csv('./iris/iris_out.csv', header=None)

X = np.array(X.values)
Y = np.array(Y.values)

# print(X.shape)
# print(Y.shape)

X_train = X[:75]
Y_train = Y[:75]
Y_train_onehot = np.zeros((Y_train.size, Y_train.max()))
Y_train_onehot[np.arange(Y_train.size), (Y_train - 1).flatten()] = 1

X_test = X[75:]
Y_test = Y[75:]
Y_test_onehot = np.zeros((Y_test.size, Y_test.max()))
Y_test_onehot[np.arange(Y_test.size), (Y_test - 1).flatten()] = 1

n = neuralNetwork(inputnodes=4, hiddennodes=3, outputnodes=3, lr=0.007)

RMSE, AC = n.train(X_train, Y_train_onehot, epochs=100)

n.query(X_test, Y_test_onehot)
plt.plot(RMSE, color='r', marker='o',
         linewidth=2, markersize=6)
plt.show()
plt.plot(AC, color='r', marker='o',
         linewidth=2, markersize=6)
plt.show()
