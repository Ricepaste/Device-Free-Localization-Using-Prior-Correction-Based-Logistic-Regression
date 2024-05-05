import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WEIGHTS_NORMALISE = True
LOG_SIGMOID = True


def save_to_csv(output):
    file_path = "./"
    filename = file_path + f"Final_Output.csv"
    print(filename)
    np.savetxt(filename, output, encoding='utf-8',
               delimiter="\t", fmt='%s')


class neuralNetwork:
    def __init__(self, inputnodes=4, hiddennodes=12, outputnodes=3, lr=0.45):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = lr

        if WEIGHTS_NORMALISE:
            self.wih = np.random.normal(
                0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes+1))
            self.who = np.random.normal(
                0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        else:
            self.wih = np.random.rand(self.hnodes, self.inodes+1)
            self.who = np.random.rand(self.onodes, self.hnodes)

        if LOG_SIGMOID:
            self.activation_function = lambda x: 1 / (1 + np.exp(-x))
        else:
            self.activation_function = lambda x: x

    def train(self, inputs, targets, epochs=100):
        RMSE = []
        for epoch in range(epochs):
            error_squared = []
            for i in range(targets.shape[0]):
                # calculate signals into hidden layer
                INPUT = np.append(np.array([[1]]), np.array(
                    inputs[i], ndmin=2).T, axis=0)
                # print(INPUT)
                hidden_inputs = np.dot(
                    self.wih, INPUT)
                # calculate the signals emerging from hidden layer
                hidden_outputs = self.activation_function(hidden_inputs)

                # calculate signals into final output layer
                final_inputs = np.dot(self.who, hidden_outputs)
                # calculate the signals emerging from final output layer
                final_outputs = self._softmax(final_inputs)

                # error is the (target - actual)
                output_errors = (targets[i] - final_outputs.T).T

                # hidden layer error is the output_errors, split by weights
                # recombined at hidden nodes
                hidden_errors = np.dot(self.who.T, output_errors)

                # update the weights for the link between the hidden and output
                self.who += self.lr * output_errors * \
                    np.transpose(hidden_outputs)

                # update the weights for the link between the input and hidden
                self.wih += self.lr * \
                    np.dot(hidden_errors * hidden_outputs *
                           (1.0 - hidden_outputs),
                           INPUT.T)

                error_squared.append(abs(output_errors))
            print("epoch {}:\tRMSE = {}".format(
                epoch, np.sqrt(np.mean(error_squared))))
            RMSE.append(np.sqrt(np.mean(error_squared)))
        print("Total number of epochs: {}".format(epochs))
        print("Final RMSE: {}".format(RMSE[-1]))
        # 畫出顏色紅色、圓形錨點、虛線、粗細 2、資料點大小 6 的線條
        return RMSE

    def query(self, inputs, targets):
        corrects = 0
        ans = []
        for i in range(targets.shape[0]):
            INPUT = np.append(np.array([[1]]), np.array(
                inputs[i], ndmin=2).T, axis=0)
            hidden_inputs = np.dot(self.wih, INPUT)
            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = np.dot(self.who, hidden_outputs)
            final_outputs = self._softmax(final_inputs)

            # if (targets[i] + 0.5 >= final_outputs.T[0]).all() and (targets[i] - 0.5 < final_outputs.T[0]).all():
            #     corrects += 1
            cur_target = list(targets[i])
            cur_output = list(final_outputs.T[0])
            if cur_target.index(max(cur_target)) == cur_output.index(max(cur_output)):
                print(f'target: {cur_target}, output: {cur_output}')
                print(
                    f'target: {cur_target.index(max(cur_target)) + 1}, output: {cur_output.index(max(cur_output)) + 1}')
                corrects += 1

            FINAL_outputs = list(final_outputs.T[0])
            ans.append(FINAL_outputs.index(max(FINAL_outputs))+1)
        correct_percent = corrects / targets.shape[0] * 100.
        print("Test Correct Percentage: {}%".format(correct_percent))
        return ans

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0)


X = pd.read_csv('iris_in.csv', header=None)
Y = pd.read_csv('iris_out.csv', header=None)

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

print(X_train.shape)
print(Y_train.shape)

n = neuralNetwork()
# RMSE = n.train(X_train, Y_train)
RMSE = n.train(X_train, Y_train_onehot)
# ans = n.query(X_test, Y_test)
ans = n.query(X_test, Y_test_onehot)
plt.plot(RMSE, color='r', marker='o',
         linewidth=2, markersize=6)
plt.show()

save_to_csv(ans)
