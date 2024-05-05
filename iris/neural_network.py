import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WEIGHTS_NORMALISE = True
LOG_SIGMOID = True


class neuralNetwork:
    def __init__(self, inputnodes=65, hiddennodes=110, outputnodes=40, lr=0.01):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = lr

        if WEIGHTS_NORMALISE:
            self.wih = np.random.normal(
                0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
            self.who = np.random.normal(
                0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        else:
            self.wih = np.random.rand(self.hnodes, self.inodes)
            self.who = np.random.rand(self.onodes, self.hnodes)

        if LOG_SIGMOID:
            self.activation_function = lambda x: 1. / (1. + np.exp(-x))
        else:
            self.activation_function = lambda x: x

    def _softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=0))  # 防止指数爆炸
        return exp_values / np.sum(exp_values, axis=0)

    def _cross_grad(self, y_true, y_pred):
        # print("yTrue")
        # print(y_true)
        # print("yPred")
        # print(y_pred)
        # print("cross grad")
        # print(-y_true.astype(np.float64) /
        #       (y_pred.astype(np.float64) + 10**(-100)))
        return -y_true.astype(np.float64)/(y_pred.astype(np.float64) + 10**(-100))

    def _cross_entropy(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 10**(-100)))

    def _softmax_dash(self, x):
        I = np.eye(x.shape[0])
        return self._softmax(x).T * (np.repeat(1, x.shape[0]).T - self._softmax(x)).T

    def _sig_dash(self, x):
        return self.activation_function(x).T @ (1 - self.activation_function(x))

    def train(self, inputs, targets, epochs=100):
        CROSS_LOSS = []
        AC = []
        old_epoch = -1
        for epoch in range(epochs):
            if (epoch % 10 == 9):
                self.lr *= 0.95
            ac = 0
            error_squared = []
            for i in range(targets.shape[0]):
                # 正向傳播
                hidden_inputs = np.dot(
                    self.wih, np.array(inputs[i], ndmin=2).T)
                hidden_outputs = self.activation_function(hidden_inputs)

                final_inputs = np.dot(self.who, hidden_outputs)
                final_outputs = self._softmax(final_inputs)

                # 反向傳播
                output_errors = final_outputs - np.array(targets[i], ndmin=2).T
                hidden_errors = self.who.T @ output_errors @ self._sig_dash(
                    hidden_outputs)

                if (np.argmax(final_outputs.flatten()) == np.argmax(np.array(targets[i], ndmin=2).T.flatten())):
                    ac += 1

                self.who -= self.lr * (np.dot(output_errors, hidden_outputs.T))
                self.wih -= self.lr * \
                    (hidden_errors @ np.array(inputs[i], ndmin=2))

                error_squared.append(self._cross_entropy(
                    np.array(targets[i], ndmin=2).T, final_outputs))
            AC.append(ac / targets.shape[0])

            print("epoch {}:\tCross Entropy Loss = {}".format(
                epoch, np.mean(error_squared)))
            CROSS_LOSS.append(np.mean(error_squared))
        print("Total number of epochs: {}".format(epochs))
        print("Final cross entropy loss: {}".format(CROSS_LOSS[-1]))
        # 畫出顏色紅色、圓形錨點、虛線、粗細 2、資料點大小 6 的線條
        return CROSS_LOSS, AC

    def query(self, inputs, targets):
        corrects = 0
        for i in range(targets.shape[0]):
            hidden_inputs = np.dot(
                self.wih, np.array(inputs[i], ndmin=2).T)
            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = np.dot(self.who, hidden_outputs)
            final_outputs = self._softmax(final_inputs)

            print(final_outputs.flatten())
            print("prediction: ", np.argmax(final_outputs.flatten()))
            print("target: ", np.array(targets[i], ndmin=2).T.flatten())
            print(np.argmax(np.array(targets[i], ndmin=2).T.flatten()))
            if (np.argmax(final_outputs.flatten()) == np.argmax(np.array(targets[i], ndmin=2).T.flatten())):
                corrects += 1
                print('AC!!!')
        correct_percent = corrects / targets.shape[0] * 100.
        print("Test Correct Percentage: {}%".format(correct_percent))
