import numpy as np

# WEIGHTS_NORMALISE = False
WEIGHTS_NORMALISE = True
LOG_SIGMOID = True
STEP_LR = False
HIDDEN_OUTPUT_FULL_CONNECTIONS = False
BIAS = False

# np.random.seed(10)


class neuralNetwork:
    def __init__(self, inputnodes=4, hiddennodes=12, outputnodes=3, lr=0.01):
        self.inodes = inputnodes
        if (HIDDEN_OUTPUT_FULL_CONNECTIONS):
            self.hnodes = hiddennodes
        else:
            self.hnodes = outputnodes
        self.onodes = outputnodes
        self.lr = lr

        bias = 0
        if BIAS:
            bias = 1

        if WEIGHTS_NORMALISE:
            self.wih = np.random.normal(
                0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes + bias))
            self.who = np.random.normal(
                0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        else:
            self.wih = np.random.rand(self.hnodes, self.inodes + bias)
            self.who = np.random.rand(self.onodes, self.hnodes)

        if HIDDEN_OUTPUT_FULL_CONNECTIONS == False:
            self.who = np.eye(self.onodes, self.hnodes)

        if LOG_SIGMOID:
            self.activation_function = lambda x: 1. / (1. + np.exp(-x))
        else:
            self.activation_function = lambda x: x

    def _softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=0))  # 防止溢位
        return exp_values / np.sum(exp_values, axis=0)

    def _cross_entropy(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 10**(-100)))

    def _sig_dash(self, x):
        return self.activation_function(x).T @ (1 - self.activation_function(x))

    def train(self, inputs, targets, epochs=100):
        CROSS_LOSS = []
        AC = []
        for epoch in range(epochs):
            if (epoch % 10 == 9 and STEP_LR):
                self.lr *= 0.95
            ac = 0
            error_squared = []
            for i in range(targets.shape[0]):
                # 正向傳播
                if (BIAS):
                    INPUT = np.append(np.array([[1]]), np.array(
                        inputs[i], ndmin=2).T, axis=0)
                else:
                    INPUT = np.array(inputs[i], ndmin=2).T

                hidden_inputs = self.wih @ INPUT
                hidden_outputs = self.activation_function(hidden_inputs)

                final_inputs = self.who @ hidden_outputs
                final_outputs = self._softmax(final_inputs)

                # 反向傳播
                output_errors = (targets[i] - final_outputs.T).T
                hidden_errors = self.who.T @ output_errors @ self._sig_dash(
                    hidden_outputs)

                if (np.argmax(final_outputs.flatten()) == np.argmax(np.array(targets[i], ndmin=2).T.flatten())):
                    ac += 1

                if (HIDDEN_OUTPUT_FULL_CONNECTIONS):
                    self.who += self.lr * (output_errors @ hidden_outputs.T)
                self.wih += self.lr * \
                    (hidden_errors @ hidden_outputs.T @
                     (1-hidden_outputs) @ INPUT.T)

                error_squared.append(self._cross_entropy(
                    np.array(targets[i], ndmin=2).T, final_outputs))
            AC.append(ac / targets.shape[0])

            # print("epoch {}:\tCross Entropy Loss = {}".format(
            #     epoch, np.mean(error_squared)))
            CROSS_LOSS.append(np.mean(error_squared))
        print("Total number of epochs: {}".format(epochs))
        print("Final cross entropy loss: {}".format(CROSS_LOSS[-1]))
        # 畫出顏色紅色、圓形錨點、虛線、粗細 2、資料點大小 6 的線條
        return CROSS_LOSS, AC

    def query(self, inputs, targets):
        corrects = 0
        for i in range(targets.shape[0]):
            if (BIAS):
                INPUT = np.append(np.array([[1]]), np.array(
                    inputs[i], ndmin=2).T, axis=0)
            else:
                INPUT = np.array(inputs[i], ndmin=2).T

            hidden_inputs = self.wih @ INPUT
            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = self.who @ hidden_outputs
            final_outputs = self._softmax(final_inputs)

            # print(final_outputs.flatten())
            # print("prediction: ", np.argmax(final_outputs.flatten()))
            # print("target: ", np.array(targets[i], ndmin=2).T.flatten())
            # print(np.argmax(np.array(targets[i], ndmin=2).T.flatten()))
            if (np.argmax(final_outputs.flatten()) == np.argmax(np.array(targets[i], ndmin=2).T.flatten())):
                corrects += 1
                print('AC!!!')
        correct_percent = corrects / targets.shape[0] * 100.
        print("Test Correct Percentage: {}%".format(correct_percent))

        return correct_percent
