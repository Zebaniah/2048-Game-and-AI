import json
import random
import math
import glob
import os

MOVE_MAP = {"up": 0, "down": 1, "left": 2, "right": 3}


def encode_board(board):
    vec = []
    for row in board:
        for val in row:
            if val == 0:
                vec.append(0.0)
            else:
                vec.append((val.bit_length() - 1) / 11.0)
    return vec


def encode_label(direction):
    y = [0.0] * 4
    y[MOVE_MAP[direction]] = 1.0
    return y


class NeuralNetwork:
    def __init__(self, input_size=16, hidden_size=32, output_size=4):
        self.W1 = [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.W2 = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(output_size)]
        self.b2 = [0.0 for _ in range(output_size)]

    def _relu(self, vec):
        return [max(0.0, v) for v in vec]

    def _softmax(self, vec):
        exps = [math.exp(v) for v in vec]
        s = sum(exps)
        return [v / s for v in exps]

    def predict(self, x):
        z1 = [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(self.W1, self.b1)]
        a1 = self._relu(z1)
        z2 = [sum(w * ai for w, ai in zip(row, a1)) + b for row, b in zip(self.W2, self.b2)]
        return self._softmax(z2)

    def train(self, data, epochs=1, lr=0.01):
        for _ in range(epochs):
            random.shuffle(data)
            for board, direction in data:
                x = encode_board(board)
                y = encode_label(direction)

                # forward
                z1 = [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(self.W1, self.b1)]
                a1 = self._relu(z1)
                z2 = [sum(w * ai for w, ai in zip(row, a1)) + b for row, b in zip(self.W2, self.b2)]
                out = self._softmax(z2)

                # gradients output layer
                delta2 = [out_i - y_i for out_i, y_i in zip(out, y)]
                for i in range(len(self.W2)):
                    for j in range(len(self.W2[i])):
                        self.W2[i][j] -= lr * delta2[i] * a1[j]
                for i in range(len(self.b2)):
                    self.b2[i] -= lr * delta2[i]

                # backpropagate to hidden layer
                delta1 = []
                for j in range(len(self.W1)):
                    if a1[j] > 0:
                        val = sum(self.W2[i][j] * delta2[i] for i in range(len(delta2)))
                    else:
                        val = 0.0
                    delta1.append(val)
                for j in range(len(self.W1)):
                    for k in range(len(self.W1[j])):
                        self.W1[j][k] -= lr * delta1[j] * x[k]
                for j in range(len(self.b1)):
                    self.b1[j] -= lr * delta1[j]

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}, f)


def load_datasets():
    data = []
    for fname in glob.glob("2048_move_history*.json"):
        with open(fname) as f:
            moves = json.load(f)
        for step in moves:
            data.append((step["board"], step["direction"]))
    return data


def main():
    data = load_datasets()
    model = NeuralNetwork()
    model.train(data, epochs=3, lr=0.01)
    model.save("model.json")


if __name__ == "__main__":
    main()
