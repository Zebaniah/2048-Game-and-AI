"""Flask server running a 2048 AI that combines a neural network
and Monte Carlo Tree Search (MCTS) to choose moves."""

import os
import random
import copy
import subprocess

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

MODEL_PATH = "2048_model.h5"

# Train the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    subprocess.run(["python", "train_model.py"], check=True)

# Load trained neural network
nn_model = tf.keras.models.load_model(MODEL_PATH)


class Game2048:
    """Minimal 2048 game logic used by the AI."""

    def __init__(self, size=4):
        self.size = size

    def make_move(self, board, move):
        def compress(b):
            new_b = []
            for row in b:
                new_row = [n for n in row if n]
                new_row += [0] * (len(row) - len(new_row))
                new_b.append(new_row)
            return new_b

        def merge(b):
            for row in b:
                for i in range(len(row) - 1):
                    if row[i] == row[i + 1] and row[i] != 0:
                        row[i] *= 2
                        row[i + 1] = 0
            return b

        def reverse(b):
            return [r[::-1] for r in b]

        def transpose(b):
            return [list(r) for r in zip(*b)]

        if move == "left":
            board = compress(board)
            board = merge(board)
            board = compress(board)
        elif move == "right":
            board = reverse(board)
            board = compress(board)
            board = merge(board)
            board = compress(board)
            board = reverse(board)
        elif move == "up":
            board = transpose(board)
            board = compress(board)
            board = merge(board)
            board = compress(board)
            board = transpose(board)
        elif move == "down":
            board = transpose(board)
            board = reverse(board)
            board = compress(board)
            board = merge(board)
            board = compress(board)
            board = reverse(board)
            board = transpose(board)

        empty = [(i, j) for i in range(self.size) for j in range(self.size) if board[i][j] == 0]
        if empty:
            i, j = random.choice(empty)
            board[i][j] = 2 if random.random() < 0.9 else 4
        return board

    def is_terminal(self, board):
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 0:
                    return False
                if i < self.size - 1 and board[i][j] == board[i + 1][j]:
                    return False
                if j < self.size - 1 and board[i][j] == board[i][j + 1]:
                    return False
        return True


def heuristic(board):
    score = sum(sum(row) for row in board)
    empty = sum(row.count(0) for row in board)
    corner = board[0][0] + board[0][3] + board[3][0] + board[3][3]
    return score + 4 * corner + 10 * empty


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == 4

    def best_child(self):
        best, best_val = None, float("-inf")
        for child in self.children:
            ucb = child.value / (child.visits + 1) + np.sqrt(2 * np.log(self.visits + 1) / (child.visits + 1))
            if ucb > best_val:
                best_val = ucb
                best = child
        return best


class MCTS:
    def __init__(self, game, model, simulations=2000):
        self.game = game
        self.model = model
        self.simulations = simulations

    def search(self, root_state):
        root = MCTSNode(copy.deepcopy(root_state))
        for _ in range(self.simulations):
            node = self._tree_policy(root)
            reward = self._default_policy(node.state)
            self._backpropagate(node, reward)
        return root.best_child().state

    def _tree_policy(self, node):
        while not self.game.is_terminal(node.state):
            if not node.is_fully_expanded():
                return self._expand(node)
            node = node.best_child()
        return node

    def _expand(self, node):
        moves = ["left", "right", "up", "down"]
        for move in moves:
            new_state = self.game.make_move(copy.deepcopy(node.state), move)
            if not any(np.array_equal(c.state, new_state) for c in node.children):
                child = MCTSNode(new_state, parent=node)
                node.children.append(child)
                return child
        return node

    def _default_policy(self, state):
        sim_state = copy.deepcopy(state)
        while not self.game.is_terminal(sim_state):
            move = self._weighted_choice(sim_state)
            sim_state = self.game.make_move(sim_state, move)
        return heuristic(sim_state)

    def _weighted_choice(self, state):
        moves = ["left", "right", "up", "down"]
        board = np.array(state).flatten().reshape(1, -1)
        nn_scores = self.model.predict(board)[0]
        scores = []
        for i, move in enumerate(moves):
            new_state = self.game.make_move(copy.deepcopy(state), move)
            value = heuristic(new_state) * (nn_scores[i] + 1e-6)
            scores.append(value)
        total = sum(scores)
        probs = [s / total for s in scores]
        return random.choices(moves, probs)[0]

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


game = Game2048()
mcts = MCTS(game, nn_model, simulations=2000)

app = Flask(__name__)
CORS(app)


@app.route("/next_move", methods=["POST"])
def next_move():
    data = request.get_json()
    board = np.array(data["board"])
    next_state = mcts.search(board)
    return jsonify([[int(c) for c in row] for row in next_state])


if __name__ == "__main__":
    app.run(port=5000)

