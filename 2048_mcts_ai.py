from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import random
import copy
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS

# Set up logging
logging.basicConfig(level=logging.DEBUG)

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
        best_value = float('-inf')
        best_child = None
        for child in self.children:
            child_value = child.value / (child.visits + 1) + np.sqrt(2 * np.log(self.visits + 1) / (child.visits + 1))
            if child_value > best_value:
                best_value = child_value
                best_child = child
        return best_child

class MCTS:
    def __init__(self, game, simulations=100000):
        self.game = game
        self.simulations = simulations

    def search(self, initial_state):
        root = MCTSNode(initial_state)
        for _ in range(self.simulations):
            node = self.tree_policy(root)
            reward = self.default_policy(node.state)
            self.backpropagate(node, reward)
        return root.best_child().state

    def tree_policy(self, node):
        while not self.game.is_terminal(node.state):
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child()
        return node

    def expand(self, node):
        state = node.state
        for move in ['left', 'right', 'up', 'down']:
            new_state = self.game.make_move(state, move)
            if not any(np.array_equal(child.state, new_state) for child in node.children):
                child_node = MCTSNode(new_state, parent=node)
                node.children.append(child_node)
                return child_node
        return node

    def default_policy(self, state):
        while not self.game.is_terminal(state):
            move = self.weighted_random_choice(state)
            state = self.game.make_move(state, move)
        return self.heuristic(state)

    def weighted_random_choice(self, state):
        moves = ['left', 'right', 'up', 'down']
        scores = []
        for move in moves:
            new_state = self.game.make_move(state, move)
            scores.append(self.heuristic(new_state))
        total_score = sum(scores)
        probabilities = [score / total_score for score in scores]
        return random.choices(moves, probabilities)[0]

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def heuristic(self, state):
        score = sum(sum(row) for row in state)
        corner_bonus = state[0][0] + state[0][3] + state[3][0] + state[3][3]
        empty_tiles = sum(row.count(0) for row in state)
        return score + 4 * corner_bonus + 10 * empty_tiles

class Game2048:
    def __init__(self, board_size=4):
        self.board_size = board_size

    def make_move(self, state, move):
        def compress(board):
            new_board = []
            for row in board:
                new_row = [num for num in row if num != 0]
                new_row += [0] * (len(row) - len(new_row))
                new_board.append(new_row)
            return new_board

        def merge(board):
            for row in board:
                for i in range(len(row) - 1):
                    if row[i] == row[i + 1] and row[i] != 0:
                        row[i] *= 2
                        row[i + 1] = 0
            return board

        def reverse(board):
            new_board = []
            for row in board:
                new_board.append(row[::-1])
            return new_board

        def transpose(board):
            new_board = []
            for i in range(len(board)):
                new_row = []
                for j in range(len(board[0])):
                    new_row.append(board[j][i])
                new_board.append(new_row)
            return new_board

        if move == 'left':
            state = compress(state)
            state = merge(state)
            state = compress(state)
        elif move == 'right':
            state = reverse(state)
            state = compress(state)
            state = merge(state)
            state = compress(state)
            state = reverse(state)
        elif move == 'up':
            state = transpose(state)
            state = compress(state)
            state = merge(state)
            state = compress(state)
            state = transpose(state)
        elif move == 'down':
            state = transpose(state)
            state = reverse(state)
            state = compress(state)
            state = merge(state)
            state = compress(state)
            state = reverse(state)
            state = transpose(state)
        
        # Add a new tile to the board
        empty_tiles = [(i, j) for i in range(self.board_size) for j in range(self.board_size) if state[i][j] == 0]
        if empty_tiles:
            i, j = random.choice(empty_tiles)
            state[i][j] = 2 if random.random() < 0.9 else 4
        
        return state

    def is_terminal(self, state):
        for i in range(self.board_size):
            for j in range(self.board_size):
                if state[i][j] == 0:
                    return False
                if i < self.board_size - 1 and state[i][j] == state[i + 1][j]:
                    return False
                if j < self.board_size - 1 and state[i][j] == state[i][j + 1]:
                    return False
        return True

    def get_reward(self, state):
        return sum(sum(row) for row in state)

game = Game2048()
mcts = MCTS(game, simulations=100000)

@app.route('/next_move', methods=['POST'])
def next_move():
    data = request.json
    logging.debug('Received board state from client: %s', data['board'])
    board = np.array(data['board'])
    next_state = mcts.search(board)
    next_state_list = [[int(cell) for cell in row] for row in next_state]
    logging.debug('Sending next state to client: %s', next_state_list)
    return jsonify(next_state_list)

if __name__ == '__main__':
    app.run(port=5000)
