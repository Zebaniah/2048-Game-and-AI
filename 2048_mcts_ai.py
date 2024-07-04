from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import random
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class MCTSNode:
    def __init__(self, state, move=None, parent=None):
        self.state = state
        self.move = move
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
            child_value = child.value / (child.visits + 1)
            if child_value > best_value:
                best_value = child_value
                best_child = child
        return best_child

class MCTS:
    def __init__(self, game, simulations=10000, max_depth=5):
        self.game = game
        self.simulations = simulations
        self.max_depth = max_depth

    def search(self, initial_state):
        root = MCTSNode(initial_state)
        for _ in range(self.simulations):
            node = self.tree_policy(root, 0)
            reward = self.default_policy(node.state)
            self.backpropagate(node, reward)
        best_move = self.best_move(root)
        return self.game.make_move(initial_state, best_move)
    
    def tree_policy(self, node, depth):
        while not self.game.is_terminal(node.state):
            if not node.is_fully_expanded() and depth < self.max_depth:
                return self.expand(node, depth)
            else:
                node = node.best_child()
        return node
    
    def expand(self, node, depth):
        state = node.state
        for move in ['left', 'right', 'up', 'down']:
            new_state = self.game.make_move(state, move)
            if not any(np.array_equal(child.state, new_state) for child in node.children):
                child_node = MCTSNode(new_state, move, parent=node)
                node.children.append(child_node)
                return child_node
        return node

    def default_policy(self, state):
        while not self.game.is_terminal(state):
            move = random.choice(['left', 'right', 'up', 'down'])
            state = self.game.make_move(state, move)
        return self.game.get_reward(state)

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def best_move(self, root):
        best_value = float('-inf')
        best_move = None
        for child in root.children:
            if child.value > best_value:
                best_value = child.value
                best_move = child.move
        return best_move

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
        score = sum(sum(row) for row in state)
        
        empty_tiles = sum(row.count(0) for row in state)
        smoothness = self.calculate_smoothness(state)
        max_tile = max(max(row) for row in state)
        monotonicity = self.calculate_monotonicity(state)
        
        score += empty_tiles * 100
        score += smoothness * 1
        score += max_tile * 10
        score += monotonicity * 100

        # Corner strategy: reward if max tile is in a corner
        if (state[0][0] == max_tile or state[0][self.board_size - 1] == max_tile or
                state[self.board_size - 1][0] == max_tile or state[self.board_size - 1][self.board_size - 1] == max_tile):
            score += max_tile * 100
        
        return score
    
    def calculate_smoothness(self, state):
        smoothness = 0
        for row in state:
            for i in range(len(row) - 1):
                smoothness -= abs(row[i] - row[i + 1])
        for col in zip(*state):
            for i in range(len(col) - 1):
                smoothness -= abs(col[i] - col[i + 1])
        return smoothness

    def calculate_monotonicity(self, state):
        monotonicity = 0
        for row in state:
            for i in range(len(row) - 1):
                if row[i] > row[i + 1]:
                    monotonicity += row[i]
                else:
                    monotonicity -= row[i]
        for col in zip(*state):
            for i in range(len(col) - 1):
                if col[i] > col[i + 1]:
                    monotonicity += col[i]
                else:
                    monotonicity -= col[i]
        return monotonicity

game = Game2048()
mcts = MCTS(game, simulations=10000)

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
