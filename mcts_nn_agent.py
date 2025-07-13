import json
import random
import math

MOVE_MAP = {0: "up", 1: "down", 2: "left", 3: "right"}
DIRS = list(MOVE_MAP.values())


def encode_board(board):
    vec = []
    for row in board:
        for val in row:
            if val == 0:
                vec.append(0.0)
            else:
                vec.append((val.bit_length() - 1) / 11.0)
    return vec


class NeuralNetwork:
    def __init__(self, model_path):
        with open(model_path) as f:
            params = json.load(f)
        self.W1 = params["W1"]
        self.b1 = params["b1"]
        self.W2 = params["W2"]
        self.b2 = params["b2"]

    def _relu(self, vec):
        return [max(0.0, v) for v in vec]

    def _softmax(self, vec):
        exps = [math.exp(v) for v in vec]
        s = sum(exps)
        return [v / s for v in exps]

    def policy(self, board):
        x = encode_board(board)
        z1 = [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(self.W1, self.b1)]
        a1 = self._relu(z1)
        z2 = [sum(w * ai for w, ai in zip(row, a1)) + b for row, b in zip(self.W2, self.b2)]
        return self._softmax(z2)


# 2048 Game mechanics
BOARD_SIZE = 4


def compress(row):
    new_row = [num for num in row if num != 0]
    new_row += [0] * (BOARD_SIZE - len(new_row))
    return new_row


def merge(row):
    for j in range(BOARD_SIZE - 1):
        if row[j] != 0 and row[j] == row[j + 1]:
            row[j] *= 2
            row[j + 1] = 0
    return row


def move_left(board):
    new_board = []
    moved = False
    for row in board:
        new = merge(compress(row))
        if new != row:
            moved = True
        new_board.append(new)
    return new_board, moved


def move_right(board):
    new_board, moved = move_left([list(reversed(r)) for r in board])
    new_board = [list(reversed(r)) for r in new_board]
    return new_board, moved


def transpose(board):
    return [list(r) for r in zip(*board)]


def move_up(board):
    t, moved = move_left(transpose(board))
    return transpose(t), moved


def move_down(board):
    t, moved = move_right(transpose(board))
    return transpose(t), moved


MOVE_FUNCS = {
    "left": move_left,
    "right": move_right,
    "up": move_up,
    "down": move_down,
}


def add_new_tile(board):
    empty = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i][j] == 0]
    if not empty:
        return False
    i, j = random.choice(empty)
    board[i][j] = 2 if random.random() < 0.9 else 4
    return True


def is_game_over(board):
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 0:
                return False
            if j < BOARD_SIZE - 1 and board[i][j] == board[i][j + 1]:
                return False
            if i < BOARD_SIZE - 1 and board[i][j] == board[i + 1][j]:
                return False
    return True


class Node:
    def __init__(self, board, parent=None, prior=0.0):
        self.board = [row[:] for row in board]
        self.parent = parent
        self.children = {}
        self.n = 0
        self.w = 0.0
        self.p = prior

    def value(self):
        return self.w / self.n if self.n > 0 else 0.0

    def ucb(self, c_puct=1.4):
        if self.parent is None:
            return self.value()
        return self.value() + c_puct * self.p * math.sqrt(self.parent.n) / (1 + self.n)


class MCTSAgent:
    def __init__(self, model_path, simulations=100):
        self.nn = NeuralNetwork(model_path)
        self.simulations = simulations

    def best_move(self, board):
        root = Node(board)
        policy = self.nn.policy(board)
        for d, p in zip(DIRS, policy):
            new_board, moved = MOVE_FUNCS[d]([row[:] for row in board])
            if moved:
                child = Node(new_board, root, p)
                root.children[d] = child

        for _ in range(self.simulations):
            node = root
            path = [node]
            # select
            while node.children:
                node = max(node.children.values(), key=lambda n: n.ucb())
                path.append(node)
            # expand
            if node.n > 0 and not is_game_over(node.board):
                policy = self.nn.policy(node.board)
                for d, p in zip(DIRS, policy):
                    new_board, moved = MOVE_FUNCS[d]([row[:] for row in node.board])
                    if moved:
                        node.children[d] = Node(new_board, node, p)
            # evaluate
            value = max(self.nn.policy(node.board))
            # backup
            for n in path:
                n.n += 1
                n.w += value

        if not root.children:
            return random.choice(DIRS)
        best = max(root.children.items(), key=lambda kv: kv[1].n)[0]
        return best
