import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

MODEL_PATH = "2048_model.h5"
EPOCHS = 30

FILE_PATHS = [
    '2048_move_history1.json',
    '2048_move_history2.json',
    '2048_move_history3.json',
    '2048_move_history4.json',
    '2048_move_history5.json',
    '2048_move_history6.json',
    '2048_move_history7.json',
    '2048_move_history8.json',
    '2048_move_history9.json',
    '2048_move_history10.json',
    '2048_move_history11.json',
    '2048_move_history12.json',
    '2048_move_history13.json',
]

def load_data(paths):
    boards, moves = [], []
    mapping = {'left': 0, 'right': 1, 'up': 2, 'down': 3}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
            for rec in data:
                boards.append([t for row in rec['board'] for t in row])
                moves.append(mapping[rec['direction']])
    return np.array(boards), np.array(moves)


def build_model():
    model = Sequential([
        Flatten(input_shape=(16,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(4, activation='softmax'),
    ])
    model.compile(optimizer=Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    X, y = load_data(FILE_PATHS)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = build_model()
    model.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.1)
    model.save(MODEL_PATH)
    test_loss, acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {acc:.3f}')


if __name__ == '__main__':
    main()
