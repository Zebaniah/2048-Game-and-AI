#'2048_move_history1.json', '2048_move_history2.json','2048_move_history3.json','2048_move_history4.json','2048_move_history5.json','2048_move_history6.json','2048_move_history7.json','2048_move_history8.json'

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Flag to reset the model
reset_model = True  # Set to True if you want to reset the model
model_path = '2048_model.h5'

# Number of epochs for training
num_epochs = 50  # Set the number of epochs for training

# List of specific JSON files
file_paths = [
    '2048_move_history1.json', '2048_move_history2.json',
    '2048_move_history3.json', '2048_move_history4.json',
    '2048_move_history5.json', '2048_move_history6.json',
    '2048_move_history7.json', '2048_move_history8.json', 
    '2048_move_history9.json', '2048_move_history10.json',
    '2048_move_history11.json', '2048_move_history12.json',
    '2048_move_history13.json' 
]

# Combine data from multiple files
combined_data = []

for file_path in file_paths:
    with open(file_path) as f:
        data = json.load(f)
        print(f'Loaded {len(data)} records from {file_path}')  # Debug statement
        combined_data.extend(data)

print(f'Total combined data records: {len(combined_data)}')  # Debug statement

# Prepare the data
boards = []
directions = []
direction_map = {'left': 0, 'right': 1, 'up': 2, 'down': 3}

for record in combined_data:
    flattened_board = [tile for row in record['board'] for tile in row]
    boards.append(flattened_board)
    directions.append(direction_map[record['direction']])

X = np.array(boards)
y = np.array(directions)

print(f'Total samples: {X.shape[0]}')  # Debug statement

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the neural network model
model = Sequential([
    Flatten(input_shape=(16,)),  # 4x4 board flattened to 16 features
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # 4 possible directions
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Reset model if the reset flag is set
if reset_model and os.path.exists(model_path):
    os.remove(model_path)

# Load existing model or create a new one
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)

# Train the model
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))

# Save the model
model.save(model_path)

# Final evaluation on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Generate classification report
y_test_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_test_pred, target_names=['left', 'right', 'up', 'down']))

# Additional evaluation metrics to assess how often high tiles are reached
def evaluate_high_tiles(model, X_test, y_test):
    predictions = np.argmax(model.predict(X_test), axis=1)
    high_tile_counts = {'128': 0, '256': 0, '512': 0, '1024': 0, '2048': 0}
    
    for i in range(len(X_test)):
        board = X_test[i].reshape(4, 4)
        highest_tile = np.max(board)
        
        if highest_tile >= 128:
            high_tile_counts['128'] += 1
        if highest_tile >= 256:
            high_tile_counts['256'] += 1
        if highest_tile >= 512:
            high_tile_counts['512'] += 1
        if highest_tile >= 1024:
            high_tile_counts['1024'] += 1
        if highest_tile >= 2048:
            high_tile_counts['2048'] += 1
    
    print('High Tile Evaluation:')
    for tile, count in high_tile_counts.items():
        print(f'Tile {tile}: {count}/{len(X_test)} ({(count / len(X_test)) * 100:.2f}%)')

evaluate_high_tiles(model, X_test, y_test)
