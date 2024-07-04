from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = tf.keras.models.load_model('2048_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    board = np.array(data['board']).flatten().reshape(1, -1)
    prediction = model.predict(board)
    direction_index = np.argmax(prediction, axis=1)[0]
    direction_map = {0: 'left', 1: 'right', 2: 'up', 3: 'down'}
    return jsonify({'direction': direction_map[direction_index]})

if __name__ == '__main__':
    app.run(port=5000)
