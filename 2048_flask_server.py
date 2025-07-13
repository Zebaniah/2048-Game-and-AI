from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os
import subprocess

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

MODEL_PATH = "2048_model.h5"

# Ensure a trained model is available. If not, run the training script.
if not os.path.exists(MODEL_PATH):
    try:
        subprocess.run(["python", "2048_training_script.py"], check=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to train the model automatically: {exc}") from exc

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

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
