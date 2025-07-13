# 2048 AI with Neural Network and MCTS

This project contains a simple 2048 implementation together with a Python based AI that combines a small neural network with Monte‑Carlo Tree Search (MCTS).

## Contents
- `ai_game.html` – playable 2048 game in the browser. Press **AI Move** to let the AI choose a move.
- `train_nn.py` – script that trains a very small neural network from the JSON move history files and saves the model as `model.json`.
- `mcts_nn_agent.py` – implements the 2048 game logic, a neural network loader and an MCTS based agent.
- `ai_server.py` – lightweight HTTP server that serves `ai_game.html` and provides a `/move` API returning the AI's chosen move.
- `2048_move_history*.json` – move history data that can be used to train the neural network.

## Usage
1. Run `python train_nn.py` to create `model.json`. Training uses the provided move history files.
2. Start the web server with `python ai_server.py`. Set the `PORT` environment
   variable if port 8000 is unavailable.
3. Open `http://localhost:8000/` in a browser (it redirects to the game page).
4. Play manually with the arrow keys or click **AI Move** to let the AI play.

The neural network is intentionally small so that it can be trained without external dependencies.
