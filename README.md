# 2048 Game AI

This repo contains move history data and a simple HTML client for the puzzle game 2048.  The Python scripts train a neural network on that data and use it to guide a Monte Carlo Tree Search (MCTS) so the game can play itself.

## Training the neural network
Run `python train_model.py` to train on the provided JSON files.  The model is saved as `2048_model.h5`.

## Running the combined AI
`nn_mcts_server.py` starts a Flask server.  If no model is present it automatically runs the training script.  The server exposes a `/next_move` endpoint that receives the current board and returns the next board state chosen by the AI.

Open `2048_Game-AI_Play.html` in your browser and press **Start AI** to watch it play.  The page communicates with the Flask server to fetch moves.

`2048_game_grid_fixed.html` is a minimal client used to create the move history files and can also be used to play manually.
