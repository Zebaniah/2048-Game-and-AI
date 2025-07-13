There are 2 different AI versions here, as well as a non-ai version of the game. 


AI #1: Neural Network
This AI is trained on datasets and learns how to predict the best move to play in the game. The Flask server now checks for a trained model on startup and will automatically run the training script if `2048_model.h5` is missing.
How to use:
Simply run `2048_flask_server.py`. The first run might take a while as it trains the network. Once the server is running, open `2048_Game-AI_PlayOG.html` and press **AI Play**.
This model was not amazing for me, and its record, with me helping it, was it got to the 256 tile.

AI #2: Monte Carlo Tree Search (MCTS)
This AI uses random sampling of the game state to simulate possible moves and select the best one. It now optionally leverages the neural network model to guide move selection which allows fewer simulations (5000 by default).
How to use:
Run `2048_mcts_ai.py` and open `2048_Game-AI_Play.html`. The script will attempt to load the neural network model and train it automatically if needed.
This model has performed a bit better than the neural network alone, with its record reaching the 1048 tile, but not further.
