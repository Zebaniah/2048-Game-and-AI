There are 2 different AI versions here, as well as a non-ai version of the game. 


AI #1: Neural Network
This AI is trained on datasets and learns how to predict the best move to play in the game. This was my first attempt. 
How to use:
You must first download/create many datasets (I created all of the datasets that are here) and put them all in a folder. Then, enter the file PATHs starting on line 21 of the training script. Then, run the training script. Assuming no errors, it will create a model. Then, run the flask server and the OG AI play in tandem. Click AI play, and it will play the game.
This model was not amazing for me, and it's record, with me helping it, was it got to the 256 tile. 

AI #2: Monte Carlo Tree Search (MCTS)
This AI uses random sampling of the game state to simulate possible moves and select the best one.
How to use: 
Download the MCTS python file and run it. It will create a flask server. Then download and run 2048_Game-Ai_Play.html, hit AI play, and watch the AI do its thing.
This model has performed a bit better than the Nural network, with its record reaching the 512 tiles, but not further. 
