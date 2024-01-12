# Snake-RL
Playing Snake using Reinforcement Learning. Using Deep Q Learning(Value NN)

# To run
Clone Repo to local machine.
```
git clone git@github.com:monopolize-all/Snake-RL.git
cd Snake-RL
```

To play the snake game yourself, run snake.py.
```
python3 snake.py
```

To see Reinforcement Learning in progress, run agent.py. Here you can't control the snake yourself.\
You can speed up/slow down the game using mousescroll. FPS <= 0 means game will run at highest possible framerate.
```
python3 agent.py
```

Periodically, game will save its current progress into game scores.png which will show details like max score reached and steps taken per game.\
On closing the window, game will save its model to 'models/snake_valueNN.dat'. It will reopen same model data next time agent.py is run and continue training from there.
To train model afresh, delete 'models/snake_valueNN.dat'


# Model used
6 -> 64 -> 1 Fully Connected Neural Network with ReLU activation function.

Input: food_front, food_left, food_right, body_front, body_left, body_right
- food_\<dir\> denoting how close food is in that direction.
- body_\<dir\> denoting how close snake's own body is in that direction.

Output: Predicted State Value for given State


# Images
<img width="617" alt="Screenshot 2024-01-12 at 11 23 51 AM" src="https://github.com/monopolize-all/Snake-RL/assets/19649720/7919978d-f072-4a51-906c-411ac4843aaf">

<img width="628" alt="Screenshot 2024-01-12 at 11 23 26 AM" src="https://github.com/monopolize-all/Snake-RL/assets/19649720/859a7ab2-fa47-4b01-a37e-54a0e7bddc50">
State Values for each position and direction. Green denotes relative high and red relative low(releative for each position)
