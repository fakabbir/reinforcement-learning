---
layout: default
title: Home
nav_order: 1
permalink: /
---

# Notes on Q -learning, Deep Q-learning, Policy Gradients, Actor Critic, and PPO.

> In draft mode. Updated on 14th Jan 2018. Please report any error or correction at fakabbir@gmail.com

## Before we start

**Supervised and Unsupervised Learning**
In the case of supervised learning, we train the network with the expected input and output and expect it to predict the output given a new input. In the case of unsupervised learning, we show the network some input and expect it to learn the structure of the data so that it can apply this knowledge to a new input.

**Reinforcement Learning: An approach with constantly reacting environment**
In RL, for each step an agent is trained by rewarding it for correct behavior and punishing it for incorrect behavior. In the context of deep reinforcement learning, the idea starts as possible way to use environemental feedbabck to train the model.
One approach could be :

a network is shown some input and is given a positive or negative reward based on whether it produces the correct output from that input.

It seems to be supervised learning but the key diffrence is that we need data to work out the algorithm, but here the key is the environement, which constantly react to our actions and we do not need to put data inside.

Lets say,

Our objective is to build a neural network to play the game of catch. Each game starts with a ball being dropped from a random position from the top of the screen. The objective is to move a paddle at the bottom of the screen using the left and right arrow keys to catch the ball by the time it reaches the bottom.

- How can we use Supervised learning and whats good with RL ?

  Astute readers might note that our problem could be modeled as a classification problem, where the input to the network are the game screen images and the output is one of three actions--move left, stay, or move right. However, this would require us to provide the network with training examples, possibly from recordings of games played by experts. An alternative and simpler approach might be to build a network and have it play the game repeatedly, giving it feedback based on whether it succeeds in catching the ball or not. This approach is also more intuitive and is closer to the way humans and animals learn.

## Reinforcement Learning

### Using Q table:

Recalling the problem statement mentioned above, we want to have an AI or Algorithm or Machine to learn and play the game by itself without human intervension with a goal to win the game or collect maximum points.

So lets say the game starts with the ball being droped randomly and we have to predict the movement of the peddle such that it leads to a step close to catching the ball. With RL the idea would be to calculate a score for each move(i.b left, right, no move) that maximizes the chance of catching the ball.

**How do we get the score ?**

The key theme for RL is always getting the score. and the way we get the score would change as we move to complex procedure. In the beggining we will use a table which stores the score for each move at a given stage. Note that each stage is called as a state. For a very simple game the number of states are small and thus the table could be filled easily by backtracking the usefullness of each action and use the same to get the score. Notice that in this learning process we have memorised everything and now will be perfect. While looks like its not learning due to the fact that we have memorized everything, for complex game the number of states are so huge that we can't memorize this table. Games with probabilitics outcome such as playing with human in which the next state depends on how the opponent reacts is then learning by constructing a function that returns the score. In this picture the idea is to get a neural network which learns the inside weights to get to a stage that it outputs the best move.

### Using Exploration and Explotation Technique

To create a q table which tells us what is the q value for each action at a given state. Instead of just copy pasting the q table we will learn it from the feedback from the environment. To do this we will use "Exploration Explotation Trade off" strategy.

While training, we will start the game and will decide if the move should be random or based on our learning (Q table). Since when the game starts, we won't have Q-table filled (learned) we will take a random move. As the game proceds we will reduce the tendency to take the random move.

```
exp_exp_tradeoff = random.uniform(0,1)

## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
if exp_exp_tradeoff > epsilon:
    action = np.argmax(qtable[state,:])

# Else doing a random choice --> exploration
else:
    action = env.action_space.sample()

# Take the action (a) and observe the outcome state(s') and reward (r)
new_state, reward, done, info = env.step(action)
state = new_state

# Reduce epsilon (because we need less and less exploration)
epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
```

### Filling up the Q table

Now lets say we made a move and now we know from the environment, how good is this move and what the next state.
We will make our pointer the next state and update the q value of that action at that state as follows.

```
# Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma *
                                    np.max(qtable[new_state, :]) - qtable[state, action])
```

Here the learning rate determines how fast we want to move to the new Q value returned by the environment.

since in many cases the future reward at the given state could not we redeemed. For example lets say we are playing a game in which collecting the future reward isn't possible at all.
You may complain that the rewards returned by the environment is but absolute. Well actually its not. Many times its a function which returns a reward which is probabilictis. So in many case its good to give more weightage to the present reward rather than going for future rewards. To reduce the effect of future rewards we use gamma, also known as discout factor.

### Deep Q learning

Lets say we want to play the game of catch, we are interested in moving the paddle to maximize reward. Now to do so we will use the neural network to predict the motion.
We will stack 4 frames of screen do pre processing to remove unwamted patch from the image and pass it to a neural network to get the Q value for eacah action.

Now how do we train the model ?

In order to train the model, we get the images in 4 channels containng next frame. this is done in order to get the feeling of motion in order to make decision.

- At first we build a model and initialize it with random weights.
- For first few observation we take the state and store it in experience
- After few observations, we use exploration-exploitation technique to decide the step and optimize the loss
- the loss is calculated as rmse with X,Y pair where Y is calculated by model.predict() and Q for action is calculated as `reward + gamme * Q_sa`

```
CODE-A
Q_target = reward(s_t,a) + gamma * max(Q(s_tp1))
Q_predicted = model.predict(s_t)
```

- Note that the q value for future and present q value both are calculated via the same model and thus the reference changes for next step.
- To fix this we will use Fixed Q-values, Prioritized Experience Replay, Double DQN, Dueling Networks

### Fixed Q Target

In `CODE-A` we saw that we are using same network for Q_predict adn Q_target.
This has a drawback that the target itself is not fixed as the network's weights are changing at each step. In order to avoid this we will have two network. Q-Network and Target-Network. The Q-Network will predict the action and Target-Network would be used to predict the Q_target. The weights of T-Network would be same as that of Q-Network and would only be updated after few time steps.

### Double Q Network

This will use the two networks mentioned above to train the model.

### Duelingg Network

Till now we were calculating the Q value using the reward formula(aks Bellman's equation). For Dueling Network, we would take the fc layer and get two parallel layers. One would predict the Q Value and other would predict the advantage of taking that actions. It is similar to `Q = reward + future_reward` as V(s,a) is the equvalent or reward and A(s,a) advantage is equvalent to future_reward.

### Priorized Experience Replay.

Till now we were storing the game play adn taking the states randomly to train the model. This had an advantage that since when taken the experience in serial order as the states would be very similar to the previous only the training tends to move to local miinimum and trains hardly. With the experience stored and randomly selecting the state from experinece behaves similar to stocastic gradient descent method and avoids the local minimum trap. In order to train more efficently we use a priority score for taking the experince.

What is the score ? - https://arxiv.org/abs/1511.05952 - Score are stores in SumTree - Works for the game where rewards are less randomly selections doesn't help much.

### Policy Gradient

In policy-based methods, instead of learning a value function that tells us what is the expected sum of rewards given a state and an action, we learn directly the policy function that maps state to action (select actions without using a value function).

Here the exploration-exploitation techique is replaced by selecting for actions based on their probality disctribution. Thus at eah step the selection is random and is decided by the probability curve.

Its not a Temporal Diffrene Method (TD) so the rewards are stored and the policy is optimized using a score based on the rewards collected. Thus we use score maximization instead of error minimization and thus gradient ascent would be used.

### A2C

To imporve the performance we update the policy at each step (TD approach), but to calculate the reward we use a diffrent network.

### A3C

We use adnvantage function instead of the value function with A2C archetecture.
