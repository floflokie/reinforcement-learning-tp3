"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
import imageio

env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore

do_video = False
current_step = 0

#################################################
# 1. Play with QLearningAgent
#################################################

# You can edit these hyperparameters!
agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        # Get agent to pick action given state 
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        if do_video:
            frame = env.render()
            gif.add_frame(frame)

        total_reward += r
        agent.update(s, a, r, next_s)
        s = next_s
        if done and do_video:
            env.close()
            break
    return total_reward

class gifmaker:
    def __init__(self):
        self.frames = []
        
    def add_frame(self, frame):
        self.frames.append(frame)

    def save(self, filename):
        imageio.mimsave(filename, self.frames, fps=20)
        
    def clear(self):
        self.frames = []

gif = gifmaker()
rewards = []
for i in range(1000):
    if i % 10 == 0:
        do_video = True
    else:
        do_video = False
    rewards.append(play_and_train(env, agent))
    if i % 10 == 0:
        print("mean reward", np.mean(rewards[-100:]))
    current_step += 1

gif.save(f'taxi_game_qlearning.gif')
assert np.mean(rewards[-100:]) > 0.0

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################

agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

gif.clear()
rewards = []
for i in range(1000):
    if i % 10 == 0:
        do_video = True
    else:
        do_video = False
    rewards.append(play_and_train(env, agent))
    if i % 10 == 0:
        print("mean reward", np.mean(rewards[-100:]))

gif.save(f'taxi_game_qlearning_epsscheduling.gif')
assert np.mean(rewards[-100:]) > 0.0

####################
# 3. Play with SARSA
####################

agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)))

gif.clear()
rewards = []
for i in range(1000):
    if i % 10 == 0:
        do_video = True
    else:
        do_video = False
    rewards.append(play_and_train(env, agent))
    if i % 10 == 0:
        print("mean reward", np.mean(rewards[-100:]))
        
gif.save(f'taxi_game_sarsa.gif')