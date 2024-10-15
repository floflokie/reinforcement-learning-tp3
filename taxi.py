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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore

#################################################
# 1. Play with QLearningAgent
#################################################

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
        total_reward += r
        agent.update(s, a, r, next_s)
        s = next_s
        if done:
            #env.close_video_recorder()
            env.close()
            break
    return total_reward

def grid_search(env, agent_type):

    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = np.zeros((9, 9))

    best_reward = float('-inf')
    best_lr = 0
    best_epsilon = 0

    grid_search_iter = 200

    for i, lr in enumerate(learning_rates):
        for j, eps in enumerate(epsilons):
            agent = agent_type(
                learning_rate=lr, epsilon=eps, gamma=0.99, legal_actions=list(range(n_actions))
            )
            
            rewards_qlearning = []
            for _ in range(grid_search_iter):
                rewards_qlearning.append(play_and_train(env, agent))
            
            mean_reward = np.mean(rewards_qlearning[-100:])
            results[i, j] = mean_reward
            print(f"LR: {lr}, Epsilon: {eps}, Mean reward: {mean_reward}")
            
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_lr = lr
                best_epsilon = eps

    # Plot the results
    plt.figure(figsize=(12, 10))
    sns.heatmap(results, annot=True, fmt='.2f', xticklabels=np.round(epsilons, 2), yticklabels=np.round(learning_rates, 2))
    plt.xlabel('Epsilon')
    plt.ylabel('Learning Rate')
    plt.title('Grid Search Results: Mean Reward for ' + str(grid_search_iter) + ' episodes')
    plt.savefig('grid_search_results_' + agent_type.__name__ + '.png')
    plt.close()

    print(f"Best hyperparameters: Learning Rate = {best_lr:.2f}, Epsilon = {best_epsilon:.2f}")
    print(f"Best mean reward: {best_reward:.2f}")
    
    return best_lr, best_epsilon

# Use the best hyperparameters for the final agent
best_lr, best_epsilon = grid_search(env, QLearningAgent)
agent = QLearningAgent(
    learning_rate=best_lr, epsilon=best_epsilon, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards_qlearning = []
for i in range(1000):
    rewards_qlearning.append(play_and_train(env, agent))
    if i % 250 == 0:
        vid = gym.wrappers.RecordVideo(env, "vids/q_learning", name_prefix="qlearning_training_ep_" + str(i))
        play_and_train(vid, agent) # play 1 episode for the video
        vid.close()
    if i % 10 == 0:
        print(f"Episode: {i}, Mean reward: {np.mean(rewards_qlearning[-100:])}")

env.close()

vid = gym.wrappers.RecordVideo(env, "vids/q_learning", name_prefix="qlearning_training_ep_1000")
play_and_train(vid, agent) # play 1 episode for the video
vid.close()

assert np.mean(rewards_qlearning[-100:]) > 0.0
print("qlearning done")

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################

# Use the best hyperparameters for the final agent
agent = QLearningAgentEpsScheduling(
    learning_rate=best_lr, epsilon=best_epsilon, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards_qlearning_eps = []
for i in range(1000):
    rewards_qlearning_eps.append(play_and_train(env, agent))
    if i % 250 == 0:
        vid = gym.wrappers.RecordVideo(env, "vids/q_learning_eps", name_prefix="qlearning_eps_training_ep_" + str(i))
        play_and_train(vid, agent) # play 1 episode for the video
        vid.close()
    if i % 10 == 0:
        print(f"Episode: {i}, Mean reward: {np.mean(rewards_qlearning_eps[-100:])}")

env.close()

vid = gym.wrappers.RecordVideo(env, "vids/q_learning_eps", name_prefix="qlearning_eps_training_ep_1000")
play_and_train(vid, agent) # play 1 episode for the video
vid.close()

assert np.mean(rewards_qlearning_eps[-100:]) > 0.0
print("qlearning_eps_scheduling done")

####################
# 3. Play with SARSA
####################

agent = SarsaAgent(
    learning_rate=0.7, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards_sarsa = []
for i in range(1000):
    rewards_sarsa.append(play_and_train(env, agent))
    if i % 250 == 0:
        vid = gym.wrappers.RecordVideo(env, "vids/sarsa", name_prefix="sarsa_training_ep_" + str(i))
        play_and_train(vid, agent) # play 1 episode for the video
        vid.close()
    if i % 10 == 0:
        print(f"Episode: {i}, Mean reward: {np.mean(rewards_sarsa[-100:])}")

env.close()

vid = gym.wrappers.RecordVideo(env, "vids/sarsa", name_prefix="sarsa_training_ep_1000")
play_and_train(vid, agent) # play 1 episode for the video
vid.close()

assert np.mean(rewards_sarsa[-100:]) > 0.0
print("sarsa done")

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(np.convolve(rewards_qlearning, np.ones(100)/100, mode='valid'), label='Q-Learning')
plt.plot(np.convolve(rewards_qlearning_eps, np.ones(100)/100, mode='valid'), label='Q-Learning with Epsilon Scheduling')
plt.plot(np.convolve(rewards_sarsa, np.ones(100)/100, mode='valid'), label='SARSA')
plt.xlabel('Episodes')
plt.ylabel('Mean Reward (100-episode moving average)')
plt.title('Learning Curves for Different Algorithms')
plt.legend()
plt.grid(True)
plt.savefig('learning_curves.png')
plt.show()
