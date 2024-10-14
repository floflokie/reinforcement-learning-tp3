from collections import defaultdict
import random
import typing as t
import numpy as np
import gymnasium as gym


Action = int
State = int
Info = t.TypedDict("Info", {"prob": float, "action_mask": np.ndarray})
QValues = t.DefaultDict[int, t.DefaultDict[Action, float]]


class QLearningAgent:
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        gamma: float,
        legal_actions: t.List[Action],
    ):
        """
        Q-Learning Agent

        You shoud not use directly self._qvalues, but instead of its getter/setter.
        """
        self.legal_actions = legal_actions
        self._qvalues: QValues = defaultdict(lambda: defaultdict(int))
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma

    def get_qvalue(self, state: State, action: Action) -> float:
        """
        Returns Q(state,action)
        """
        return self._qvalues[state][action]

    def set_qvalue(self, state: State, action: Action, value: float):
        """
        Sets the Qvalue for [state,action] to the given value
        """
        self._qvalues[state][action] = value

    def get_value(self, state: State) -> float:
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_a Q(s, a) over possible actions.
        """
        value = 0.0
        # BEGIN SOLUTION
        if len(self.legal_actions) == 0:
            return value
        value = max([self.get_qvalue(state, action) for action in self.legal_actions])
        # END SOLUTION
        return value

    def update(
        self, state: State, action: Action, reward: t.SupportsFloat, next_state: State
    ):
        """
        You should do your Q-Value update here:

           TD_target(s, a, r, s') = r + gamma * V(s')
           TD_error(s, a, r, s') = TD_target(s, a, r, s') - Q_old(s, a)
           Q_new(s, a) := Q_old(s, a) + learning_rate * TD_error(s, a, R(s, a), s')
        """
        q_value = 0.0
        # BEGIN SOLUTION
        TD_target = reward + self.gamma * self.get_value(next_state)
        TD_error = TD_target - self.get_qvalue(state, action)
        q_value = self.get_qvalue(state, action) + self.learning_rate * TD_error
        # END SOLUTION

        self.set_qvalue(state, action, q_value)

    def get_best_action(self, state: State) -> Action:
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_q_values = [
            self.get_qvalue(state, action) for action in self.legal_actions
        ]
        index = np.argmax(possible_q_values)
        best_action = self.legal_actions[index]
        return best_action

    def get_action(self, state: State) -> Action:
        """
        Compute the action to take in the current state, including exploration.

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """
        action = self.legal_actions[0]

        # BEGIN SOLUTION
        if np.random.uniform(size=1)[0] < self.epsilon:
            action = random.choice(self.legal_actions)
        # on choisit l'action qui maximise Q(s, a)
        else:
            action = self.get_best_action(state=state)
        # END SOLUTION

        return action
