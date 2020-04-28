from abc import ABCMeta
from abc import abstractmethod

import numpy as np


class Bandit(metaclass=ABCMeta):
    """
    Base abstract class to inherit from for Multi-Armed-Bandits implementations.

    Arm ids are 0-based indexed.
    """
    def __init__(self, num_arms):
        self.num_arms = num_arms

    @abstractmethod
    def choose(self, context=None):
        """
        Chooses an arm id

        :param context: [dict, default=None], context of the request. Only discrete context values
        are supported of type str. Example:
        {
            "context_a": "value_a1",
            "context_b: "value_b3"
        }

        :return: [tuple[int, list[int]], a tuple where the left item is the best chosen arm id
        and the right item has a list of arm ids (including the best one in the first position)
        ordered in a decreasing order in terms of best possible arm.
        """
        pass

    @abstractmethod
    def update(self, arm_id, reward, context=None):
        """
        Updates algorithm given the feedback
        :param arm_id: [int], arm id
        :param reward: [float], reward for arm id
        :param context: [dict, default=None], context of the request. Only discrete context values
        are supported of type str. Example:
        {
            "context_a": "value_a1",
            "context_b: "value_b3"
        }
        """
        pass


class BetaThompsonSampling(Bandit):

    def __init__(self, num_arms):
        super().__init__(num_arms)
        self.rewards = [1.0 for _ in range(self.num_arms)]
        self.num_tries = [2.0 for _ in range(self.num_arms)]

    def choose(self, context=None):
        probs = [
            np.random.beta(total_reward, n - total_reward)
            for n, total_reward in zip(self.num_tries, self.rewards)
        ]

        sorted_indices = np.argsort(probs)[::-1][-len(probs):].tolist()

        return sorted_indices[0], sorted_indices

    def update(self, arm_id, reward, context=None):
        self.num_tries[arm_id] += 1
        self.rewards[arm_id] += reward
