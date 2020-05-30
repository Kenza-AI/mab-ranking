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


class DirichletThompsonSampling(Bandit):
    def __init__(self, num_arms):
        super().__init__(num_arms)
        self.rewards = np.asarray([[1.0]] * self.num_arms ** 2).reshape(self.num_arms, self.num_arms, 1)

    @staticmethod
    def _get_previous_action(context):
        return context['previous_action'] if context else 0

    def _calculate_cond_prob(self, previous_action):
        unnorm_cond = [np.random.gamma(self.rewards[previous_action][i][0], 1) for i in range(self.num_arms)]
        sum_unnorm_cond = sum(unnorm_cond)
        probs = [x/sum_unnorm_cond for x in unnorm_cond]
        return probs

    def choose(self, context=None):
        previous_action = self._get_previous_action(context)

        # sample from the distribution given the previous action
        if previous_action != 0:
            cond_probs = self._calculate_cond_prob(previous_action)
            uncond_probs = self._calculate_cond_prob(0)
            probs = [x * y for x, y in zip(cond_probs, uncond_probs)]
        else:
            probs = self._calculate_cond_prob(previous_action)
        sorted_indices = np.argsort(probs)[::-1][-len(probs):].tolist()

        return sorted_indices[0], sorted_indices

    def update(self, arm_id, reward, context=None):
        previous_action = self._get_previous_action(context)

        # for the time being we assume that the transitions are symmetric
        self.rewards[previous_action][arm_id] += reward
        self.rewards[arm_id][previous_action] += reward

        if previous_action != 0:
            # makes sure that the branch that leads to current state is updated as well
            self.rewards[0][previous_action] += reward
