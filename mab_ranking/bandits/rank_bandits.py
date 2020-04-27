from abc import ABCMeta, abstractmethod


class RankBandit(metaclass=ABCMeta):
    """
    Base abstract class to inherit from for Ranking Multi-Armed-Bandits implementations.

    Arm ids are 0-based indexed.
    """
    def __init__(self, num_ranks, bandit_class, **kwargs):
        self.num_ranks = num_ranks
        self.rank_bandits = [bandit_class(**kwargs) for _ in range(self.num_ranks)]

    @abstractmethod
    def choose(self, context=None):
        """
        Chooses an arm id for each position of the rank

        :param context: [dict, default=None], context of the request. Only discrete context values
        are supported of type str. Example:
        {
            "context_a": "value_a1",
            "context_b: "value_b3"
        }

        :return: [list[int]], a list of chosen arm id per position. It has non-duplicates.
        """
        pass

    def update(self, selected_arms, rewards, context=None):
        """
        Updates bandit algorithms

        :param selected_arms: [list[int]], a list of arm ids
        :param rewards: [list[float]], a reward per arm id in the list
        :param context: [dict, default=None], context of the request. Only discrete context values
        are supported of type str. Example:
        {
            "context_a": "value_a1",
            "context_b: "value_b3"
        }
        """
        for i, arm_reward_tuple in enumerate(zip(selected_arms, rewards)):
            arm = arm_reward_tuple[0]
            reward = arm_reward_tuple[1]
            self.rank_bandits[i].update(arm, reward, context=context)


class IndependentBandits(RankBandit):

    @staticmethod
    def find_next_possible_arm(selected_arms, ranked_arms):
        for arm in ranked_arms:
            if arm not in selected_arms:
                return arm

    def choose(self, context=None):
        selected_arms = []

        for i in range(self.num_ranks):
            selected_arm, ranked_arms = self.rank_bandits[i].choose()
            if selected_arm in selected_arms:
                selected_arm = self.find_next_possible_arm(selected_arms, ranked_arms)

            selected_arms.append(selected_arm)

        return selected_arms
