from collections import Counter
from unittest import TestCase

from mab_ranking.bandits.bandits import BetaThompsonSampling
from mab_ranking.bandits.rank_bandits import IndependentBandits


class IndependentBanditsTest(TestCase):
    def test_choose(self):
        num_ranks = 3
        rank_bandit = IndependentBandits(num_ranks, BetaThompsonSampling, num_arms=10)

        selected_arms = rank_bandit.choose()

        assert len(selected_arms) == num_ranks
        assert len(selected_arms) == len(set(selected_arms))

    def test_update(self):
        rank_bandit = IndependentBandits(3, BetaThompsonSampling, num_arms=8)

        selected_arms = [0, 1, 2]
        rank_bandit.update(selected_arms=selected_arms, rewards=[1.0, 0.0, 0.0])

        assert rank_bandit.rank_bandits[0].rewards == [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert rank_bandit.rank_bandits[0].num_tries == [3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

    def test_choose_and_update(self):
        rank_bandit = IndependentBandits(2, BetaThompsonSampling, num_arms=4)

        chosen_arm_in_first_position = []
        for _ in range(500):
            selected_arms = rank_bandit.choose()
            first_chosen_arm = selected_arms[0]
            chosen_arm_in_first_position.append(first_chosen_arm)

            # Only arm with id 1 at 1st position receives a reward
            reward = 1.0 if first_chosen_arm == 1 else 0.0
            rank_bandit.update(
                selected_arms=selected_arms,
                rewards=[reward, 0.0]
            )

        assert Counter(chosen_arm_in_first_position).most_common(1)[0][0] == 1
