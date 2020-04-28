from collections import Counter
from unittest import TestCase

from mab_ranking.bandits.bandits import BetaThompsonSampling


class BetaThompsonSamplingTest(TestCase):
    def test_choose(self):
        num_arms = 10
        bandit = BetaThompsonSampling(num_arms=num_arms)
        best_arm_id, sorted_arm_ids = bandit.choose()

        assert best_arm_id in range(num_arms)

        assert len(sorted_arm_ids) == len(set(sorted_arm_ids))

        for _arm_id in sorted_arm_ids:
            _arm_id in range(num_arms)

    def test_update(self):
        bandit = BetaThompsonSampling(num_arms=4)

        bandit.update(1, reward=1.0)

        assert bandit.rewards == [1.0, 2.0, 1.0, 1.0]
        assert bandit.num_tries == [2.0, 3.0, 2.0, 2.0]

    def test_choose_and_update(self):
        bandit = BetaThompsonSampling(num_arms=2)

        chosen_arms = []
        # arm id 0 provides much more reward than arm id 1
        for _ in range(500):
            best_arm_id, _ = bandit.choose()
            chosen_arms.append(best_arm_id)

            reward = 1.0 if best_arm_id == 0 else 0.0
            bandit.update(best_arm_id, reward=reward)

        chosen_arms_with_frequency = Counter(chosen_arms)
        assert chosen_arms_with_frequency[0] > chosen_arms_with_frequency[1]
