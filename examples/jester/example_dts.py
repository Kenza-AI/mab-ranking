import numpy as np
import random
from tqdm import tqdm
import codecs
import csv
import matplotlib.pyplot as plt
from urllib.request import urlopen

from mab_ranking.bandits.rank_bandits import IndependentBandits
from mab_ranking.bandits.bandits import DirichletThompsonSampling

N_RANKS = 10
URL = 'https://jester-jokes-data.s3.amazonaws.com/jesterfinal151cols.csv'


def get_jester_data(url):
    data = []
    file_stream = urlopen(url)
    csv_reader = csv.reader(codecs.iterdecode(file_stream, 'utf-8'), delimiter=',')
    for row in csv_reader:
        data_row = []
        for _item in row[1:]:
            if _item in {'99', ''} or float(_item) < 7.0:  # Rates above or equal to 7 are considered positive i.e. 1. Oterhwise 0.0.
                data_row.append(0.0)
            else:
                data_row.append(1.0)
        data.append(data_row)

    return np.asarray(data)


def main():
    jester_data = get_jester_data(URL)

    filtered_data = []
    for _row in jester_data:
        if sum(_row) > 1:  # Keep only the rows where a user has rated at least one 1 joke
            filtered_data.append(_row.tolist())

    data = np.asarray(filtered_data)

    independent_bandits = IndependentBandits(
        num_arms=data.shape[1],
        num_ranks=N_RANKS,  # Recommend the best 10 jokes
        bandit_class=DirichletThompsonSampling
    )

    num_steps = 1000
    sum_binary = 0.0
    ctr_list = []

    for i in tqdm(range(1, num_steps + 1)):
        # Pick a user randomly
        random_user_idx = random.randint(0, data.shape[0] - 1)

        ground_truth = np.argwhere(data[random_user_idx] == 1).flatten().tolist()
        n = len(ground_truth)

        selected_items = independent_bandits.choose(context={'previous_action': 0})

        hit_rate = len(set(ground_truth).intersection(set(selected_items))) / len(set(ground_truth))

        feedback_list = [1.0 if _item in ground_truth else 0.0 for _item in selected_items]
        independent_bandits.update(selected_items, feedback_list)

        user_binary_relevancy = 1.0 if hit_rate > 0 else 0.0
        sum_user_binary_relevancy = user_binary_relevancy
        # update the parameters sequentially given the user's last rated joke
        for j in range(1, n):
            selected_items = independent_bandits.choose(context={'previous_action': ground_truth[j-1]})

            hit_rate = len(set(ground_truth).intersection(set(selected_items))) / len(set(ground_truth))

            feedback_list = [1.0 if _item in ground_truth else 0.0 for _item in selected_items]
            independent_bandits.update(selected_items, feedback_list, context={'previous_action': ground_truth[j-1]})

            user_binary_relevancy = 1.0 if hit_rate > 0 else 0.0
            sum_user_binary_relevancy += user_binary_relevancy
        sum_binary += sum_user_binary_relevancy/n
        ctr_list.append(sum_binary / i)

    print('CTR at the last step: ' + str(ctr_list[-1]))

    def plot_ctr(num_iterations, ctr):
        plt.plot(range(1, num_iterations + 1), ctr)
        plt.xlabel('num_iterations', fontsize=14)
        plt.ylabel('ctr', fontsize=14)
        return plt

    plot_ctr(len(ctr_list), ctr_list).show()


if __name__ == '__main__':
    main()
