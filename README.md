# mab-ranking

A library for Online Ranking with Multi-Armed-Bandits. This library is useful to find the best Top N items among a relatively small candidate set. 

For example:
1. Show the Top 5 News Articles among 200 articles every day
2. Recommend the Top 10 Trending Products every week
3. Rank components on websites to drive more engagement (Whole Page Optimization)

## Installation

### Prerequisites

mab-ranking requires the following:

1. Python (3.6, 3.7, 3.8)

### Install mab-ranking

At the command line:

    pip install mab-ranking

## Getting started

Let's say that you want to recommend the top 5 trending products to your website visitors every week among the 300 most selling products. Every Monday at some time T you'll define a new `RankBandit` implementation. For example:

```
num_ranks = 5
num_arms = 300
rank_bandit = IndependentBandits(num_ranks, BetaThompsonSampling, num_arms=num_arms)
```

Then, every time a visitor X is landed on your home page for the rest of the week, you need to select which 5 products to show them in section `Top Trending Products`. So, you'll do the following:

```
selected_arms = rank_bandit.choose()
```

Let's say that the `selected_arms` equals `[30, 2, 200, 42]`. That means that you need to show products, 30, 2, 200 and 42 in this order. You can keep your own mapping from product UUIDS to integer arm ids in your app's business logic.

The visitor X viewed this selected order. Let's say that (s)he clicked on products 2 and 42. Then the `rewards` list will be `[0.0, 1.0, 0.0, 1.0]`. So:

```
rewards = [0.0, 1.0, 0.0, 1.0]
rank_bandit.choose(selected_arms, rewards)
```

## Implementation Details

Implementations of `RankBandit`:
- `IndependentBandits`: from [Microsoft Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/kss_aaai2013.pdf), "A Fast Bandit Algorithm for Recommendations to Users with Heterogeneous" 

Implementations of `Bandit`:
- `BetaThompsonSampling`: Beta Thompson Sampling MAB
