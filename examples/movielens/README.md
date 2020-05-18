# Recommend the Top K Movies

The jupyter notebook provides an application of the Dirichlet Thompson Sampling algorithm on movielens [dataset](http://files.grouplens.org/datasets/movielens/ml-100k.zip).

Here's the logic:

- Load the dataset and keep only events that include a rating greater or equal than 4 (positive feedback).
- Keep only events that include one of the top 100 rated movies based on popularity.
- Sort the events by timestamp.
- At each time step, call the `IndependentBandits` with `Dirichlet Thompson sampling` to choose the best 10 of them.
- Get the ground truth for the selected event.
- Get a reward of 1, if the recommended best 10 movies had a match with the ground truth.
- Update the `actions_dict` dictionary for the current user by adding the ground truth.