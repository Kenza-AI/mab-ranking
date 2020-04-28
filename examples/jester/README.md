# Recommend the Top K Jokes

The jupyter notebook has implemented a simple recommender system that selects at each time step 10 jokes using the Jester [dataset](http://konect.cc/networks/jester2/).

Here's the logic:

- Load the dataset and keep only the users that have rated at least 1 joke.
- The rating is in the range [-10, 10].
- Transform ratings into binary ones. Translate ratings below 7 to 0. Otherwise, to 1.
- Set a number of time steps.
- At each time step, select a random user from the dataset and call the `IndependentBandits` to choose the best 10 of them.
- Get the ground truth for the selected random user.
- Increment the clicks by 1, if the recommended best 10 had at least one match with the ground truth.
