#!/usr/bin/env python3
import argparse
import sys

import numpy as np

class MultiArmedBandits():
    def __init__(self, bandits, episode_length, seed=42):
        self._generator = np.random.RandomState(seed)

        self._bandits = []
        for _ in range(bandits):
            self._bandits.append(self._generator.normal(0., 1.))
        self._done = True
        self._episode_length = episode_length


    # start the episode by reset
    def reset(self):
        self._done = False
        self._trials = 0
        return None

    # do one step in environment
    # returns new_state, reward, done, info dict
    def step(self, action):
        if self._done:
            raise ValueError("Cannot step in MultiArmedBandits when there is no running episode")
        self._trials += 1
        self._done = self._trials == self._episode_length
        reward = self._generator.normal(self._bandits[action], 1.)
        return None, reward, self._done, {}

parser = argparse.ArgumentParser()
parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
parser.add_argument("--episodes", default=100, type=int, help="Training episodes.")
parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")

parser.add_argument("--mode", default="greedy", type=str, help="Mode to use -- greedy, ucb and gradient.")
parser.add_argument("--alpha", default=0, type=float, help="Learning rate to use (if applicable).")
parser.add_argument("--c", default=1, type=float, help="Confidence level in ucb (if applicable).")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor (if applicable).")
parser.add_argument("--initial", default=0, type=float, help="Initial value function levels (if applicable).")


class Greedy:
    def __init__(self, args):
        self.t = 0
        self.args = args

        # initialize action values
        self.q = {}
        for i in range(args.bandits):
            self.q[i] = args.initial

        self.n = {}
        for i in range(args.bandits):
            self.n[i] = 0

    def update(self, action, reward):
        # compute ordinary mean iteratively
        if self.args.alpha == 0:
            self.n[action] += 1
            self.q[action] = self.q[action] + (1/self.n[action]) * (reward - self.q[action])
            self.t += 1
        else:
            self.q[action] = self.q[action] + self.args.alpha * (reward - self.q[action])



    def get_best_action(self):
        if np.random.rand() < self.args.epsilon:
            # exploration
            return np.random.randint(0, self.args.bandits)


        # exploitation

        best_score = float("-inf")
        best_action = 0
        for action in range(self.args.bandits):
            if self.q[action] > best_score:
                best_score = self.q[action]
                best_action = action
        return best_action




class Gradient:
    def __init__(self, args):
        self.args = args

        # initialize numerial preference
        self.h = {}
        for i in range(args.bandits):
            # Maybe initialize with 0
            self.h[i] = args.initial

    def update(self, action, reward):
        new_h = {}
        for a in range(self.args.bandits):
            new_h[a] = self.h[a] + self.args.alpha * reward * ((1 if a == action else 0) - self.softmax(a))

        self.h = new_h

    def softmax(self, action):
        sum  = 0
        for i in range(self.args.bandits):
            sum += np.e ** self.h[i]
        return (np.e ** self.h[action]) / sum

    def get_best_action(self):
        # sample from softmax distribution
        distribution = [self.softmax(a) for a in range(self.args.bandits)]
        return np.random.choice(list(range(self.args.bandits)), p=distribution)


class Ucb:
    def __init__(self, args):
        self.t = 0
        self.args = args

        # initialize action values
        self.q = {}
        for i in range(args.bandits):
            self.q[i] = args.initial

        self.n = {}
        for i in range(args.bandits):
            self.n[i] = 0

    def update(self, action, reward):
        # compute ordinary mean iteratively
        if self.args.alpha == 0:
            self.n[action] += 1
            self.q[action] = self.q[action] + (1/self.n[action]) * (reward - self.q[action])
            self.t += 1
        else:
            self.q[action] = self.q[action] + self.args.alpha * (reward - self.q[action])


    def get_best_action(self):
        vals = []

        for action in range(self.args.bandits):
            val = self.q[action] + self.args.c * ((np.log(self.t) / self.n[action]) ** 0.5)
            vals.append(val)
        return np.argmax(vals)





def main(args):
    # Fix random seed
    np.random.seed(args.seed)

    # Create environment
    env = MultiArmedBandits(args.bandits, args.episode_length)


    avg_rewards = []
    for episode in range(args.episodes):
        env.reset()
        rewards = []

        # TODO: Initialize parameters (depending on mode).
        if args.mode == "greedy":
            manager = Greedy(args)
        elif args.mode == "ucb":
            manager = Ucb(args)
        elif args.mode == "gradient":
            manager = Gradient(args)
        else:
            assert False

        done = False
        while not done:
            # TODO: Action selection according to mode
            action = manager.get_best_action()

            _, reward, done, _ = env.step(action)
            rewards.append(reward)

            # Update parameters
            manager.update(action, reward)

        avg_rewards.append(np.mean(rewards))

    # TODO: For every episode, compute its average reward (a single number),
    # obtaining `args.episodes` values. Then return the final score as
    # mean and standard deviation of these `args.episodes` values.
    return np.mean(avg_rewards), np.std(avg_rewards)

if __name__ == "__main__":
    mean, std = main(parser.parse_args())
    # Print the mean and std for ReCodEx to validate
    print("{:.2f} {:.2f}".format(mean, std))




# interpretation of softmax
# continuous generalization of argmax
    # not differentiable
# z_i - log of odds
# odds = p_i / (1 - p_i)