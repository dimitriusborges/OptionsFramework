import collections
import random
from RoomEnv import TheRoom
import numpy as np
import csv

"""
Tabular Q-learning applied to "The Room" environment.
The Q(s,a) values are saved on a Table, to help deciding which action is the best one for each state.
Since this values are updated during the Agent-Environment interactions, there is no need to know the probability dis-
tribution to forecast results and apply the bellman equation 
"""

EPSILON = 0.1
ALPHA = 1
GAMMA = 0.9

UPDATE = 1000000000
DECAY = 0.02
MIN_EPSILON = 0.02

TRAINING = 100
CONVERGENCE = 17
TEST = 1


class AgentQ:

    def __init__(self):
        """
        Agent is the entity that interacts with the environment
        """
        self.env = TheRoom(initial_state=(1, 1), objective=(10, 9))

        self.state = self.env.reset()
        self.q_table = collections.defaultdict(float)  # general Q-table, with 0 as default value

    def sample_env(self):
        """
        Takes a random action and execute it on the current state
        :return: current/old state, action taken, reward received, new state accessed by (s,a), conclusion of the
        environment
        """
        action = self.env.sample_action()
        old_state = self.state

        new_state, reward, is_done = self.env.step(action)

        # if the objective was reached, reset the environment
        self.state = self.env.reset() if is_done else new_state

        return old_state, action, reward, new_state, is_done

    def best_value_and_action(self, state):
        """
        Get form the Q-table the best action and its respective value
        :param state: state to search for best action
        :return:
        """

        best_value, best_action = None, None

        # check every action on the selected state
        for action in range(self.env.primordial_actions_space_n):

            # q-table has 0 as default
            action_value = self.q_table[(state, action)]

            if best_value is None or best_value < action_value:

                best_value = action_value
                best_action = action

            # elif best_value == action_value:
            #     best_value = random.choices([best_value, action_value])[0]

        return best_value, best_action

    def get_action(self, state, explore=0.0):
        """
        Chooses an action, respecting the relation Exploit x Explore and using the q-table
        :param explore:
        :param state:
        :return:
        """

        action_space = list.copy(self.env.primordial_actions_space)

        # if the sum of the possible actions for a given state is 0, it indicates that the agent never iterated
        # through here. In this case, we select the next action randomly, independent of the explore x exploit
        # reasoning. It must be done because, if we don't select at random, the agent will always take the first
        # action on the list, skewing the result in the direction of this action.
        aux = sum([self.q_table[state, act] for act in action_space])
        if aux == 0:
            action = random.choices(action_space)[0]

        else:

            _, action = self.best_value_and_action(state)

            # Explore
            if random.random() < explore:

                action_space.remove(action)
                action = random.choices(action_space)[0]

        return action

    def value_update(self, s, a, r, next_s):
        """
        update q-table according to the formula
        Q(s,a) = (1 - ALPHA) * oldQ(s,a) + ALPHA * (r + GAMMA * newQ(s,a))

        where old denotes the value before the update

        :param s: current state
        :param a: action taken
        :param r: reward received
        :param next_s:  new state accessed
        :return:
        """
        best_v, _ = self.best_value_and_action(next_s)

        old_val = self.q_table[(s, a)]
        new_val = r + GAMMA*best_v

        self.q_table[(s, a)] = (1 - ALPHA) * old_val + ALPHA * new_val

    def play_episode(self, explore=0.0):
        """
        Play an entire episode on the environment, i.e., agent navigates on it until the objective is reached
        :param explore: probability of taking a random action instead of best action (explore x exploit)
        :return: total steps taken until the objective was reached
        """

        # number of steps taken until the objective
        total_steps = 0

        total_reward = 0.0

        state = self.env.reset()

        while True:

            action = self.get_action(state, explore)

            # interaction with env
            new_state, reward, is_done = self.env.step(action)

            # if a new state was reached, update q-table
            # a new state isn't reach only when the movement directs to a wall
            if state != new_state:
                self.value_update(state, action, reward, new_state)

            total_reward += reward
            total_steps += 1

            if is_done:
                break

            state = new_state

        return total_reward, total_steps

    def play_step(self, state):
        """
        Execute an action on the chosen state. Action selection follows the q-table (there is no explore option)
        :param state: state where the action must be taken
        :return:
        """

        _, action = self.best_value_and_action(state)
        new_state, reward, is_done = self.env.step(action)

        return new_state, is_done

    def training(self, verbose=False):
        """
        Execute several episodes, until the mean of the last 10 reaches a desired threshold
        :return:
        """

        explore = EPSILON
        episode = 0
        total_steps = 0
        min_steps = 9999999
        steps_episode = []

        self.q_table.clear()

        while 1:

            episode += 1

            _, steps = self.play_episode(explore)

            if verbose:
                print("{}".format(steps), end=" ")

            total_steps = total_steps + steps

            steps_episode.append(steps)

            # min tracer
            if steps < min_steps:
                min_steps = steps

            if episode % UPDATE == 0:

                if explore < DECAY:
                    explore = MIN_EPSILON
                else:
                    explore = explore - DECAY

            if min_steps <= CONVERGENCE:
                """"""
                if verbose:

                    if verbose:
                        print("\nModel converged on {} episodes, afeter executing {} interactions. "
                              "Best result was {}\n".format(episode, total_steps, min_steps))

                break

        return episode, total_steps, steps_episode

    def random_hyper_parameter_tuning(self):
        """
        :return:
        """

        episodes = []
        interactions = []

        epsilons = [0.01, 0.1]
        alphas = [1, 0.3]
        gammas = [0.9, 0.99]

        # epsilons = [0.01, 0.1, 0.2, 0.3]
        # alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 1]
        # gammas = [0.9, 0.95, 0.99]

        trained = []

        best_steps = 9999999
        lower_std = 999999
        best_hyper = 0

        train_test = 0
        while 1:

            if len(trained) >= (len(epsilons)*len(alphas)*len(gammas)):
                break

            else:

                EPSILON = random.choices(epsilons)[0]
                ALPHA = random.choices(alphas)[0]
                GAMMA = random.choices(gammas)[0]

                train = (EPSILON, ALPHA, GAMMA)

                if train in trained:
                    continue
                else:
                    trained.append(train)

                episodes.clear()
                interactions.clear()

                print("Training {}, with {}".format(train_test, train))

                for _ in range(TRAINING):
                    ep, inter, _ = self.training()

                    episodes.append(ep)
                    interactions.append(inter)

                # self.remove_outliers(interactions, 10)
                mean_int = np.mean(interactions)
                std_int = np.std(interactions)

                if mean_int < best_steps and std_int < lower_std:

                    best_steps = mean_int
                    lower_std = std_int
                    best_hyper = train
                    print("new avg min: {0:.2f}, std {1:.2f}. Parameters {2}".format(best_steps, lower_std, best_hyper))

                train_test += 1

        print("Best avg steps: {0:.2f}, std {1:.2f}. Best parameters {2}.".format(best_steps, lower_std, best_hyper))

    def remove_outliers(self, data, percent):
        """

        :param percent:
        :return:
        """

        percent = percent/2
        percent = percent / 100

        remove_n = int(len(data) * percent)

        for _ in range(0, remove_n):
            data.remove(max(data))

        for _ in range(0, remove_n):
            data.remove(min(data))

    def export_csv(self, steps_episode):

        with open("./q-learning.csv", "w", newline='') as csvfile:

            writer = csv.writer(csvfile)

            columns = [("Episodes"), ("Steps")]
            writer.writerow(columns)

            episodes = 1
            for step in steps_episode:

                formatted = [(episodes), step]
                writer.writerow(formatted)

                episodes += 1


if __name__ == "__main__":

    agent = AgentQ()

    if TEST == 1:

        # _, _, steps_ep = agent.training(verbose=True)
        # agent.export_csv(steps_ep)

        for _ in range(25):
            agent.training(verbose=True)

    else:
        agent.random_hyper_parameter_tuning()
