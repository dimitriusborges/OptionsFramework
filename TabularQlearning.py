import collections
import random
from RoomEnv import TheRoom
import numpy as np

"""
Tabular Q-learning applied to "The Room" environment.
The Q(s,a) values are saved on a Table, to help deciding which action is the best one for each state.
Since this values are updated during the Agent-Environment interactions, there is no need to know the probability dis-
tribution to forecast results and apply the bellman equation 
"""

ALPHA = 0.4
GAMMA = 0.99

EPSILON = 0.1
UPDATE = 1000000000
DECAY = 0.02
MIN_EPSILON = 0.02


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

            elif best_value == action_value:
                best_value = random.choices([best_value, action_value])[0]

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
        episodes = 0
        memory_steps = []   # save the steps of the last 10 episodes
        total_interactions = 0

        self.q_table.clear()

        while 1:

            episodes += 1

            if len(memory_steps) >= 10:
                memory_steps.pop(0)

            _, steps = self.play_episode(explore)

            memory_steps.append(steps)
            total_interactions = total_interactions + steps

            if episodes % UPDATE == 0:

                if explore < DECAY:
                    explore = MIN_EPSILON
                else:
                    explore = explore - DECAY

            if min(memory_steps) < 18:
                """"""
                if verbose:

                    if total_interactions > 20000:
                        print(memory_steps)
                        print("Solved in {} episodes, with a total of {} interactions. "
                              "Best result was {}".format(episodes, total_interactions, min(memory_steps)))

                break

        return episodes, total_interactions

    def random_hyper_parameter_tuning(self):
        """
        :return:
        """

        episodes = []
        interactions = []

        epsilons = [0.01, 0.1, 0.2, 0.3]
        alphas = [0.1, 0.2, 0.3, 0.4]
        gammas = [0.9, 0.95, 0.99]

        trained = []

        best_steps = 9999999
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

                print(train_test)

                for _ in range(100):
                    # print("Test:", train_test, _)
                    ep, inter = self.training()

                    episodes.append(ep)
                    interactions.append(inter)

                m_int = np.mean(interactions)

                if best_steps > m_int:

                    best_steps = m_int
                    best_hyper = train
                    print("new min: {}. Parameters {}".format(best_steps, best_hyper))

                train_test += 1

        print("Best steps: {}. Best parameters {}.".format(best_steps, best_hyper))


if __name__ == "__main__":

    agent = AgentQ()
    EPSILON = 0.2
    ALPHA = 0.2
    GAMMA = 0.95
    for _ in range(50):
        agent.training(verbose=True)
    # agent.random_hyper_parameter_tuning()
