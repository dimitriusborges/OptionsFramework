import random
import numpy as np
from operator import add

"""
The environment "The room", based on the proposed model presented by the article "Between MDPs and semi-MDPs 
- A framework for temporal abstraction in reinforcement learning".
"""

env_model = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
             [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
             [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
             [-1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0, -1],
             [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
             [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
             [-1, -1,  1, -1, -1, -1, -1,  0,  0,  0,  0,  0, -1],
             [-1,  0,  0,  0,  0,  0, -1, -1, -1,  1, -1, -1, -1],
             [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
             [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
             [-1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0, -1],
             [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
             [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
             ]


class TheRoom:

    def __init__(self, initial_state=None, objective=None, rdm_error=None, options=None):
        """
        The Room environment: 4 rooms with hallways between them, 2 by 2. It is modeled by a matrix, where each cell
        represents a state. Cells with -1 value are walls, with 1 are hallways and with 0 are common cells. The only
        possible reward (+1) is granted when reaching the objective state

        :param initial_state: Starting point, can be configured or use the default 1,1
        :param objective: Objective state, can be configured or use the default 11,11
        :param rdm_error: Random factor. If used, each action has a 1/3 chance to fail, where any other action can be
        chosen instead, 1/3 of probability each, making 2/3 chance to take the right action and 1/9 for each other
        action
        :param options: if the actions will only be options of K > 1
        """

        self.primordial_actions_space = [0,  # UP
                                         1,  # DOWN
                                         2,  # LEFT
                                         3]  # RIGHT

        self.primordial_actions_space_n = len(self.primordial_actions_space)

        # states that belongs to each room
        self.room1 = ((1, 1), (6, 6))
        self.room2 = ((1, 7), (7, 12))
        self.room3 = ((7, 1), (12, 6))
        self.room4 = ((8, 7), (12, 12))

        # hallways coordinates
        self.hall_1_2 = [3, 6]
        self.hall_1_3 = [6, 2]
        self.hall_2_4 = [7, 9]
        self.hall_3_4 = [10, 6]

        if initial_state is None:
            initial_state = [1, 1]

        if objective is None:
            objective = [7, 9]

        if rdm_error is True:
            self.rdm_error = True
            self.action_suc = [1, 0]  # yes, no
            self.action_suc_prob = [2 / 3, 1 / 3]  # prob yes, prob no
            self.action_failed = [1 / 3, 1 / 3, 1 / 3]  # if action failed, take another one with this prob distr

        self.options = options
        self.objective = objective
        self.initial_state = initial_state

        self.state = []

    def env_model(self):
        """
        Return the model matrix, 13x13
        :return:
        """
        return env_model

    def print_env(self):
        """
        Print the model on the console
        :return:
        """
        print(np.matrix(env_model))

    def sample_action(self):
        """
        Take a random choice from action set, with equal prob
        :return:
        """

        return random.choices(self.primordial_actions_space)[0]

    def reset(self, rdm=False):
        """
        Reset the environment, i.e., put the Agent on the initial state that can be random (excluding hallways)
        or a fixed one
        :return:
        """

        if rdm:

            # columns limits
            col_limit = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]

            # rows limits
            row_limit = [[1, 2, 3, 4, 5, 7, 8, 9, 10, 11],
                         [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]]

            y = random.choices(col_limit)[0]

            # pick a random position instead of a random value, each side of the model has a different set of possible rows
            x = random.choices(range(0, len(row_limit[0])))[0]

            # model isn't symmetric, must pick a side before picking a row
            side = 0 if y < 6 else 1

            self.state = [row_limit[side][x], y]

        else:
            self.state = self.initial_state

        return tuple(self.state)

    def act(self, action):
        """
        Get the selected action and return its movement direction on the environment matrix
        :param action:
        :return:
        """
        if self.options is None:

            return {0: [+1, 0],     # UP
                    1: [-1, 0],     # DOWN
                    2: [0, +1],     # LEFT
                    3: [0, -1]      # RIGHT
                    }.get(action)

        else:
            """"""

    def step(self, action, state=None, rdm_error=None):
        """
        Execute the primordial action and return the new state with the instant reward
        :param action:
        :return:
        """
        reward = 0
        done = False

        if state is None:
            state = self.state

        # if there is chance of error on the action execution
        if rdm_error:

            # check success
            success = random.choices(self.action_suc, self.action_suc_prob)[0]

            # if failed
            if success == 0:
                # take all other actions
                other_actions = [act for act in self.primordial_actions_space if action != act]

                # and select one of them randomly, 1/3 chance each
                final_action = random.choices(other_actions, self.action_failed)[0]

            # success, keep action
            else:
                final_action = action

        else:
            final_action = action

        target_state = list(map(add, state, self.act(final_action)))

        # if target state is a wall, keep original state
        # else, assume new_state
        if env_model[target_state[0]][target_state[1]] != -1:
            state = target_state
            self.state = state

        # objective reached, end of the model
        if state[0] == self.objective[0] and state[1] == self.objective[1]:
            reward = 1
            done = True

        return tuple(state), reward, done


# Test execution
if __name__ == "__main__":
    """"""
    env = TheRoom(options=True)
    env.print_env()
    env.reset()

    # for _ in range(0, 100):
    #     action = env.sample_action()
    #     env.step(action)



