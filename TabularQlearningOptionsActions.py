import collections
import random
from RoomEnv import TheRoom
import numpy as np
import csv

"""
Tabular Q-learning with options and actions applied to "The Room" environment.
Both Q(s,a) and Q(s,o) values are saved on a Table, to help deciding which option or action is the best for each state.
Options are "actions" that takes more than one step (k, time interval) to conclude, making the model a Semi-Markov Decision
Process. In the Options Framework context, actions are also known as "primitive actions" and can be considered a particular
case of option, where k (time intervals) = 1.
Each of these options have a sub-objective to pursuit, i.e., each have their own policy, which leads to specific
q-table per option.     

"""

EPSILON = 0.2
ALPHA = 1
GAMMA = 0.9

EPSILON_OPT = 0.1
ALPHA_OPT = 1
GAMMA_OPT = 0.9

UPDATE = 1000000000
DECAY = 0.02
MIN_EPSILON = 0.02
TRAINING = 50
CONVERGENCE = 17
TEST = 1


class AgentQOA:

    def __init__(self):
        """
        Agent is the entity that interacts with the environment
        """
        self.env = TheRoom(initial_state=(1, 1), objective=(10, 9))     # options=True)

        # general Q-table, with 0 as default value
        self.q_table = collections.defaultdict(float)

        # Set of possible options
        self.options_space = [12,
                              13,
                              21,
                              24,
                              31,
                              34,
                              42,
                              43]

        # Options can have a specific starting point (instead of being selectable in any state), i.e., each state can
        # start a specific set of options. This set is represented by the greek letter IOTA
        self.options_I = {
            12: [self.env.room1, self.env.hall_1_3],    # option 12 can start on any state from room1 or on hallway between room 1 and 3
            13: [self.env.room1, self.env.hall_1_2],
            21: [self.env.room2, self.env.hall_2_4],
            24: [self.env.room2, self.env.hall_1_2],
            31: [self.env.room3, self.env.hall_3_4],
            34: [self.env.room3, self.env.hall_1_3],
            42: [self.env.room4, self.env.hall_3_4],
            43: [self.env.room4, self.env.hall_2_4],
        }

        # Each option has a (sub)-objective that tries to help the model to reach its main objective
        # This sub-objective is represented by the greek letter BETA
        self.options_B = {12: self.env.hall_1_2,        # option 12 objective is to reach the hallway between room 1 and 2
                          13: self.env.hall_1_3,
                          21: self.env.hall_1_2,
                          24: self.env.hall_2_4,
                          31: self.env.hall_1_3,
                          34: self.env.hall_3_4,
                          42: self.env.hall_2_4,
                          43: self.env.hall_3_4}

        # each option has a specific policy/q-table
        self.q_table_o1 = collections.defaultdict(float)
        self.q_table_o2 = collections.defaultdict(float)
        self.q_table_o3 = collections.defaultdict(float)
        self.q_table_o4 = collections.defaultdict(float)
        self.q_table_o5 = collections.defaultdict(float)
        self.q_table_o6 = collections.defaultdict(float)
        self.q_table_o7 = collections.defaultdict(float)
        self.q_table_o8 = collections.defaultdict(float)

        self.options_q_tables = {12: self.q_table_o1,
                                 13: self.q_table_o2,
                                 21: self.q_table_o3,
                                 24: self.q_table_o4,
                                 31: self.q_table_o5,
                                 34: self.q_table_o6,
                                 42: self.q_table_o7,
                                 43: self.q_table_o8}

    def get_state_options_set(self, state):
        """
        get the set of options the said state can takes. Options have a limited set of states that can start them
        :param state: State to check the possible options it can take
        :return: the set of options for the said state
        """

        options_set = []

        for option, limits in self.options_I.items():

            room = limits[0]
            hallway = limits[1]

            # if the state is within the limits of the option, add it to the list of possible options
            if state[0] in range(room[0][0], room[1][0]) and state[1] in range(room[0][1], room[1][1]):
                options_set.append(option)

            elif tuple(state) == tuple(hallway):
                options_set.append(option)

        return options_set

    def best_value_and_action(self, state, q_table, option=2):
        """
        Get form the Q-table the best action or option and its respective value
        :param q_table: q_table to search for best value
        :param state: state to search for best action
        :param option: type of action to chose (primitive action, option or both)
        :return: best Q(s,a/o) and its a/o
        """

        # check the kind of action to search for

        # only primitive actions
        if option == 0:
            set_actions = self.env.primordial_actions_space
        # only options
        elif option == 1:
            set_actions = self.options_space
        # Both options and primitive actions
        elif option == 2:
            set_actions = self.env.primordial_actions_space + self.options_space

        best_value, best_action = None, None

        # check every action on the selected state
        for action in set_actions:

            # q-table has 0 as default
            action_value = q_table[(state, action)]

            if best_value is None or best_value < action_value:

                best_value = action_value
                best_action = action

            # elif best_value == action_value:
            #     best_value = random.choices([best_value, action_value])[0]

        return best_value, best_action

    def get_action(self, state, q_table, explore=0.0, option=2):
        """
        Chooses an action or option, respecting the relation Exploit x Explore and using the q-table
        :param state: state on where to act
        :param q_table: q_table to consult for best value
        :param explore: EPSILON hyperparameter
        :param option: type of action to chose (primitive action, option or both)
        :return: a/o according to Explore x Exploit hyperparameter
        """

        if option is 0:
            action_space = list.copy(self.env.primordial_actions_space)
        elif option is 1:
            action_space = self.get_state_options_set(state)
        elif option is 2:
            action_space = self.env.primordial_actions_space + self.get_state_options_set(state)

        # if the sum of the possible actions for a given state is 0, it indicates that the agent never iterated
        # through here. In this case, we select the next action randomly, independent of the explore x exploit.
        # It must be done because, if we don't select at random, the agent will always take the first
        # action on the list, skewing the result in the direction of this action.
        aux = sum([q_table[state, act] for act in action_space])
        if aux == 0:
            action = random.choices(action_space)[0]

        else:

            # take the best action
            _, action = self.best_value_and_action(state, q_table, option)

            # Explore
            # Ignores best action and take another one randomly
            if random.random() < explore:

                action_space.remove(action)
                action = random.choices(action_space)[0]

        return action

    def value_update(self, state, action, reward, next_s, q_table, steps=1):
        """
        update q-table according to the formula
        Q(s,a) = (1 - ALPHA) * oldQ(s,a) + ALPHA * (r + GAMMA * newQ(s,a))

        where old denotes the value before the update

        :param state: current state
        :param action: action taken
        :param reward: reward received
        :param next_s:  new state accessed
        :param q_table: q-table being updated
        :param steps: how many steps between s and next_s, required for options value update, since it uses the formula

        E{r + (GAMMA**k) * Vo(s')}, where
        k = steps
        r = ro + ... + (rt+k)*(GAMMA**k-1)

        :return:
        """

        best_v, _ = self.best_value_and_action(next_s, q_table, 2)

        if steps > 1:
            alpha = ALPHA_OPT
            gamma = GAMMA_OPT
        else:
            alpha = ALPHA
            gamma = GAMMA

        old_val = q_table[(state, action)]
        new_val = reward + (gamma ** steps) * best_v

        q_table[(state, action)] = (1 - alpha) * old_val + alpha * new_val

    def play_episode(self, explore=0.0, options_trained=True):
        """
        Play an entire episode on the environment, i.e., agent navigates on it until the objective is reached
        :param explore: probability of taking a random option/action instead of the best one (explore x exploit)
        :param options_trained: just an indicative if the options are already trained
        :return: number of interactions, i.e., how many times the agent selected an option; and total steps (time
        intervals) taken until the objective was reached
        """

        if options_trained is True:
            options_explore = 0.0
        else:
            options_explore = EPSILON_OPT

        # number of steps (k = 1) taken until the objective
        total_steps = 0

        # total reward accumulated
        total_reward = 0.0

        state = self.env.reset()
        is_done = False

        while True:

            action = self.get_action(state, self.q_table, explore, 2)

            # each type of action has an specific interaction with the env

            # action do a single transition between consecutive states
            if action in self.env.primordial_actions_space:

                new_state, reward, is_done = self.env.step(action, state)

                # actions take only one step
                steps = 1

            # options take N steps between states, not necessarily consecutive
            else:
                reward, steps, new_state = self.play_option(action, options_explore, state)

                # options only check their sub-objective, so we must check if the main objective was reached
                if new_state == tuple(self.env.objective):
                    reward = reward + 1  # accessed objective = +1 reward
                    is_done = True

            total_steps += steps

            self.value_update(state, action, reward, new_state, self.q_table, steps)

            total_reward += reward

            if is_done:
                break

            state = new_state

        return total_steps

    def play_step(self, state):
        """
        Execute an action on the chosen state. Action selection follows the q-table (there is no explore option)
        :param state: state where the action must be taken
        :return:
        """

        is_done = False

        _, action = self.best_value_and_action(state, self.q_table, 2)

        if action in self.env.primordial_actions_space:

            new_state, reward, is_done = self.env.step(action, state)

        else:
            reward, steps, new_state = self.play_option(action, 0, state)

            # if the main objective was reached, end of episode
            if new_state == tuple(self.env.objective):
                is_done = True

        return new_state, is_done

    def play_option(self, option, explore, state=None):
        """
        Each option take N steps to accomplish its sub-objective, which end the interactions.
        These steps use regular actions, therefore, to select the best one, a q-table particular to the option is used.

        q-table -> policy

        :param option: option being executed
        :param explore: Explore x Exploit on the selecion of actions for the option
        :param state: state where the option must start
        :return: total reward accumulated from the main q-table, number of steps that generated reward (state transactions),
        total number of steps (even attempts to move to a wall), last state accessed
        """

        # option's q-table
        q_table = self.options_q_tables[option]

        # each option has its own objective
        sub_objective = self.options_B[option]

        # take default initial state if no other is indicated
        if state is None:
            state = self.env.reset()

        total_steps = 0

        # reward from the main policy
        total_reward = 0.0

        # relative_reward is specific for the option q-table
        relative_reward = 0

        is_done = False

        while True:

            action = self.get_action(state, q_table, explore, option=0)

            # interaction with env
            new_state, reward, _ = self.env.step(action, state)

            total_steps += 1

            if new_state == tuple(sub_objective):
                # we reward the agent only when the objective is reached, thus, there is no need to accumulate the reward
                relative_reward = 1
                is_done = True

            # if the state is outside the option's states set (its room + hallway), it shouldn't be accessed by it.
            # The interaction is ignored
            if not is_done:

                options_set = self.get_state_options_set(new_state)

                if option not in options_set:
                    continue

            # if the agent tried to move to a wall, the state doesn't change. If thats the case, there is no update to do
            if state != new_state:
                self.value_update(state, action, relative_reward, new_state, q_table)

            state = new_state

            total_reward = (GAMMA**total_steps) * reward

            if is_done:
                break

        return total_reward, total_steps, state

    def training(self, verbose=False):
        """
        Training both model and options
        :return:
        """

        self.q_table.clear()
        self.q_table_o1.clear()
        self.q_table_o2.clear()
        self.q_table_o3.clear()
        self.q_table_o4.clear()
        self.q_table_o5.clear()
        self.q_table_o6.clear()
        self.q_table_o7.clear()
        self.q_table_o8.clear()

        episode = 0
        explore = EPSILON
        model_conv_steps = 0
        min_steps = 99999999
        steps_episode = []

        while 1:

            steps = self.play_episode(explore, options_trained=False)
            episode += 1

            if verbose:
                print("{}".format(steps), end=" ")

            model_conv_steps += steps
            steps_episode.append(steps)

            # min tracer
            if steps < min_steps:
                min_steps = steps

            if episode % UPDATE == 0:

                if explore < DECAY:
                    explore = MIN_EPSILON
                else:
                    explore = explore - DECAY

            # 5 consecutive mins = convergence
            if min_steps <= CONVERGENCE:
                if verbose:
                    print("\nModel converged on {} episodes, after executing {} steps. "
                          "Best result was {} steps\n".format(episode, model_conv_steps, min_steps))
                break

        return episode, model_conv_steps, steps_episode

    def random_hyper_parameter_tuning(self):
        """
        :return:
        """

        episodes = []
        interactions = []

        epsilons = [0.2, 0.3, 0.1]        # [0.01, 0.1, 0.2, 0.3]
        alphas = [0.2, 0.3, 0.5, 1]             # [0.1, 0.2, 0.3, 0.4, 0.5, 1]
        gammas = [0.9, 0.95, 0.99]              # [0.9, 0.95, 0.99]

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
                    # print("Test:", train_test, _)
                    ep, inter, _ = self.training()

                    episodes.append(ep)
                    interactions.append(inter)

                mean_int = np.mean(interactions)
                std_int = np.std(interactions)

                if mean_int < best_steps and std_int < lower_std:

                    best_steps = mean_int
                    lower_std = std_int
                    best_hyper = train
                    print("new min: {}, std {}. Parameters {}".format(best_steps, lower_std, best_hyper))

                train_test += 1

        print("Best steps: {}, std {}. Best parameters {}.".format(best_steps, lower_std, best_hyper))

    def export_csv(self, steps_episode):

        with open("./q-learningOptionsActions.csv", "w", newline='') as csvfile:

            writer = csv.writer(csvfile)

            columns = [("Episodes"), ("Steps")]
            writer.writerow(columns)

            episodes = 1
            for step in steps_episode:

                formatted = [(episodes), step]
                writer.writerow(formatted)

                episodes += 1


if __name__ == "__main__":

    agent = AgentQOA()

    if TEST == 1:

        _, _, steps_ep = agent.training(True)

        agent.export_csv(steps_ep)

        for _ in range(25):
            agent.training(True)
    else:
        agent.random_hyper_parameter_tuning()
