import collections
import random
from RoomEnv import TheRoom
import numpy as np

"""
Tabular Q-learning with options applied to "The Room" environment.
The Q(s,o) values are saved on a Table, to help deciding which option is the best one for each state.
Options are "actions" that takes more than one step (time interval) to conclude, making the model a Semi-Markov Decision
Process. 
Here,  each of these options have a sub-objective to pursuit, i.e., each have their own policy, which leads to specific
q-table per option.     

"""

GAMMA = 0.95
EPSILON = 0.1
ALPHA = 0.4

UPDATE = 1000000000
DECAY = 0.02
MIN_EPSILON = 0.02


class AgentQO:

    def __init__(self):
        """
        Agent is the entity that interacts with the environment
        """
        self.env = TheRoom(initial_state=(1, 1), objective=(7, 9),)     # options=True)

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
        get the set of options the said state can takes
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

    def best_value_and_action(self, state, q_table, option=None):
        """
        Get form the Q-table the best action or option and its respective value
        :param q_table: q_table to search for best value
        :param state: state to search for best action
        :param option: type of action to chose
        :return: best Q(s,a/o) and its a/o
        """

        # check the kind of action to search for
        if option is None:
            set_actions = self.env.primordial_actions_space
        else:
            set_actions = self.options_space

        best_value, best_action = None, None

        # check every action on the selected state
        for action in set_actions:

            # q-table has 0 as default
            action_value = q_table[(state, action)]

            if best_value is None or best_value < action_value:

                best_value = action_value
                best_action = action

            elif best_value == action_value:
                best_value = random.choices([best_value, action_value])[0]

        return best_value, best_action

    def get_action(self, state, q_table, explore=0.0, option=None):
        """
        Chooses an action or option, respecting the relation Exploit x Explore and using the q-table
        :param state: state on where to act
        :param q_table: q_table to consult for best value
        :param explore: EPSILON hyperparameter
        :param option: type of action to chose
        :return: Q(s, a/o) and a/o
        """

        if option is None:
            action_space = list.copy(self.env.primordial_actions_space)
        else:
            action_space = self.get_state_options_set(state)

        # if the sum of the possible actions for a given state is 0, it indicates that the agent never iterated
        # through here. In this case, we select the next action randomly, independent of the explore x exploit
        # reasoning. It must be done because, if we don't select at random, the agent will always take the first
        # action on the list, skewing the result in the direction of this action.
        aux = sum([q_table[state, act] for act in action_space])
        if aux == 0:
            action = random.choices(action_space)[0]

        else:

            _, action = self.best_value_and_action(state, q_table, option)

            # Explore
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

        E{r + (GAMMA**k) * Vo(s')}, where k is the number of steps and r = ro + ... + (rt+k)*(GAMMA**k-1)

        :return:
        """

        if action in self.env.primordial_actions_space:
            best_v, _ = self.best_value_and_action(next_s, q_table)
        else:
            best_v, _ = self.best_value_and_action(next_s, q_table, option=True)

        old_val = q_table[(state, action)]
        new_val = reward + (GAMMA ** steps) * best_v

        q_table[(state, action)] = (1 - ALPHA) * old_val + ALPHA * new_val

    def play_episode(self, explore=0.0, options_trained=True):
        """
        Play an entire episode on the environment, i.e., agent navigates on it until the objective is reached
        :param explore: probability of taking a random option instead of best option (explore x exploit)
        :param options_trained: just an indicative if the options are already trained
        :return: number of interactions, i.e., how many times the agent selected an option; and total steps (time
        intervals) taken until the objective was reached
        """

        if options_trained is True:
            options_explore = 0.0
        else:
            options_explore = explore

        # number of steps (k = 1) taken until the objective
        total_steps = 0

        # total reward accumulated
        total_reward = 0.0

        state = self.env.reset()
        is_done = False

        while True:

            option = self.get_action(state, self.q_table, explore, option=True)

            # interaction with env
            reward, steps, new_state = self.play_option(option, options_explore, state)

            total_steps += steps

            # if the main objective was reached, end of episode
            if new_state == tuple(self.env.objective):
                reward = reward + 1     # accessed objective = +1 reward
                is_done = True

            # keep updating q-table
            self.value_update(state, option, reward, new_state, self.q_table, steps)

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

        option = self.get_action(state, self.q_table, 0, option=True)

        reward, steps, new_state = self.play_option(option, 0, state)

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
        :return: total reward accumulated from the main q-table, number of steps that generated reward (state transaction),
        total number of steps (even the atemps to move to a wall), final state reached,
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

            action = self.get_action(state, q_table, explore)

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

    def traininig_options(self):
        """
        Train all the options
        :return:
        """

        # each option has a strict set of states that can initialize them. So, to train each one, it is necessarily
        # to adjust the starting point
        initial_states = {self.env.room1: (1, 1),
                          self.env.room2: (1, 11),
                          self.env.room3: (11, 1),
                          self.env.room4: (11, 11)}

        # number of steps taken until the solution
        total_steps = 0

        for option in self.options_space:

            limits = self.options_I[option]

            initial_state = initial_states[limits[0]]

            self.env = TheRoom(initial_state, (7, 9))

            explore = 0.25 #EPSILON

            # How many consecutive episodes reached the obj using minimum steps
            conv_counting = 0

            episodes = 0
            min_steps = 9999
            steps_until_conv = 0

            while True:

                episodes += 1

                # epsilon adjustment
                if episodes % 5 == 0:

                    if explore < 0.1:
                        explore = 0.01
                    else:
                        explore = explore - 0.1

                # returns the number of steps until the obj was reached
                _, option_steps, _ = agent.play_option(option, explore)

                steps_until_conv += option_steps

                # min steps tracer
                if option_steps < min_steps:
                    min_steps = option_steps
                    conv_counting = 0

                elif option_steps == min_steps:
                    conv_counting += 1

                else:
                    conv_counting = 0

                # 5 times the same min is considered a convergence
                if conv_counting >= 5:
                    print("Option {} converged with {} steps, after {} total steps and {} episodes"
                          .format(option, min_steps, steps_until_conv, episodes))
                    total_steps += steps_until_conv
                    break

        print("All options trained in {} steps".format(total_steps))

        return total_steps

    def training_model(self):
        """
        Train the model using already trained options
        :return:
        """
        opt_conv_steps = self.traininig_options()

        self.env = TheRoom((1, 1), (7, 9))

        episodes = 0
        explore = EPSILON
        model_conv_steps = 0
        min_steps = 9999
        count_conv = 0

        while 1:

            steps = self.play_episode(explore)
            episodes += 1
            model_conv_steps += steps

            # min tracer
            if steps < min_steps:

                min_steps = steps
                count_conv = 0

            elif steps == min_steps:
                count_conv += 1

            else:
                count_conv = 0

            # epsilon adjustment
            if episodes % 5 == 0:
                if explore <= 0.1:
                    explore = 0.01
                else:
                    explore = explore - 0.1

            # 5 consecutive mins = convergence
            if count_conv >= 5:
                print("Model converged on {} episodes, after executing {} steps. "
                      "Best result was {} steps".format(episodes, model_conv_steps, min_steps))
                break

        total_steps = opt_conv_steps + model_conv_steps
        print("The model executed {} steps on the enviroment".format(total_steps))

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

        self.env = TheRoom((1, 1), (7, 9))

        episodes = 0
        explore = EPSILON
        model_conv_steps = 0
        min_steps = 9999

        while 1:

            steps = self.play_episode(explore, options_trained=False)
            episodes += 1
            model_conv_steps += steps

            # min tracer
            if steps < min_steps:
                min_steps = steps

                # if verbose:
                #     print("New min foound {}, on episode {}".format(min_steps, episodes))

            if episodes % UPDATE == 0:

                if explore < DECAY:
                    explore = MIN_EPSILON
                else:
                    explore = explore - DECAY

            # 5 consecutive mins = convergence
            if min_steps == 14:

                if verbose:
                    print("Model converged on {} episodes, after executing {} steps. "
                          "Best result was {} steps".format(episodes, model_conv_steps, min_steps))
                break

        return episodes, model_conv_steps

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

                for _ in range(50):
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

    agent = AgentQO()
    EPSILON = 0.1
    ALPHA = 0.4
    GAMMA = 0.95

    # for _ in range(10):
    #     agent.training(True)

    agent.random_hyper_parameter_tuning()

