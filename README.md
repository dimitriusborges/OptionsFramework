# OptionsFramework

## Definition

Options Framework (OF) is extension of Reinforcement Learning, which proposes the use of *options*, a special type of action that takes
more than one time step to end.
The combination of RL with OF may be referenced as Hierarchical Learning (HL).

**Main OF's article:**
 *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning*, by R. S. Sutton, D. Precup, and S. Singh.

The objective of this project is to apply a Prove of Concept of the said framework, where three approaches are proposed:

- [Tabular Q-learning applying only actions](https://github.com/dimitriusborges/OptionsFramework/new/master?readme=1#tabular-q-learning-with-only-actions)
- [Tabular Q-learning applying only indivisible options](https://github.com/dimitriusborges/OptionsFramework/new/master?readme=1#tabular-q-learning-applying-only-indivisible-options)
- [Tabular Q-learning appyling both indivisible options and actions](https://github.com/dimitriusborges/OptionsFramework/new/master?readme=1#tabular-q-learning-appyling-both-indivisible-options-and-actions)

Offers a simple Graphic Interface to show best case results. 

## Environment ([RoomEnv](RoomEnv.py))

The environment is inspired by the one defined in the main article:

>As a simple illustration of planning with options, consider the rooms example, a
>gridworld environment of four rooms as shown in Fig. 2. The cells of the grid correspond to
>the states of the environment. From any state the agent can perform one of four actions, up,
>down, left or right, which have a stochastic effect. With probability 2/3, the actions
>cause the agent to move one cell in the corresponding direction, and with probability 1/3,
>the agent moves instead in one of the other three directions, each with probability 1/9. In
>either case, if the movement would take the agent into a wall then the agent remains in the
>same cell. For now we consider a case in which rewards are zero on all state transitions.

It is important to note that, here, **the stochastic behavior is ignored**, i.e., every action has 100% chance of being executed 
according.

The model is represented by a matrix. 
Each action: up, down, left, right; is represented by INTs: 0, 1, 2 and 3; respectively.
The reward is different of 0 only when the objective is reached. 

## Agents

### Tabular Q-learning with only actions ([TabularQlearning](TabularQlearning.py))

Applies the traditional Tabular Q-learning.

- The initial state is the coordinate (1,1)
- The objective is the coordinate (10,9)
- Training consists in playing full episodes (going from initial state to objective) until less than 18 steps are taken. The q-table
is updated between actions.

### Tabular Q-learning applying only indivisible options ([TabularQlearningOptions](TabularQlearningOptions.py))

Applies Tabular Q-learning where the available actions are only options. These actions are seen as one entity by the agent, i.e., 
the agent doesn't know what steps the option took.
Each option has as objective to take the agent from its actual room to one of the hallways:

- Option 1: Lead agent on the room 1 to hallway between room 1 and 2
- Option 2: Lead agent on the room 1 to hallway between room 1 and 3
- Option 3: Lead agent on the room 2 to hallway between room 2 and 1
- Option 4: Lead agent on the room 2 to hallway between room 2 and 4
- Option 5: Lead agent on the room 3 to hallway between room 3 and 1
- Option 6: Lead agent on the room 3 to hallway between room 3 and 4
- Option 7: Lead agent on the room 4 to hallway between room 4 and 2
- Option 8: Lead agent on the room 4 to hallway between room 4 and 3

Each option has its own q-table, which is oriented to execute is own sub-objective. When this objective is reached, the env returns 
+1 as reward and the option is ended.
It is important to notice that this model can only navigate to hallways, i.e., there is no way to go from any coordinate to an 
adjacent one.

- The initial state is the coordinate (1,1)
- The objective is the coordinate (7,9)
- Training consists in play full episodes (going from intial state to objective) until 14 steps are taken. Each option executes a 
subset of interactions with the environment, where its q-table values are updated. The main q-table is updated between options/actions.

### Tabular Q-learning appyling both indivisible options and actions ([TabularQlearningOptionsActions](TabularQlearningOptionsActions.py))


Applies Tabular Q-learning using options and actions. These options are seen as one entity by the agent, i.e., 
the agent doesn't know what steps the option took.
Each option has as objective to take the agent from its actual room to one of the hallways:

- Option 1: Lead agent on the room 1 to hallway between room 1 and 2
- Option 2: Lead agent on the room 1 to hallway between room 1 and 3
- Option 3: Lead agent on the room 2 to hallway between room 2 and 1
- Option 4: Lead agent on the room 2 to hallway between room 2 and 4
- Option 5: Lead agent on the room 3 to hallway between room 3 and 1
- Option 6: Lead agent on the room 3 to hallway between room 3 and 4
- Option 7: Lead agent on the room 4 to hallway between room 4 and 2
- Option 8: Lead agent on the room 4 to hallway between room 4 and 3

Each option has its own q-table, which is oriented to execute is own sub-objective. When this objective is reached, the env returns 
+1 as reward and the option is ended.

- The initial state is the coordinate (1,1)
- The objective is the coordinate (10,9)
- Training consists in play full episodes (going from intial state to objective) until less than 18 steps are taken. Each option executes a 
subset of interactions with the environment, where its q-table values are updated. The main q-table is updated between options/actions.

## Graphic Interface ([GUI](GUI.py))

A simply Graphic Interface is implemented. It executes the training of each agent and is capable of show the interactions between 
agent and environment, **after convergence**.

## Disclaimer

Some liberties has been taken:

- The stochastic behavior while taking actions was ignore
- The paper says:

> (...) we also provide a priori its (options) accurate model (...)

  Which implies that the options weren't trained with the main-objective training. Here, both trainings were executed together
