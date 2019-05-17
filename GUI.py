import threading
from TabularQlearning import AgentQ
from TabularQlearningOptions import AgentQO
from TabularQlearningOptionsActions import AgentQOA
from tkinter import *
from tkinter import ttk

import time
from RoomEnv import TheRoom

lngt = 20


class Gui:

    def __init__(self, agents, envs):
        """

        """
        self.root = Tk()
        self.agents = agents
        self.room = TheRoom()
        self.envs = envs
        self.state = ()
        self.stopped = True

    def build_window(self):
        """

        :return:
        """

        main_frame = Frame(self.root, bg='white')
        main_frame.pack()

        model = self.room.env_model()
        width = len(model[0])
        height = len(model[1])

        self.box_agents = ttk.Combobox(main_frame,
                                       values=("Q-Learning", "Q-learning with options", "Q-learning with options and actions"),
                                       width=28)
        self.box_agents.grid(row=0, column=0)
        self.box_agents.current(0)

        self.restart = Button(main_frame, text='Start', command=self.restart_anim)
        self.restart.grid(row=0, column=1)

        self.canv = Canvas(main_frame, width=width*lngt, height=height*lngt, bg='gray')
        self.canv.grid(row=1, column=0, columnspan=2)

        for h in range(height):
            y = h * lngt
            for w in range(width):
                x = w * lngt

                if model[h][w] == 1:
                    color = 'purple'
                elif model[h][w] == 0:
                    color = 'red'
                else:
                    color = 'grey'

                self.canv.create_rectangle(x, y, x+lngt, y+lngt, fill=color, tag="state({},{})" .format(h, w))

        self.thread_agent = threading.Thread(target=self.anim_agent)
        self.thread_agent.daemon = True
        self.thread_agent.start()

    def remove_agent(self):

        room = self.envs[self.box_agents.current()]

        if self.state[0] == room.objective[0] and self.state[1] == room.objective[1]:
            color = 'green'
        elif room.env_model()[self.state[0]][self.state[1]] == 0:
            color = 'red'
        elif room.env_model()[self.state[0]][self.state[1]] == 1:
            color = 'purple'

        self.canv.itemconfigure("state({},{})".format(self.state[0], self.state[1]), fill=color)

    def place_agent(self):

        self.canv.itemconfigure("state({},{})".format(self.state[0], self.state[1]), fill='yellow')

    def restart_anim(self):

        try:
            self.remove_agent()
        except IndexError:
            """"""

        room = self.envs[self.box_agents.current()]

        self.state = room.reset()

        self.canv.itemconfigure("state({},{})".format(room.objective[0], room.objective[1]), fill='green')

        self.place_agent()

        self.stopped = False
        self.restart.config(state='disabled')

    def anim_agent(self):

        while self.stopped:
            time.sleep(0.3)

        while True:

            agent = self.agents[self.box_agents.current()]

            self.remove_agent()

            self.state, is_done = agent.play_step(self.state)

            self.place_agent()
            time.sleep(0.2)

            if is_done:
                self.restart.config(state='active')
                self.restart.config(text='Restart')
                self.stopped = True
                while self.stopped:
                    time.sleep(0.3)

                time.sleep(0.5)

                self.restart.config(state='disabled')

    def open_window(self):

        self.build_window()
        self.root.mainloop()


if __name__ == "__main__":

    agentq = AgentQ()
    agentqo = AgentQO()
    agentqoa = AgentQOA()

    agentq.training(True)
    time.sleep(1)
    agentqo.training(True)
    time.sleep(1)
    agentqoa.training(True)

    gui = Gui([agentq, agentqo, agentqoa], [agentq.env, agentqo.env, agentqoa.env])

    gui.open_window()



