import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
from env2 import Environment, Directions, AgentStates


def create_quiver(X, Y, board):
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    mask = np.ones_like(U) == 1
    full_rows = len(board)
    full_cols = len(board[0])
    for row in range(full_rows):
        for col in range(full_cols):
            value = board[row][col]
            if value == Directions.NOP:
                mask[row, col] = False
            elif value == Directions.UP:
                U[row, col] = 0
                V[row, col] = 1
            elif value == Directions.RIGHT:
                U[row, col] = 1
                V[row, col] = 0
            elif value == Directions.DOWN:
                U[row, col] = 0
                V[row, col] = -1
            elif value == Directions.LEFT:
                U[row, col] = -1
                V[row, col] = 0
    return (U, V, mask)


def agent_data(agents):
    free_coords = []
    occupied_coords = []
    for agent in agents:
        if agent.state == AgentStates.AGENT_FREE:
            free_coords.append(agent.coord())
        elif agent.state == AgentStates.AGENT_OCCUPIED:
            occupied_coords.append(agent.coord())
    return (np.array(free_coords), np.array(occupied_coords))


class BoardGraphic:

    def __init__(self, ax, board, agents):
        cm = ListedColormap(['w', 'g', 'purple', 'b', 'y'], N=5)
        self.ax = ax
        self.ax_image = self.ax.matshow(board, cmap=cm, vmin=0.0, vmax=4.0)
        self.ax.grid(which="minor", lw=2.0)
        self.ax.xaxis.set_minor_locator(MultipleLocator(base=1.0, offset=-0.5))
        self.ax.yaxis.set_minor_locator(MultipleLocator(base=1.0, offset=-0.5))
        X, Y = np.meshgrid(np.arange(0, env.full_rows),
                           np.arange(0, env.full_cols))
        self.X = X
        self.Y = Y
        (U, V, mask) = create_quiver(X, Y, board)
        self.quiver = self.ax.quiver(X, Y, U, V)

        (free_agent_coords, occupied_agent_coords) = agent_data(env.agents)

        if free_agent_coords.size > 0:
            self.free_scat = self.ax.scatter(
                free_agent_coords[:, 1], free_agent_coords[:, 0], color="red")
        else:
            self.free_agent_coords = self.ax.scatter([], [], color="red")

        if occupied_agent_coords.size > 0:
            self.occupied_scat = self.ax.scatter(
                occupied_agent_coords[:, 1], occupied_agent_coords[:, 0], color="cyan")
        else:
            self.occupied_scat = self.ax.scatter([], [], color="cyan")

    def update_board(self, board):
        (U, V, mask) = create_quiver(self.X, self.Y, env.get_board())
        U = np.ma.array(U, mask=np.logical_not(mask))
        V = np.ma.array(V, mask=np.logical_not(mask))
        self.quiver.set_UVC(U, V)
        self.ax_image.set(data=board)

    def update_agents(self, agents):
        (free_data, occupied_data) = agent_data(agents)

        # free_data = np.stack([free_agent_cols, free_agent_rows]).T
        # occupied_data = np.stack([occupied_agent_cols, occupied_agent_rows]).T
        if free_data.size != 0:
            self.free_scat.set_offsets(free_data[:, ::-1])
        else:
            self.free_scat.set_offsets(np.stack([[], []]).T)

        if occupied_data.size != 0:
            self.occupied_scat.set_offsets(occupied_data[:, ::-1])
        else:
            self.occupied_scat.set_offsets(np.stack([[], []]).T)


env = Environment(32, 32, num_agents=40, inv_temp=0.5)
# env.board[0, 1:(env.full_cols-1)] = NOP
# env.board[1:(env.full_rows - 1), -1] = NOP
# env.board[-1, 1:(env.full_cols-1)] = NOP
# env.board[1:(env.full_rows - 1), 0] = NOP
#
# env.board[env.full_rows - 1, 8] = DOWN
# env.board[0, 8] = UP
# env.board[8, 0] = LEFT
# env.board[8, env.full_cols - 1] = RIGHT

env.set_cell(env.full_rows - 1, 8, Directions.DOWN, energy=-10, goal=True)
# env.set_cell(0, 8, Directions.UP, energy=-10, goal=True)
# env.set_cell(8, 0, Directions.LEFT, energy=-10, goal=True)
# env.set_cell(8, env.full_cols - 1, Directions.RIGHT, energy=-10, goal=True)

# env.random_goal()
# env.random_goal()
# env.random_goal()
# env.random_goal()
# env.random_goal()

# plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots()
env.update_board()

bg = BoardGraphic(ax, env.get_board(), env.agents)

#
#
#cm = ListedColormap(['w', 'g', 'purple', 'b', 'y'], N=5)
#ax_image = ax.matshow(env.get_board(), cmap=cm, vmin=0.0, vmax=4.0)
#ax.grid(which="minor", lw=2.0)
#ax.xaxis.set_minor_locator(MultipleLocator(base=1.0, offset=-0.5))
#ax.yaxis.set_minor_locator(MultipleLocator(base=1.0, offset=-0.5))

X, Y = np.meshgrid(np.arange(0, env.full_rows), np.arange(0, env.full_cols))


def fig_update(frame):
    # inv_temp = 0.85 + frame * 0.0025
    # env.inv_temp = inv_temp
    if frame % 10 == 0:
        env.random_goal()

    if frame % 2 == 0:
        env.update_board()
    else:
        env.update_agents()

    board = env.get_board()
    bg.update_board(board)
    bg.update_agents(env.agents)

    bg.ax.set_title(r"t: {}, Done: {} ".format(
        frame, env.is_done()))
    return bg.ax_image


frames = 100
ani = animation.FuncAnimation(
    fig=fig, func=fig_update, frames=frames, interval=100)
ani.save("32x32.gif")
