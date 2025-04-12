import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
from env2 import Environment, Directions

env = Environment(32, 32, num_agents=10, inv_temp=0.5)
# env.board[0, 1:(env.full_cols-1)] = NOP
# env.board[1:(env.full_rows - 1), -1] = NOP
# env.board[-1, 1:(env.full_cols-1)] = NOP
# env.board[1:(env.full_rows - 1), 0] = NOP
#
# env.board[env.full_rows - 1, 8] = DOWN
# env.board[0, 8] = UP
# env.board[8, 0] = LEFT
# env.board[8, env.full_cols - 1] = RIGHT

env.set_cell(env.full_rows - 1, 8, Directions.DOWN, energy=-10)
env.set_cell(0, 8, Directions.UP, energy=-10)
env.set_cell(8, 0, Directions.LEFT, energy=-10)
env.set_cell(8, env.full_cols - 1, Directions.RIGHT, energy=-10)

plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots()
env.update_board()


cm = ListedColormap(['w', 'g', 'purple', 'b', 'y'], N=5)
ax_image = ax.matshow(env.get_board(), cmap=cm, vmin=0.0, vmax=4.0)
ax.grid(which="minor", lw=2.0)
ax.xaxis.set_minor_locator(MultipleLocator(base=1.0, offset=-0.5))
ax.yaxis.set_minor_locator(MultipleLocator(base=1.0, offset=-0.5))

X, Y = np.meshgrid(np.arange(0, env.full_rows), np.arange(0, env.full_cols))


def create_quiver(X, Y, env):
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    mask = np.ones_like(U) == 1
    board = env.get_board()
    for row in range(env.full_rows):
        for col in range(env.full_cols):
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


(U, V, mask) = create_quiver(X, Y, env)
quiver = ax.quiver(X, Y, U, V)
# agent_rows = [row for (row, _) in env.occupied]
# agent_cols = [col for (_, col) in env.occupied]


agent_rows = []
agent_cols = []
for agent in env.agents:
    agent_rows.append(agent.row)
    agent_cols.append(agent.col)

data = np.stack([agent_rows, agent_cols]).T
scat = ax.scatter(agent_cols, agent_rows, color="red")


def fig_update(frame):
    # inv_temp = 0.85 + frame * 0.0025
    # env.inv_temp = inv_temp
    if frame % 2 == 0:
        env.update_board()
    else:
        env.update_agents()
    board = env.get_board()
    (U, V, mask) = create_quiver(X, Y, env)
    U = np.ma.array(U, mask=np.logical_not(mask))
    V = np.ma.array(V, mask=np.logical_not(mask))
    quiver.set_UVC(U, V)
    ax_image.set(data=board)

    agent_rows = []
    agent_cols = []
    for agent in env.agents:
        agent_rows.append(agent.row)
        agent_cols.append(agent.col)

    data = np.stack([agent_cols, agent_rows]).T
    scat.set_offsets(data)

    ax.set_title(r"Inv Temp={:.2f}, Done: {} ".format(
        env.inv_temp, env.is_done()))
    return ax_image


ani = animation.FuncAnimation(
    fig=fig, func=fig_update, frames=400, interval=100)
ani.save("32x32.gif")
