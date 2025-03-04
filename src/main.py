import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.animation as animation
from multiprocessing import Pool
from environment import Environment, NOP, UP, RIGHT, DOWN, LEFT

env = Environment(16, 16, inv_temp=1.7)
env.board[0, 1:(env.full_cols-1)] = NOP
env.board[1:(env.full_rows - 1), -1] = NOP
env.board[-1, 1:(env.full_cols-1)] = NOP
env.board[1:(env.full_rows - 1), 0] = NOP

env.board[0, 8] = UP


plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots()
env.update_board()

ax_image = ax.matshow(env.board, cmap="Blues")
ax.grid(which="minor", lw=2.0)
ax.xaxis.set_minor_locator(MultipleLocator(base=1.0, offset=-0.5))
ax.yaxis.set_minor_locator(MultipleLocator(base=1.0, offset=-0.5))

X, Y = np.meshgrid(np.arange(0, env.full_rows), np.arange(0, env.full_cols))


def create_quiver(X, Y):
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    mask = np.ones_like(U) == 1
    for row in range(env.full_rows):
        for col in range(env.full_cols):
            value = env.board[row, col]
            if value == NOP:
                mask[row, col] = False
            elif value == UP:
                U[row, col] = 0
                V[row, col] = 1
            elif value == RIGHT:
                U[row, col] = 1
                V[row, col] = 0
            elif value == DOWN:
                U[row, col] = 0
                V[row, col] = -1
            elif value == LEFT:
                U[row, col] = -1
                V[row, col] = 0
    return (U, V, mask)


(U, V, mask) = create_quiver(X, Y)
quiver = ax.quiver(X, Y, U, V)
agent_rows = [row for (row, _) in env.occupied]
agent_cols = [col for (_, col) in env.occupied]

data = np.stack([agent_rows, agent_cols]).T
scat = ax.scatter(agent_cols, agent_rows, color="red")


def fig_update(frame):
    # inv_temp = 0.85 + frame * 0.0025
    # env.inv_temp = inv_temp
    if frame % 2 == 0:
        env.update_board()
    else:
        env.update_agents()
    (U, V, mask) = create_quiver(X, Y)
    U = np.ma.array(U, mask=np.logical_not(mask))
    V = np.ma.array(V, mask=np.logical_not(mask))
    quiver.set_UVC(U, V)
    ax_image.set(data=env.board.astype(int))
    agent_rows = [row for (row, _) in env.occupied]
    agent_cols = [col for (_, col) in env.occupied]

    data = np.stack([agent_cols, agent_rows]).T
    scat.set_offsets(data)

    ax.set_title(r"Inv Temp={:.2f}".format(env.inv_temp))
    return ax_image


ani = animation.FuncAnimation(
    fig=fig, func=fig_update, frames=400, interval=100)
ani.save("test.gif")
