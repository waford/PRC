import numpy as np
from multiprocessing import Pool
from environment import Environment, UP, RIGHT, DOWN, LEFT

ROWS = 16
COLS = 16

top_border = [(0, col) for col in range(1, COLS)]
bottom_border = [(ROWS, col) for col in range(1, COLS)]
left_border = [(row, 0) for row in range(1, ROWS)]
right_border = [(row, COLS) for row in range(1, ROWS)]


inv_temps = np.linspace(0.0, 4.0, endpoint=False, num=16)
SAMPLES = 100
TIMESTEPS = 2 * 256


def run_experiments(inv_temp):
    env = Environment(ROWS, COLS, inv_temp=inv_temp)
    side_ind = np.random.choice(4)
    rand_ind = np.random.choice(ROWS) + 1
    if side_ind == 0:
        env.board[0, rand_ind] = UP
    elif side_ind == 1:
        env.board[rand_ind, COLS + 1] = RIGHT
    elif side_ind == 2:
        env.board[ROWS + 1, rand_ind] = DOWN
    elif side_ind == 3:
        env.board[rand_ind, 0] = LEFT

    iteration = 0
    while not env.is_done() and iteration < TIMESTEPS:
        env.update_board()
        env.update_agents()
        iteration += 1

    return (env.is_done(), iteration)


completed_results = []
time_results = []

for i in range(SAMPLES):
    print("Iteration: {}".format(i))
    with Pool() as pool:
        np.random.seed()
        results = pool.map(run_experiments,  inv_temps)
        completed_results.append([done for (done, _) in results])
        time_results.append([time for (_, time) in results])

completed_results = np.array(completed_results)
time_results = np.array(time_results)

np.savetxt("completed_results.txt", completed_results)
np.savetxt("time_results.txt", time_results)
