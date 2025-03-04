import numpy as np
from environment import *

ROWS = 16
COLS = 16

env = Environment(ROWS, COLS, inv_temp=0.0)
top_border = [(0, col) for col in range(1, COLS)]
bottom_border = [(ROWS, col) for col in range(1, COLS)]
left_border = [(row, 0) for row in range(1, ROWS)]
right_border = [(row, COLS) for row in range(1, ROWS)]

boundary_conditions = np.array((top_border, bottom_border, left_border, right_border)).flatten(
).reshape(2 * ROWS + 2 * COLS - 4, 2)
