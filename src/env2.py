import numpy as np
import pdb

NOP = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4
MAX_ENERGY = 50
IN_PATH_ENERGY = 2

AGENT_FREE = 0
AGENT_OCCUPIED = 1


class Cell:
    def __init__(self, row, col):
        self.direction = NOP
        self.energy = 0.0
        self.disapation = 0.9
        self.row = row
        self.col = col

    def is_aligned(self, other_cell):
        return self.direction == other_cell.direction

    def aligned_energy(self, neighbor, position):
        if neighbor.direction == position:
            return -IN_PATH_ENERGY
        else:
            return 0

    def update_direction(self, board):
        neighbors = self.get_neighbors(board)

        #if self.row == 1 and self.col == 8:
        #    pdb.set_trace()

        energies = np.zeros(5)
        for (i, neighbor) in enumerate(neighbors, start=1):
            if neighbor is not None:
                energies[i] = neighbor.energy
                #energies[i] += self.aligned_energy(neighbor, i)
            else:
                energies[i] = MAX_ENERGY


        probs = np.exp(-energies)
        probs = probs / np.sum(probs)
        if np.isnan(probs).any():
            pdb.set_trace()

        randomized_direction = np.random.choice(5, p=probs)
        self.direction = randomized_direction
        self.energy = self.disapation * energies[randomized_direction]

    def get_neighbors(self, board):
        rows = len(board)
        cols = len(board[0])

        neighbors = []
        row = self.row
        col = self.col
        if row == 0:
            neighbors.append(None)
        else:
            neighbors.append(board[row - 1][col])

        if col == cols-1:
            neighbors.append(None)
        else:
            neighbors.append(board[row][col+1])

        if row == rows-1:
            neighbors.append(None)
        else:
            neighbors.append(board[row+1][col])

        if col == 0:
            neighbors.append(None)
        else:
            neighbors.append(board[row][col-1])

        return neighbors


class Agent:

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.state = AGENT_FREE


class Environment:

    def __init__(self, rows, cols, num_agents=1, inv_temp=0.0):
        self.rows = rows
        self.cols = cols
        self.full_rows = rows + 2
        self.full_cols = cols + 2
        self.inv_temp = inv_temp
        self.sample_fraction = 0.80
        self.agents = []

        self.board = [[Cell(row, col) for col in range(0, self.full_cols)]
                      for row in range(0, self.full_rows)]
        agent_coords_flat = np.random.choice(
            self.rows * self.cols, size=num_agents, replace=False)

        agent_coords = np.unravel_index(
            agent_coords_flat, (self.rows, self.cols))

        for i in range(len(agent_coords[0])):
            row = agent_coords[0][i]+1
            col = agent_coords[1][i]+1
            self.agents.append(Agent(row, col))

    def get_board(self):
        board = np.zeros((self.full_rows, self.full_cols))
        for row in range(0, self.full_rows):
            for col in range(0, self.full_cols):
                board[row, col] = self.board[row][col].direction

        return board

    def set_cell(self, row, col, direction, energy=0.0):
        self.board[row][col].direction = direction
        self.board[row][col].energy = energy

    def update_board(self):
        new_board = list(self.board)
        for row in range(1, self.full_rows-1):
            for col in range(1, self.full_cols-1):
                new_board[row][col].update_direction(self.board)

        self.board = new_board

    def update_agents(self):
        pass

    def is_done(self):
        return False
