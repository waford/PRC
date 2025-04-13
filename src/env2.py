import numpy as np
import pdb
import copy


class Directions:
    NOP = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class AgentStates:
    AGENT_FREE = 0
    AGENT_OCCUPIED = 1


MAX_ENERGY = 50
IN_PATH_ENERGY = 2


class Cell:
    def __init__(self, row, col):
        self.direction = Directions.NOP
        self.energy = 0.0
        self.disapation = 0.8
        self.row = row
        self.col = col
        self.goal = False

    def is_aligned(self, other_cell):
        return self.direction == other_cell.direction

    def aligned_energy(self, neighbor, position):
        if neighbor.direction == position:
            return -IN_PATH_ENERGY
        else:
            return 0

    def update_direction(self, board):
        neighbors = self.get_neighbors(board)

        # if self.row == 1 and self.col == 8:
        #    pdb.set_trace()

        energies = np.zeros(5)
        for (i, neighbor) in enumerate(neighbors, start=1):
            if neighbor is not None:
                energies[i] = neighbor.energy
                # energies[i] += self.aligned_energy(neighbor, i)
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
        self.state = AgentStates.AGENT_FREE

    def desired_move(self, board):
        direction = board[self.row][self.col].direction
        match direction:
            case Directions.UP:
                return (self.row - 1, self.col)
            case Directions.RIGHT:
                return (self.row, self.col + 1)
            case Directions.DOWN:
                return (self.row + 1, self.col)
            case Directions.LEFT:
                return (self.row, self.col - 1)
            case Directions.NOP:
                return (self.row, self.col)

    def coord(self):
        return (self.row, self.col)

    def update_location(self, coord):
        self.row = coord[0]
        self.col = coord[1]


class Environment:

    def __init__(self, rows, cols, num_agents=1, inv_temp=0.0):
        self.rows = rows
        self.cols = cols
        self.full_rows = rows + 2
        self.full_cols = cols + 2
        self.inv_temp = inv_temp
        self.sample_fraction = 0.80
        self.agents = []

        self.free_board = [[Cell(row, col) for col in range(0, self.full_cols)]
                           for row in range(0, self.full_rows)]
        self.occupied_board = copy.deepcopy(self.free_board)

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
                board[row, col] = self.free_board[row][col].direction

        return board

    def set_cell(self, row, col, direction, energy=0.0, goal=False):
        self.free_board[row][col].direction = direction
        self.free_board[row][col].energy = energy
        self.free_board[row][col].goal = goal

    def update_board(self):
        new_free_board = copy.deepcopy(self.free_board)
        new_occupied_board = copy.deepcopy(self.occupied_board)
        for row in range(1, self.full_rows-1):
            for col in range(1, self.full_cols-1):
                new_free_board[row][col].update_direction(self.free_board)
                new_occupied_board[row][col].update_direction(
                    self.occupied_board)

        self.free_board = new_free_board
        self.occupied_board = new_occupied_board

    def is_boundary(self, coord):
        row = coord[0]
        col = coord[1]
        return (row == 0) or (row == self.full_rows - 1) or (col == 0) or (col == self.full_cols - 1)

    def is_goal(self, coord):
        return self.free_board[coord[0]][coord[1]].goal

    def get_cell(self, coord):
        return self.free_board[coord[0]][coord[1]]

    def is_north(self, coord):
        return coord[0] == 0

    def is_west(self, coord):
        return coord[1] == 0

    def is_south(self, coord):
        return coord[0] == self.full_rows - 1

    def is_east(self, coord):
        return coord[1] == self.full_cols - 1

    def random_goal(self):
        rand_row = np.random.choice(self.full_rows - 1) + 1
        rand_col = np.random.choice(self.full_cols - 1) + 1

        options = [(0, rand_col), (self.full_rows - 1, rand_col),
                   (rand_row, 0), (rand_row, self.full_cols - 1)]
        goal_coord = options[np.random.choice(len(options))]
        cell = self.get_cell(goal_coord)
        if goal_coord[0] == 0:
            cell.direction = Directions.UP
        elif goal_coord[0] == self.full_rows-1:
            cell.direction = Directions.DOWN
        elif goal_coord[1] == 0:
            cell.direction = Directions.RIGHT
        elif goal_coord[1] == self.full_cols - 1:
            cell.direction = Directions.LEFT

        cell.energy = -10.0
        cell.goal = True

    def update_agents(self):
        occupied = set()
        for agent in self.agents:
            occupied.add((agent.row, agent.col))

        for agent in self.agents:
            current_pos = agent.coord()
            if self.is_goal(current_pos):
                agent.state = AgentStates.AGENT_OCCUPIED
                cell = self.get_cell(current_pos)
                cell.energy = 0.0
                cell.goal = False
                if self.is_north(current_pos):
                    cell.direction = Directions.DOWN
                elif self.is_east(current_pos):
                    cell.direction = Directions.LEFT
                elif self.is_south(current_pos):
                    cell.direction = Directions.UP
                elif self.is_west(current_pos):
                    cell.direction = Directions.RIGHT
                continue

            move = agent.desired_move(self.free_board)
            if (not self.is_boundary(move) or self.is_goal(move)) and (move not in occupied):
                old_location = agent.coord()
                agent.update_location(move)
                occupied.remove(old_location)
                occupied.add(move)

    def is_done(self):
        return False
