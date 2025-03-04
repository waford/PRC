import numpy as np


NOP = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4


class Environment:

    def __init__(self, rows, cols, num_agents=1, inv_temp=0.0):
        self.rows = rows
        self.cols = cols
        self.full_rows = rows + 2
        self.full_cols = cols + 2
        self.inv_temp = inv_temp

        self.board = np.zeros((self.full_rows, self.full_cols))
        agent_coords_flat = np.random.choice(
            self.rows * self.cols, size=num_agents, replace=False)
        agent_coords = np.unravel_index(
            agent_coords_flat, (self.rows, self.cols))

        self.occupied = set()
        for (row, col) in zip(agent_coords[0], agent_coords[1]):
            row = row + 1
            col = col + 1
            self.occupied.add((row, col))

        self.coords = [(row, col) for row in range(1, self.full_rows - 1)
                       for col in range(1, self.full_cols - 1)]

    def neighbors(self, coord):
        row = coord[0]
        col = coord[1]
        return [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]

    def is_available(self, coord):
        row = coord[0]
        col = coord[1]
        is_boundary = (row <= 0 or row >= self.full_rows - 1) or (
            col <= 0 or col >= self.full_cols - 1)
        is_occupied = (row, col) in self.occupied

        return not (is_occupied or is_boundary)

    def random_walk(self, coord):
        neighbors = self.neighbors(coord)
        valid_movements = [
            neighbor for neighbor in neighbors if self.is_available(neighbor)]

        valid_movements.append(coord)
        next_movement_ind = np.random.choice(len(valid_movements))
        return valid_movements[next_movement_ind]

    def potts_model_motion(self, coord):
        row = coord[0]
        col = coord[1]
        row_inds = row + np.array([-1, 0, 1, 0])
        col_inds = col + np.array([0, 1, 0, -1])

        values = self.board[row_inds, col_inds]

        """
            For now only consider monochromatic edges.
            Each monochromatic edge decreases the energy of the system.
        """
        energies = np.ones(5)
        for i in range(0, 5):
            energies[i] = -np.sum(values == i)

        probs = np.exp(-self.inv_temp * energies)
        probs = probs / np.sum(probs)

        value = np.random.choice(5, p=probs)
        return value

    def potts_model_no_nop(self, coord):
        row = coord[0]
        col = coord[1]
        row_inds = row + np.array([-1, 0, 1, 0])
        col_inds = col + np.array([0, 1, 0, -1])
        values = self.board[row_inds, col_inds]

        energies = -np.zeros(5)
        MAX_ENERGY = np.log(np.finfo(np.float64).max)
        for i in range(1, 5):
            energies[i] = -np.sum(values == i)
            if i == UP and row == 1 and values[0] != UP:
                energies[i] += MAX_ENERGY
            elif i == RIGHT and col == self.full_cols - 2 and values[1] != RIGHT:
                energies[i] += MAX_ENERGY
            elif i == DOWN and row == self.full_rows - 2 and values[2] != DOWN:
                energies[i] += MAX_ENERGY
            elif i == LEFT and row == 0 and values[3] != LEFT:
                energies[i] += MAX_ENERGY

            if i == UP and values[0] == UP:
                energies[i] -= 20
            elif i == RIGHT and values[1] == RIGHT:
                energies[i] -= 20
            elif i == DOWN and values[2] == DOWN:
                energies[i] -= 20
            elif i == LEFT and values[2] == LEFT:
                energies[i] -= 20

        energies[0] = 0

        probs = np.exp(-self.inv_temp * energies)
        probs = probs / np.sum(probs)

        value = np.random.choice(5, p=probs)
        return value

    def update_multicore(self):
        with Pool() as pool:
            values = pool.map(self.potts_model_motion, self.coords)
            values = np.array(values).reshape(self.rows, self.cols)
            # new_board[row, col] = self.potts_model_no_nop((row, col))
            self.board[1:self.full_rows - 1, 1:self.full_cols - 1] = values

    def update_board(self):
        values = self.board.copy()
        for coord in self.coords:
            values[coord[0], coord[1]] = self.potts_model_no_nop(coord)
        self.board = values

    def is_goal(self, coord):
        row = coord[0]
        col = coord[1]
        if (row == 0 and col == 8):
            return True

    def update_agents(self):
        new_occupied = set()
        for (row, col) in self.occupied:
            value = self.board[row, col]
            new_row = row
            new_col = col

            if value == UP:
                new_row -= 1
            elif value == RIGHT:
                new_col += 1
            elif value == DOWN:
                new_row += 1
            elif value == LEFT:
                new_col -= 1
            else:
                new_occupied.add((row, col))
                continue

            new_coord = (new_row, new_col)
            if self.is_available(new_coord) or self.is_goal(new_coord):
                new_occupied.add(new_coord)
            else:
                new_occupied.add((row, col))

        self.occupied = new_occupied
