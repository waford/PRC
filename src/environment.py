import numpy as np
import pdb
from multiprocessing import Pool


NOP = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4
MAX_ENERGY = 100
IN_PATH_ENERGY = 50


class Environment:

    def __init__(self, rows, cols, num_agents=1, inv_temp=0.0):
        self.rows = rows
        self.cols = cols
        self.full_rows = rows + 2
        self.full_cols = cols + 2
        self.inv_temp = inv_temp
        self.sample_fraction = 0.80

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
        is_occupied = (row, col) in self.occupied

        return not (is_occupied or self.is_boundary(coord))

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

    def is_done(self):
        done = True
        for coord in self.occupied:
            done = done and (self.is_goal(coord) or self.is_boundary(coord))

        return done

    def potts_model_no_nop(self, cord_ind):
        cord = self.coords[cord_ind]
        row = cord[0]
        col = cord[1]
        row_inds = row + np.array([-1, 0, 1, 0])
        col_inds = col + np.array([0, 1, 0, -1])
        values = self.board[row_inds, col_inds]

        energies = -np.zeros(5)
        for i in range(0, 5):
            energies[i] = -np.sum(values == i)
            # if i == UP and row == 1 and values[0] != UP:
            #    energies[i] += IN_PATH_ENERGY
            # elif i == RIGHT and col == self.full_cols - 2 and values[1] != RIGHT:
            #    energies[i] += IN_PATH_ENERGY
            # elif i == DOWN and row == self.full_rows - 2 and values[2] != DOWN:
            #    energies[i] += IN_PATH_ENERGY
            # elif i == LEFT and col == 1 and values[3] != LEFT:
            #    energies[i] += IN_PATH_ENERGY

            if i == UP and values[0] == UP:
                energies[i] -= IN_PATH_ENERGY
            elif i == RIGHT and values[1] == RIGHT:
                energies[i] -= IN_PATH_ENERGY
            elif i == DOWN and values[2] == DOWN:
                energies[i] -= IN_PATH_ENERGY
            elif i == LEFT and values[3] == LEFT:
                energies[i] -= IN_PATH_ENERGY

            if i == UP and (values[0] == RIGHT or values[0] == LEFT):
                energies[i] -= 0.05 * IN_PATH_ENERGY
            elif i == RIGHT and (values[1] == UP or values[1] == DOWN):
                energies[i] -= 0.05 * IN_PATH_ENERGY
            elif i == DOWN and (values[2] == RIGHT or values[2] == LEFT):
                energies[i] -= 0.05 * IN_PATH_ENERGY
            elif i == LEFT and (values[3] == UP or values[3] == DOWN):
                energies[i] -= 0.05 * IN_PATH_ENERGY

            if i == UP and row == 1 and values[0] == UP:
                energies[i] = -MAX_ENERGY
            elif i == RIGHT and col == self.full_cols - 2 and values[1] == RIGHT:
                energies[i] = -MAX_ENERGY
            elif i == DOWN and row == self.full_rows - 2 and values[2] == DOWN:
                energies[i] = -MAX_ENERGY
            elif i == LEFT and col == 1 and values[3] == LEFT:
                energies[i] = -MAX_ENERGY

        probs = np.exp(-self.inv_temp * energies)
        probs = probs / np.sum(probs)

        value = np.random.choice(5, p=probs)
        return value

    def value_at(self, coord):
        return self.board[coord[0], coord[1]]

    def update_boundary(self):
        for col in range(1, self.full_cols-1):
            for row in [0, self.full_rows-1]:
                if self.is_goal((row, col)):
                    continue
                left_coord = (row, col-1)
                right_coord = (row, col+1)

                left_energy = 0.0
                right_energy = 0.0

                if self.is_goal(left_coord):
                    left_energy -= MAX_ENERGY

                if self.is_goal(right_coord):
                    right_energy -= MAX_ENERGY

                if self.value_at(left_coord):
                    left_energy -= IN_PATH_ENERGY

                if self.value_at(right_coord):
                    right_energy -= IN_PATH_ENERGY

                probs = np.exp(-self.inv_temp *
                               np.array([left_energy, right_energy]))
                probs = probs / np.sum(probs)
                self.board[row, col] = np.random.choice([LEFT, RIGHT], p=probs)

        for row in range(1, self.full_rows-1):
            for col in [0, self.full_cols-1]:
                if self.is_goal((row, col)):
                    continue
                down_coord = (row+1, col)
                up_coord = (row-1, col)

                down_energy = 0.0
                up_energy = 0.0

                if self.is_goal(down_coord):
                    down_energy -= MAX_ENERGY

                if self.is_goal(up_coord):
                    up_energy -= MAX_ENERGY

                if self.value_at(down_coord):
                    down_energy -= IN_PATH_ENERGY

                if self.value_at(up_coord):
                    up_energy -= IN_PATH_ENERGY

                probs = np.exp(-self.inv_temp *
                               np.array([down_energy, up_energy]))
                probs = probs / np.sum(probs)
                self.board[row, col] = np.random.choice([DOWN, UP], p=probs)

    def update_board(self):
        self.update_boundary()

        values = self.board.copy()
        num_coords = len(self.coords)
        coord_inds = np.random.choice(
            num_coords, size=np.round(num_coords * self.sample_fraction).astype(int), replace=False)
        # for coord_ind in coord_inds:
        #    coord = self.coords[coord_ind]
        #    values[coord[0], coord[1]] = self.potts_model_no_nop(coord)

        with Pool() as p:
            new_values = p.map(self.potts_model_no_nop, coord_inds)

        for (i, coord_ind) in enumerate(coord_inds):
            coord = self.coords[coord_ind]
            values[coord[0], coord[1]] = new_values[i]

        for coord in self.occupied:
            if self.is_goal(coord):
                values[coord[0], coord[1]] = NOP

        self.board = values

    def is_boundary(self, coord):
        row = coord[0]
        col = coord[1]
        return (row <= 0 or row >= self.full_rows - 1) or (
            col <= 0 or col >= self.full_cols - 1)

    def is_goal(self, coord):
        row = coord[0]
        col = coord[1]
        is_north_goal = row == 0 and self.board[row, col] == UP
        is_east_goal = col == self.full_cols - \
            1 and self.board[row, col] == RIGHT
        is_south_goal = row == self.full_rows - \
            1 and self.board[row, col] == DOWN
        is_west_goal = col == 0 and self.board[row, col] == LEFT

        return is_north_goal or is_east_goal or is_south_goal or is_west_goal

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

            if value == NOP or self.is_goal((row, col)):
                new_occupied.add((row, col))
                continue

            new_coord = (new_row, new_col)
            if (self.is_available(new_coord) or self.is_goal(new_coord)) and (not new_coord in new_occupied):
                new_occupied.add(new_coord)
            else:
                new_occupied.add((row, col))

        if len(self.occupied) != len(new_occupied):
            pdb.set_trace()
        self.occupied = new_occupied
