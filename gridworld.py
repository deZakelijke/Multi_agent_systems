import numpy as np
import matplotlib.pyplot as plt

class Gridworld:

    def __init__(self, alpha=0.03, gamma=0.9, epsilon=0.1):
        # Create world as grid. Each location on the grid is a tuple (reward, type).
        # type: 1 is non-terminal, 0 is wall, -1 is terminal
        f = (-1, 1)
        w = (-1, 0)
        s = (-20, -1)
        t = (10, -1)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.world = np.array([[f, f, f, f, f, f, f, f],
                               [f, f, w, w, w, w, f, f],
                               [f, f, f, f, f, w, f, f],
                               [f, f, f, f, f, w, f, f],
                               [f, f, f, f, f, w, f, f],
                               [f, f, f, f, s, f, f, f],
                               [f, w, w, w, f, f, f, f],
                               [f, f, f, f, f, f, f, t]])
        self.state_action_values = np.zeros((4, *self.world.shape[:-1]))

    def terminal_location(self, location):
        if self.world[location[:]][1] == -1:
            return True
        else:
            return False

    def valid_location(self, location):
        if self.world[location[:]][1]:
            return True
        else:
            return False

    def greedy_move(self, location):
        possible_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        m = max(self.state_action_values[:, location[0], location[1]])
        best_directions = [i for i,v in enumerate(self.state_action_values[:, location[0], location[1]]) if v==m]
        if len(best_directions) == 1:
            direction = possible_directions[best_directions[0]]
        else:
            direction = possible_directions[np.random.choice(best_directions)]

        return self.resolve_new_move(location, direction), possible_directions.index(direction)

    def random_move(self, location):
        possible_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        direction_index = np.random.choice(len(possible_directions))
        direction = possible_directions[direction_index]
        return self.resolve_new_move(location, direction), direction_index

    def resolve_new_move(self, location, direction):
        new_location = tuple(map(sum, zip(location, direction)))
        if  new_location[0] < 0 or new_location[0] >= self.world.shape[0] or \
            new_location[1] < 0 or new_location[1] >= self.world.shape[1]:
            return location
        if not self.valid_location(new_location):
            return location
        return new_location


    def update_values(self, location, new_location, reward, direction):
        old_value = self.state_action_values[direction, location[0], location[1]]
        next_value = max(self.state_action_values[:, new_location[0], new_location[1]])
        new_value = old_value + self.alpha * (reward + self.gamma * next_value - old_value)
        self.state_action_values[direction, location[0], location[1]] = new_value

    def random_walk(self):
        dims = self.world.shape
        location = np.random.uniform((0, 0), (dims[0], dims[1]), 2)
        location = (int(location[0]), int(location[1]))
        while not self.valid_location(location):
            location = np.random.uniform((0, 0), (dims[0], dims[1]), 2)
            location = (int(location[0]), int(location[1]))

        while not self.terminal_location(location):
            if np.random.binomial(1, self.epsilon):
                new_location, direction = self.random_move(location)
            else:
                new_location, direction = self.greedy_move(location)
            reward = self.world[new_location[:]][0]
            self.update_values(location, new_location, reward, direction)
            location = new_location

    def display_state_values(self):
        fig, ax = plt.subplots()
        best_values = np.max(self.state_action_values, axis=0)
        im = ax.imshow(best_values)
        for i in range(best_values.shape[0]):
            for j in range(best_values.shape[1]):
                if self.world[i, j][1] == 1:
                    test = ax.text(j, i, f"{best_values[i, j]:.2f}",
                               ha="center", va="center", color="w")
                elif self.world[i, j][1] == 0:
                    test = ax.text(j, i, f"###\n###\n###",
                               ha="center", va="center", color="w")
                elif self.world[i, j][1] == -1:
                    test = ax.text(j, i, f"T",
                               ha="center", va="center", color="w")

        plt.show()

    def display_state_actions(self):
        possible_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        best_dirs = np.argmax(self.state_action_values, axis=0)
        print(best_dirs)
        dims = self.state_action_values.shape
        x_arrow = np.zeros(dims[1:])
        y_arrow = np.zeros(dims[1:])
        for i in range(dims[1]):
            for j in range(dims[2]):
                best_dir = possible_directions[best_dirs[i, j]]
                x_arrow[i, j] = best_dir[1]
                y_arrow[i, j] = best_dir[0] * -1
        y_arrow = np.flip(y_arrow, 0)
        x_arrow = np.flip(x_arrow, 0)
        plt.quiver(x_arrow, y_arrow, color='black')
        plt.show()

    def display_combined(self):
        pass

if __name__ == "__main__":
    iterations = 5000
    alpha = 0.01
    gamma = 0.9
    world = Gridworld(alpha, gamma)
    for _ in range(iterations):
        world.random_walk()
    world.display_state_values()
    world.display_state_actions()
