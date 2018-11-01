import numpy as np
import matplotlib.pyplot as plt

class Gridworld:

    def __init__(self, alpha=0.03, gamma=0.9):
        # Create world as grid. Each location on the grid is a tuple (reward, type).
        # type: 1 is non-terminal, 0 is wall, -1 is terminal
        f = (-1, 1)
        w = (-1, 0)
        s = (-20, -1)
        t = (10, -1)
        self.alpha = alpha
        self.gamma = gamma
        self.world = np.array([[f, f, f, f, f, f, f, f],
                               [f, f, w, w, w, w, f, f],
                               [f, f, f, f, f, w, f, f],
                               [f, f, f, f, f, w, f, f],
                               [f, f, f, f, f, w, f, f],
                               [f, f, f, f, s, f, f, f],
                               [f, w, w, w, f, f, f, f],
                               [f, f, f, f, f, f, f, t]])
        self.state_values = np.zeros(self.world.shape[:-1])

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


    def random_move(self, location):
        possible_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        direction = possible_directions[np.random.choice(len(possible_directions))]
        new_location = tuple(map(sum, zip(location, direction)))
        if new_location[0] < 0 or new_location[0] >= self.world.shape[0]:
            return location
        if new_location[1] < 0 or new_location[1] >= self.world.shape[1]:
            return location
        if not self.valid_location(new_location):
            return location
        return new_location

    def update_values(self, location, new_location, reward):
        old_value = self.state_values[location[:]]
        next_value = self.state_values[new_location[:]]
        new_value = old_value + self.alpha * (reward + self.gamma * next_value - old_value)
        self.state_values[location[:]] = new_value

    def random_walk(self):
        dims = self.world.shape
        location = np.random.uniform((0, 0), (dims[0], dims[1]), 2)
        location = (int(location[0]), int(location[1]))
        while not self.valid_location(location):
            location = np.random.uniform((0, 0), (dims[0], dims[1]), 2)
            location = (int(location[0]), int(location[1]))

        while not self.terminal_location(location):
            new_location = self.random_move(location)
            reward = self.world[new_location[:]][0]
            self.update_values(location, new_location, reward)
            location = new_location

    def display_state_values(self):
        fig, ax = plt.subplots()
        im = ax.imshow(self.state_values)
        for i in range(self.state_values.shape[0]):
            for j in range(self.state_values.shape[1]):
                test = ax.text(j, i, f"{self.state_values[i, j]:.2f}",
                               ha="center", va="center", color="w")
        plt.show()

if __name__ == "__main__":
    iterations = 10000
    alpha = 0.3
    gamma = 0.5
    world = Gridworld(alpha, gamma)
    for _ in range(iterations):
        world.random_walk()
    world.display_state_values()
