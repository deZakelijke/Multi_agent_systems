import sys
import numpy as np
import copy


def choose_strategy(payoff_matrix, history):
    try:
        opponent_defect_ratio = history[1] / sum(history)
    except ZeroDivisionError:
        opponent_defect_ratio = 0.5

    silent_utility = payoff_matrix[0, 0][0] * (1 - opponent_defect_ratio) + \
                     payoff_matrix[0, 1][0] * opponent_defect_ratio
    defect_utility = payoff_matrix[1, 0][0] * (1 - opponent_defect_ratio) + \
                     payoff_matrix[1, 1][0] * opponent_defect_ratio
    return np.random.binomial(1, silent_utility / (silent_utility + defect_utility))

def play_game(payoff_matrix, strategy_1, strategy_2):
    result = copy.deepcopy(payoff_matrix[strategy_1, strategy_2])
    result[0] += np.random.normal(0, 0.5)
    result[1] += np.random.normal(0, 0.5)
    return result

def infer_opponent_strategy(payoff_matrix, strategy, reward):
    possible_row = payoff_matrix[strategy]
    difference_list = []
    for i in range(len(possible_row)):
        difference_list.append(abs(possible_row[i][0] - reward))
    return difference_list.index(min(difference_list))

if __name__ == "__main__":
    payoff_matrix = np.array([[(-1, -1), (-12, 0)],
                               [(0, -12), (-8, -8)]])
    history_player_1 = [0, 0]
    history_player_2 = [0, 0]
    games = 10000
    for _ in range(games):
        strategy_1 = choose_strategy(payoff_matrix, history_player_2)
        strategy_2 = choose_strategy(payoff_matrix, history_player_1)

        result = play_game(payoff_matrix, strategy_1, strategy_2)

        inferred_strategy_player_2 = infer_opponent_strategy(payoff_matrix, strategy_1, result[0])
        inferred_strategy_player_1 = infer_opponent_strategy(payoff_matrix, strategy_2, result[1])

        history_player_1[inferred_strategy_player_1] += 1
        history_player_2[inferred_strategy_player_2] += 1

    print(history_player_1)
    print(history_player_2)
