import pandas as pd

from predict_functions import build_rmsa_map, predict_match
from utils.constants import Maps, Teams
import matplotlib.pyplot as plt

rmsa_frame = pd.read_csv('results/rmsa.csv')

rmsa_map = build_rmsa_map(rmsa_frame)

week_one_rotation = [Maps.Control, Maps.Hybrid, Maps.Escort, Maps.Assault, Maps.Control]
week_two_rotation = [Maps.Control, Maps.Escort, Maps.Assault, Maps.Hybrid, Maps.Control]
week_three_rotation = [Maps.Control, Maps.Assault, Maps.Hybrid, Maps.Escort, Maps.Control]
finals_rotation = [Maps.Control, Maps.Assault, Maps.Hybrid, Maps.Escort, Maps.Control, Maps.Hybrid, Maps.Escort]

###
# Change the values below
###
team_one = Teams.Gladiators
team_two = Teams.Hunters
map_order = finals_rotation
maps_to_win = 4
num_simulations = 10000


######
#
######


def build_map_diff_dict(maps_to_win):
    possible_loser_scores = list(range(0, maps_to_win))
    combos = []
    dict = {}
    for score in possible_loser_scores:
        combos.append(score - maps_to_win)
        combos.append(maps_to_win - score)
    combos.sort()
    for c in combos:
        dict[c] = 0
    return dict


def ticks_to_labels(ticks, maps_to_win):
    labels = []
    for t in ticks:
        if t < 0:
            labels.append('{}-{}'.format(maps_to_win, maps_to_win + t))
        else:
            labels.append('{}-{}'.format(maps_to_win - t, maps_to_win))
    return labels


def bar_colors(ticks, team_one, team_two):
    color = []
    for t in ticks:
        if t < 0:
            color.append(Teams.TeamColors[team_one])
        else:
            color.append(Teams.TeamColors[team_two])
    return color

def convert_average_diff_to_score(diff):
    diff = round(diff, 1)
    if diff < 0:
        return maps_to_win, round(maps_to_win + diff,1)
    else:
        return round(maps_to_win - diff, 1), maps_to_win

def plot_histogram(results_arr, maps_to_win):
    map_diff = build_map_diff_dict(maps_to_win)
    team_one = results_arr[0]['team_one']
    team_two = results_arr[0]['team_two']

    differential_arr = []
    for r in results_arr:
        differential = r['team_one_map_wins'] - r['team_two_map_wins']
        differential_arr.append(differential)
        map_diff[differential] += 1
    average_differential = sum(differential_arr)/len(differential_arr)
    team_two_ewins, team_one_ewins = convert_average_diff_to_score(average_differential)

    frequency = [v / num_simulations for v in map_diff.values()]
    xvalues = list(map_diff.keys())
    xvalues.reverse()

    color = bar_colors(xvalues, team_one, team_two)

    plt.figure(figsize=(8, 5))
    rects = plt.bar(xvalues, frequency, color=color)
    labels = ticks_to_labels(xvalues, maps_to_win)

    plt.xticks(ticks=xvalues, labels=labels)
    plt.title('{}:{}  {}:{} ({} Simulations)'.format(team_one, team_one_ewins, team_two, team_two_ewins, num_simulations))
    plt.xlabel('{} - {}'.format(team_one, team_two))
    plt.ylabel('Frequency')
    plt.ylim((0, 1))

    for rect in rects:
        height = rect.get_height()
        y = round(height * num_simulations)
        if y > 0:
            plt.text(rect.get_x() + rect.get_width() / 2, height * 1.01, y, fontweight='bold', ha='center')
    filename = '-'.join('{}-{}'.format(team_one, team_two).split(' '))
    plt.savefig('./plots/countdown_cup/finals-{}'.format(filename))


results = []

for i in range(0, num_simulations):
    result = predict_match(team_one, team_two, map_order, rmsa_map, maps_to_win)
    result['sim'] = i
    results.append(result)

plot_histogram(results, maps_to_win)
