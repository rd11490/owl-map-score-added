import pandas as pd
import matplotlib.pyplot as plt
from utils.constants import Teams

# Pandas options for better printing

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)

final_tables_west = pd.read_csv('../results/cc_final_sim_results_west.csv')
final_tables_east = pd.read_csv('../results/cc_final_sim_results_east.csv')


def label_bars(rects, num_simulations):
    for rect in rects:
        height = rect.get_width()
        y = round(height * num_simulations)
        if y > 0:
            plt.text(height + 0.005, rect.get_y() + rect.get_height() / 2, y, fontweight='bold', ha='left', va='center')


def calc_ticks(hist):
    min_val = min(hist.index)
    max_val = max(hist.index)
    return list(range(min_val, max_val + 1))


def plot_team(frame, team, division_size):
    print(team)
    sims = frame.shape[0]
    point_hist = frame['league_points'].value_counts().sort_index()
    wins_hist = frame['wins'].value_counts().sort_index()
    rank_hist = frame['rank'].value_counts().sort_index()

    color = Teams.TeamColors[team]

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle('{} Simulated Season Results ({} Simulations)'.format(team, sims), fontsize=20, y=.95)

    plt.subplot(3, 1, 1)
    rects = plt.barh(rank_hist.index, rank_hist.values / sims, color=color)
    label_bars(rects, sims)
    plt.ylabel('Final Standing')
    plt.xlim((0, 1))
    ticks = list(range(1, division_size+1))
    plt.yticks(ticks)
    plt.gca().invert_yaxis()


    plt.subplot(3, 1, 2)
    rects = plt.barh(point_hist.index, point_hist.values / sims, color=color)
    label_bars(rects, sims)
    plt.ylabel('League Points')
    plt.xlim((0, 1))
    ticks = calc_ticks(point_hist)
    plt.yticks(ticks)


    plt.subplot(3, 1, 3)
    rects = plt.barh(wins_hist.index, wins_hist.values / sims, color=color)
    label_bars(rects, sims)
    plt.ylabel('Wins')
    plt.xlim((0, 1))
    plt.xlabel('Frequency')

    ticks = calc_ticks(wins_hist)
    plt.yticks(ticks)

    plt.savefig('plots/{}'.format(team))

    return 100 * rank_hist / sims


west_ranks = []
east_ranks = []

for team in Teams.West:
    team_results = final_tables_west[final_tables_west['team'] == team][['league_points', 'wins', 'rank']]
    ranks = plot_team(team_results, team, len(Teams.West))
    ranks['team'] = team
    west_ranks.append(ranks)

for team in Teams.East:
    team_results = final_tables_east[final_tables_east['team'] == team][['league_points', 'wins', 'rank']]
    ranks = plot_team(team_results, team, len(Teams.East))
    ranks['team'] = team
    east_ranks.append(ranks)

west_ranks = pd.DataFrame(west_ranks)
east_ranks = pd.DataFrame(east_ranks)

west_ranks = west_ranks.fillna(0.0)
west_columns = ['team', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
west_ranks = west_ranks[west_columns]
west_ranks = west_ranks.sort_values(by=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ascending=False)
print(west_ranks)

east_ranks = east_ranks.fillna(0.0)
east_columns = ['team', 1, 2, 3, 4, 5, 6, 7, 8]
east_ranks = east_ranks[east_columns]
east_ranks = east_ranks.sort_values(by=[1, 2, 3, 4, 5, 6, 7, 8], ascending=False)

print(east_ranks)